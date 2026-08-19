"""多信号检索融合引擎
====================
三路并行融合（向量 + BM25 + 实体匹配）：
  - VECTOR:    向量相似度检索（FAISS）
  - BM25:      关键词检索（sklearn TfidfVectorizer IDF + BM25 评分）
  - ENTITY:    实体名称匹配（GraphLite GQL CONTAINS）

融合权重：vector=0.5, bm25=0.3, entity=0.2
时序衰减：score = score * (1 + 1/(1 + exp(-τ/60)))
去重策略：按 content[:100] 保留最高分

向后兼容降级链：
  L1 — HYPERGRAPH: 超图检索（GraphLite + FAISS 联合）
  L2 — VECTOR:     纯向量检索（FAISS-only）
  L3 — KEYWORD:    关键词检索（TF-IDF）
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.user_profile import profile_hit, profile_values
from config.settings import EntityExpansionConfig, ScopeRecallConfig, get_settings
from core.schema_distiller import extract_terms
from graph.common import CircuitBreakerOpen
from retrieval.hyde import generate_hypothesis
from retrieval.vector_store import FaissStore

from observability.logger import get_logger

logger = get_logger(__name__)


def _safe_float_tau(value) -> float:
    """GraphLite 真实引擎 NULL 序列化为字符串 'Null'（非 None），float('Null') 抛 ValueError。

    防御性解析：None/'Null'/'null'/''/非数字 → 0.0，其余转 float。
    修复 v5.50 真实引擎评测 bm25/entity 通道静默崩溃（bm25=0 entity=0）。
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "nan"):
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


class _EntityChannelDegraded(Exception):
    """entity 通道基础设施降级哨兵（private）。

    query_cypher 永不抛异常契约下，基础设施降级（熔断 open / 重试耗尽）表现为
    返回 []。_entity_match 读 thread-local 降级信号区分「正常无匹配」与「基础设施
    降级」，后者抛本哨兵 → _fusion_retrieve 现有 per-channel `except Exception`
    捕获并置 fusion_channel_skipped=True（降级信号不漏报）。
    """


# 【P0-1 实体-属性-时间】属性版本链检索通道常量（append 相对尾分缩放，严格低于种子）
_PROPERTY_BOOST = 0.6
_PROPERTY_MAX_RESULTS = 5
# 【P3c 跨消息多跳增强】实体扩召回通道常量（append max 锚 × boost 0.9，仅低于最高种子）
_ENTITY_EXPANSION_MAX_APPEND = 20  # 总 append 硬上限（实体 top-3 × 每实体 max-10 的钳制）
# 【阶段4-1 Schema 蒸馏】Schema 节点召回通道常量（append 相对尾分缩放，严格低于种子）
_SCHEMA_BOOST = 0.5
_SCHEMA_MAX_RESULTS = 3
# 【P3a R2】reranker 预热超时（秒）：CPU 冷加载 ~5-15s，60s 兜底防线程泄漏
_RERANK_PREWARM_TIMEOUT = 60.0
# 句首大写词误当实体的停用词（How/What/The/In ... 不是实体名）
_PROPERTY_CANDIDATE_STOPWORDS = frozenset({
    "how", "what", "when", "why", "where", "which", "who", "whose", "whom",
    "the", "this", "that", "these", "those", "and", "or", "but", "for", "with",
    "from", "into", "onto", "upon", "about", "than", "then", "there", "here",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her",
    "my", "your", "our", "their", "its", "in", "on", "at", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "please", "tell", "give",
    "show", "find", "search", "list", "explain", "describe", "summarize",
    "summarise", "what's", "how's", "don't", "doesn't", "didn't", "not",
    # 【R3 P2-1】常见介词/助动词补全——"of/to/has/had" 等小写英文词不再当伪实体
    "of", "to", "has", "had", "have", "by", "an", "as", "am",
    # 【R4 P2-1】let（let's 还原）/all（y'all 还原）不再当伪实体
    "let", "all",
})
# 【P0-1-R2 N5】查询属性词 → attr_name 匹配片段（中文属性词 → 英文 attr_name 同义词）
_PROPERTY_QUERY_TERM_MAP = {
    "收入": "revenue",
    "营收": "revenue",
    "市值": "market_cap",
    "估值": "valuation",
    "价格": "price",
    "金额": "amount",
    "利润": "profit",
    "销量": "sales",
    "销售额": "sales",
    "用户数": "users",
    "员工数": "employees",
    "收购": "acquired_value",
    "并购": "acquired_value",
    "投资": "invested_value",
}
# 【P1-3】英文属性词表（revenue/market_cap/income/age/occupation/salary 等）
# → attr_name 直接命中（意图分类先于检索，此时无 store 的 attr_names 可用）。
# "market_cap" 在查询中可能写作 "market cap"（空格），匹配时兼容两种写法。
_PROPERTY_EN_TERMS = frozenset({
    "revenue", "income", "salary", "age", "occupation", "market_cap",
    "valuation", "price", "amount", "profit", "sales", "users",
    "employees", "acquired_value", "invested_value",
})
# 【P1-3】时间词/年份数字不当实体（"What happened in 2023" 误判 event 根治）：
# 小写英文词（动词/时间词）已由候选提取器排除（仅取首字母大写专名 + 中文机构词），
# 此处再防首字母大写的时间词（Yesterday/Today）与 4 位年份数字混入实体。
_TIME_WORD_STOPWORDS = frozenset({
    "yesterday", "today", "tomorrow", "year", "month", "week", "day",
    "ago", "recent", "recently", "last", "next", "previous", "earlier",
    "later", "now", "current", "latest", "date", "time",
    "happened", "happen", "happens", "happening",
})
# 【P1-2】时间锚哨兵：计入 _AnchorSet.all 参与 seen_anchors 差集（时间锚也是有效
# 新锚点，仅时间锚触发下轮不被截断）；字符串哨兵不与实体/属性词名冲突。
_TIME_ANCHOR_SENTINEL = "__time_anchor__"

# 【R3 P2-1】常见英文缩写还原表：撇号缩写先还原为完整词再提取实体，
# 防 "don't" 被 \b[a-z]{2,}\b 切成 "don" 漏过滤（还原后 "do"/"not" 均在停用表）。
# 仅覆盖已知缩写；所有格（Apple's）不在表中，保持原样不影响专名提取。
_EN_CONTRACTION_MAP = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "can't": "can not", "couldn't": "could not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "mustn't": "must not", "it's": "it is", "that's": "that is",
    "what's": "what is", "there's": "there is", "here's": "here is",
    "who's": "who is", "he's": "he is", "she's": "she is",
    "i'm": "i am", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have",
    # 【R4 P2-1】补 let's/ain't/o'clock/y'all：防 "Let's find Apple" 切出 "Let" 伪实体
    "let's": "let us", "ain't": "is not", "o'clock": "of the clock",
    "y'all": "you all",
}


def _expand_contractions(text: str) -> str:
    """撇号缩写还原（大小写不敏感，弯撇号先归一），消除 "don't" 切出 "don" 的伪实体。"""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    for k, v in _EN_CONTRACTION_MAP.items():
        # 【R4 P3-1】\b 词边界防子串误替换（如 "she's" 内的 "he's" 被先替换）
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text, flags=re.IGNORECASE)
    return text

# 【User-Profile】内存常驻用户画像（app.py 启动时全量重建后经 set_user_profile
# 注入；单实例服务模块级共享，_deduplicate_and_sort 静态方法可直接读取）。
# 【P2-单租户语义】画像为模块级全局，检索加分/旁路 context 均不感知
# req.namespace——当前产品为单租户/单 Agent 部署，跨 namespace 画像泄露为
# 已知接受语义；多租户需将 _USER_PROFILE 改为 {namespace: profile} 键控。
_USER_PROFILE: dict = {}


def set_user_profile(profile: dict) -> None:
    """注入/更新用户画像（v5.39.0；启动全量重建后调用，内存常驻）。"""
    global _USER_PROFILE
    _USER_PROFILE = profile or {}


class RetrievalLevel(Enum):
    """三级检索级别 + 并行融合"""

    HYPERGRAPH = "hypergraph"  # L1: 超图检索（GraphLite + FAISS）
    VECTOR = "vector"  # L2: 纯向量检索（FAISS-only）
    KEYWORD = "keyword"  # L3: 关键词检索（TF-IDF）
    FUSION = "fusion"  # F: 三路并行融合（向量+BM25+实体匹配）


class RetrievalStrategy(str, Enum):
    TAU_FIRST = "tau_first"  # τ 值优先
    VECTOR_FIRST = "vector_first"  # 向量相似度优先
    HYBRID = "hybrid"  # 混合加权
    FUSION = "fusion"  # 三路并行融合加权


class FAISSUnavailable(Exception):
    """FAISS 不可用异常，触发降级到 KEYWORD"""

    pass


@dataclass
class QueryRouterConfig:
    """查询路由配置"""

    tau_weight: float = 0.4  # τ 值权重（混合模式）
    vector_weight: float = 0.6  # 向量相似度权重（混合模式）
    top_k_l1: int = 5  # L1 FAISS 检索 top-K (直接查 EpisodeNode)
    top_k_vector: int = 20  # L2 向量检索 top-K
    top_k_keyword: int = 20  # L3 关键词检索 top-K
    # 融合模式权重
    weight_fusion_vector: float = 0.35  # 向量检索权重（降权，短查询区分度不足）
    weight_fusion_bm25: float = 0.40  # BM25 关键词权重（提权，适合短关键字查询）
    weight_fusion_entity: float = 0.25  # 实体匹配权重（提权，专名精确匹配）
    # BM25 参数
    bm25_k1: float = 1.5  # BM25 k1 参数
    bm25_b: float = 0.75  # BM25 b 参数
    top_k_fusion: int = 30  # 融合检索总 top-K
    max_bm25_corpus: int = 50000  # BM25 索引最大语料数
    bm25_build_timeout: float = 30.0  # BM25 索引构建超时（秒），超时静默降级
    bm25_retry_cooldown: float = 30.0  # BM25 构建失败后重试冷却窗口（秒），避免失败后每次检索都全量重建
    # 图扩散配置（v5.26.0）
    graph_expansion_hop: int = 1  # 固定 1 跳（防图爆炸）
    graph_expansion_max: int = 20  # 扩散补充最大条数
    graph_expansion_alpha: float = 0.8  # 扩散新节点分数 = 归一化共现 × 向量尾分 × α
    # Agentic 检索配置（P0-2，默认关 = 单轮 FUSION 全路径，字节级向后兼容）
    agentic_enabled: bool = False  # 多步锚点检索编排开关（默认关）
    agentic_max_steps: int = 3  # 含首轮最多 3 轮（死循环硬上限）
    agentic_min_new: int = 3  # 每轮须新增 ≥3 条未见过锚点，否则提前停
    agentic_score_gap: float = 0.25  # 首轮 top-k 归一化分差 < 该值判证据不足
    agentic_top_k: int = 12  # 每轮召回 top-k（充分性/锚点提取窗口）
    # MESA 记忆增强检索配置（v5.49.0，默认关零回归；mesa_boost 由自演化同步）
    mesa_enabled: bool = False  # MESA 合成节点通道开关（默认关，与 community 默认开不同）
    mesa_boost: float = 0.4  # 合成分 = relevance × min(种子分) × mesa_boost（严格 < community boost 0.6）
    mesa_threshold: float = 0.5  # BM25-on-summary 相关度阈值（对齐 community_expansion）
    mesa_max_nodes: int = 5  # 每查询最多合成节点数
    # bge-reranker 重排配置（P3a，默认开；仅 FUSION 路径生效，失败静默降级）
    rerank_enabled: bool = True  # 重排开关（retrieve(rerank=False) 可显式关）
    rerank_input_k: int = 40  # 送入 reranker 的头部候选数（尾部保持原序 append）
    # HyDE 假设文档增强检索配置（P3b，默认关零回归；仅 FUSION 路径生效，失败静默降级单路）
    hyde_enabled: bool = False  # HyDE 开关（retrieve(hyde=True) 可显式开）
    hyde_mode: str = "dual"  # dual=原始+假设双路融合合并；replace=仅假设向量单路
    hyde_timeout: float = 1.5  # LLM 生成超时（秒）【P3b R1 P0-2】2.0→1.5：3s 检索预算内
    # 预留 1.5s LLM + 融合检索，失败静默降级单路
    # 实体扩召回配置（P3c v5.53.0 跨消息多跳增强，默认开；关闭/异常静默回落现状）
    entity_expansion: EntityExpansionConfig = field(default_factory=EntityExpansionConfig)
    # 图作用域召回配置（阶段3 v6.0.0；仅 overgraph 后端生效，graphlite hasattr 假 no-op）
    scope_recall: ScopeRecallConfig = field(default_factory=ScopeRecallConfig)

    def __post_init__(self) -> None:
        # 【P3b R1 P2】hyde_mode 枚举校验：镜像 RetrievalConfig 的做法——检索路径只分
        # dual/replace 两分支，非法值会静默落回单路与操作者意图不符（配置期 fail-fast）。
        if self.hyde_mode not in ("dual", "replace"):
            raise ValueError(
                f"QueryRouterConfig.hyde_mode={self.hyde_mode} 必须 ∈ {{dual, replace}}"
            )


@dataclass
class _IntentPlan:
    """P0-2 意图分类结果（规则原语 _classify_intent 输出）。

    intent ∈ {time, identity, attribute, event, multi_hop}；at_ts 为 session_ts
    锚定后的时间锚（相对时间词/年份换算，cat=2 时间推理根治点）。
    """

    intent: str
    time_mode: str = "current"  # latest / at_time / current
    at_ts: Optional[float] = None  # 时间锚（相对时间词换算 / 年份年末）
    time_key: Optional[str] = None  # 相对时间锚稳定语义键（绝对时间/无锚 → None）
    entities: list[str] = field(default_factory=list)  # 候选实体名
    property_terms: list[str] = field(default_factory=list)  # 属性词（英文 attr_name）


@dataclass
class _AnchorSet:
    """P0-2 证据消息锚点集（_extract_anchors 输出，供下一轮 refine）。

    all = 实体 ∪ 属性词 ∪ 时间锚哨兵（全部锚点）；_agentic_retrieve 维护
    seen_anchors 求差得 new（真正未见过的新锚点），len(new) 用于
    agentic_min_new 枯竭判定——修复前每轮直接重算并集，相同锚点重复满足 min_new。
    """

    all: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    time_anchor: Optional[float] = None
    property_terms: list[str] = field(default_factory=list)


class QueryRouter:
    """
    查询路由 — 多信号检索融合引擎。

    融合模式（FUSION）— 三路并行融合（向量+BM25+实体匹配）：
      - 向量检索 (FAISS):   权重 0.5
      - BM25 关键词检索:     权重 0.3
      - 实体名称匹配 (GraphLite): 权重 0.2

    向后兼容降级链模式：
      - 超图检索 (HYPERGRAPH): GraphLite + FAISS 联合检索
      - 向量检索 (VECTOR): FAISS-only 降级
      - 关键词检索 (KEYWORD): TF-IDF 最终兜底

    当 GraphLite 断路器跳闸时自动降级到 VECTOR，
    当 FAISS 不可用时自动降级到 KEYWORD。
    """

    def __init__(
        self,
        graphlite_store,
        faiss_index,
        tfidf_index,
        encoder=None,
        config: Optional[QueryRouterConfig] = None,
        faiss_id_map: Optional[dict] = None,
        episode_cache: Optional[dict] = None,
        services=None,
        attr_aliases: Optional[dict] = None,
    ) -> None:
        """
        Args:
            graphlite_store: GraphLiteStore 实例
            faiss_index: FAISS 向量索引
            tfidf_index: TF-IDF 关键词索引
            encoder: 文本嵌入编码器（可选）
            config: 路由配置
            faiss_id_map: FAISS int id → GraphLite UUID string 映射（用于 L1 超图检索反查）
            services: Services 容器引用（可选，【P2-a V-Mem】共享 CLIP 嵌入器 /
                512→384 投影，保证视觉 query 与写路径同空间）
            attr_aliases: 属性别名归一表 {canonical: [alias...]}（【v5.50.0 P2】；
                空表 → 属性通道检索逐字节等价零回归）
        """
        self.graphlite_store = graphlite_store
        self.faiss_index = faiss_index
        self.tfidf_index = tfidf_index
        self.encoder = encoder
        self.faiss_id_map = faiss_id_map if faiss_id_map is not None else {}
        self._services = services
        # 【R4 P1-7】构造注入路径 dict 校验：启动时 api/app.py 注入
        # extended JSON 顶层 attr_aliases，可能是 truthy 的 list/string（文件损坏），
        # 未校验则 :2396 `.items()` 抛 AttributeError 被外层 try 吞掉 → 属性通道静默降级。
        self._attr_aliases = attr_aliases if isinstance(attr_aliases, dict) else {}
        # 【M5】共享 Services._episode_cache（EpisodeCache: OrderedDict LRU + TTL）。
        # flush_faiss_buffer 是唯一写入方；传入 None（测试/降级）时退化为裸 dict，
        # 行为与改造前一致（恒空）。
        self._episode_cache = episode_cache if episode_cache is not None else {}
        # 【M4】CJK 通道跳过一次性标志（进程内首次 warning，不刷屏）
        self._cjk_warned = False
        self.config = config or QueryRouterConfig()
        self._time_keywords = [
            "最近",
            "刚刚",
            "刚才",
            "之前说的",
            "上一条",
            "昨天",
            "今天",
            "几分钟前",
            "上一次",
            "recent",
            "just now",
            "earlier",
            "last",
            "previous",
            "yesterday",
        ]
        # 中英文技术术语映射（all-MiniLM-L6-v2 是英文优化，中文术语→英文提升对齐）
        self._zh_en_tech_map: dict[str, str] = {
            "深度学习": "deep learning",
            "框架": "framework",
            "向量": "vector",
            "数据库": "database",
            "编码器": "encoder",
            "解码器": "decoder",
            "图数据库": "graph database",
            "神经网络": "neural network",
            "机器学习": "machine learning",
            "自然语言": "natural language",
            "训练": "training",
            "推理": "inference",
            "离线": "offline",
            "在线": "online",
            "加载": "load",
            "嵌入": "embedding",
            "相似度": "similarity",
            "搜索": "search",
            "检索": "retrieval",
            "分类": "classification",
            "回归": "regression",
            "聚类": "clustering",
            "社区": "community",
            "梦境": "dream",
            "记忆": "memory",
            "知识图谱": "knowledge graph",
            "超图": "hypergraph",
            "图查询": "graph query",
            "监听": "monitor",
            "健康": "health",
            "错误": "error",
            "恢复": "recovery",
            "备份": "backup",
            "缓存": "cache",
            "索引": "index",
            "重建": "rebuild",
            "部署": "deploy",
            "容器": "container",
            "服务器": "server",
            "爬虫": "crawler",
        }
        # BM25 检索索引（延迟初始化）
        self._bm25_vectorizer = None  # TfidfVectorizer 实例
        self._bm25_docs: list[str] = []  # 文档列表
        self._bm25_doc_ids: list[str] = []  # 文档 ID 列表
        self._bm25_doc_contents: list[str] = []  # 文档 content 列表
        self._bm25_doc_tau: list[float] = []  # 文档 tau 值列表
        self._bm25_doc_term_matrix = None  # (n_docs, n_features) sparse matrix
        self._bm25_idf: np.ndarray = None  # (n_features,) IDF 向量
        self._bm25_doc_lens: np.ndarray = None  # (n_docs,) 每个文档的 term 数
        self._bm25_avgdl: float = 0.0  # 平均文档长度
        self._bm25_ready: bool = False  # BM25 索引是否就绪
        self._bm25_built: bool = False  # 标记构建是否成功（仅成功路径置位）
        self._bm25_last_attempt: float = 0.0  # 上次构建尝试时间（失败后冷却重试）
        self._bm25_build_lock = threading.Lock()  # 【P1-2】BM25 构建锁（防并发构建竞态）
        # 【P4】BM25 构建进行中标志：构建期间 _bm25_search 返回旧索引/None，
        # 不阻塞事件循环等待构建完成；锁只保护最终 swap（短临界区）。
        self._bm25_building: bool = False
        # 【P2-a V-Mem】视觉检索通道（prewarm_visual 构建 + add_visual_node 增量，_visual_recall 消费）：
        #   _visual_index    — 384d FaissStore（VisualNode embedding 空间）
        #   _visual_id_map   — faiss int id → VisualNode id
        #   _visual_meta     — VisualNode id → {caption, created_at, image_path}
        #   _visual_vecs     — VisualNode id → 384d 向量（【P1-1】写路径增量节点
        #                      留存，供 prewarm 全量重建时合并，防快照覆盖丢节点）
        #   _visual_projection — 512→384 随机投影（无 services 时本实例自持）
        #   _visual_built    — prewarm 成功置位（防重复构建）
        self._visual_index: Optional[FaissStore] = None
        self._visual_id_map: dict[int, str] = {}
        self._visual_meta: dict[str, dict] = {}
        self._visual_vecs: dict[str, np.ndarray] = {}
        self._visual_projection: Optional[np.ndarray] = None
        self._visual_built: bool = False
        # 【P2-1】视觉索引一致性锁：写侧（add_visual_node / prewarm 重建）持锁
        # 完成「index 变更 + 三个字典 swap」，读侧（_visual_snapshot）锁内一次性
        # 快照 —— 保证并发下读到全旧或全新的 (index, id_map, meta) 一致三元组，
        # 杜绝「新 index + 旧 map」的 fid 错配与「新节点已入 index 未入 map」的漏召回。
        # 锁内只做内存操作（FaissStore.add 自带内部锁），临界区微秒级，读侧性能无损。
        self._visual_lock = threading.Lock()
        # 【P3a】bge-reranker 懒加载状态：__init__ 不加载模型（CPU ~5-15s / ~400MB），
        # 首次 FUSION 检索经 _get_reranker 双重检查锁加载；失败置 _rerank_failed=True
        # 永久跳过（不再重试），主检索零回归降级。
        self._reranker = None
        self._rerank_failed = False
        self._rerank_lock = threading.Lock()

    def _normalize_query(self, query: str) -> str:
        """查询归一化：修复中文/英文混合输入，提升跨语言检索质量。

        1. 统一中文/英文标点
        2. 中文术语→英文（提升 all-MiniLM-L6-v2 对齐）
        3. 去除多余空格
        """
        q = query.strip()
        # 统一标点：中文标点→英文
        q = q.replace("，", " ").replace("。", " ").replace("；", " ")
        q = q.replace("：", " ").replace("？", " ").replace("！", " ")
        q = q.replace("「", " ").replace("」", " ").replace("『", " ").replace("』", " ")
        q = q.replace("（", " (").replace("）", ") ").replace("【", "").replace("】", "")
        # 统一空格
        q = re.sub(r"\s+", " ", q).strip()
        # 中文技术术语→英文（仅替换出现在文本中的术语）
        q_lower = q.lower()
        for zh, en in self._zh_en_tech_map.items():
            if zh in q_lower:
                q = q.replace(zh, en)
        # 最终清理多余空格
        return re.sub(r"\s+", " ", q).strip()

    # ──────────────────────────────
    # 三路并行融合引擎
    # ──────────────────────────────

    def _build_bm25_index(self) -> None:
        """构建 BM25 检索索引（同步入口，供测试/兼容直接调用）。

        【P4】重构（对比 M1-a 的整构建持锁）：
          - _bm25_building 标志防并发构建（构建中后续调用直接跳过，返回旧索引/None）
          - 锁只保护最终 swap（_swap_bm25_index 短临界区），GQL 拉取 + fit 在锁外
            以本地变量计算 → prewarm 超时遗留的 zombie 构建不会永久钉死锁
          - 失败/空语料不置位 _bm25_built（保留冷却窗口重试机会）
        """
        if (getattr(self, "_bm25_built", False)
                or getattr(self, "_bm25_ready", False)
                or getattr(self, "_bm25_building", False)):
            return
        self._bm25_building = True
        self._bm25_last_attempt = time.time()
        try:
            state = self._build_bm25_index_core()
            if state is not None:
                self._swap_bm25_index(state)
        finally:
            self._bm25_building = False

    async def _build_bm25_index_async(self) -> None:
        """异步构建 BM25：fit_transform 经 asyncio.to_thread 在线程执行。

        供 prewarm_bm25 使用（不阻塞事件循环）；构建期间查询线程见
        _bm25_building 标志即返回旧索引/None。
        """
        if (getattr(self, "_bm25_built", False)
                or getattr(self, "_bm25_ready", False)
                or getattr(self, "_bm25_building", False)):
            return
        self._bm25_building = True
        self._bm25_last_attempt = time.time()
        try:
            state = await asyncio.to_thread(self._build_bm25_index_core)
            if state is not None:
                self._swap_bm25_index(state)
        finally:
            self._bm25_building = False

    def _build_bm25_index_core(self):
        """BM25 索引构建核心（纯计算：不持锁、不改实例属性）。

        从 GraphLite Store 拉取 EpisodeNode 内容（LIMIT 下推到数据库侧），
        使用 sklearn TfidfVectorizer 计算 IDF，并预计算文档长度用于 BM25 评分。

        【P4】全程本地变量计算 → 返回 state 元组，由调用方在短临界区内一次性
        swap（_swap_bm25_index）。锁不再横跨 GQL 拉取 + fit。

        【M1】_bm25_built 仅由 _swap_bm25_index 在成功路径置位：query_cypher
        异常/空语料/TfidfVectorizer 异常时返回 None，保留重试机会。

        Returns:
            state = (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
                     term_matrix, idf, doc_lens, avgdl)，失败返回 None。
        """
        if self.graphlite_store is None:
            logger.warning("BM25: graphlite_store unavailable, skipping index build")
            return None

        try:
            rows = self.graphlite_store.query_cypher(
                "MATCH (e:EpisodeNode) "
                "WHERE (e.archived IS NULL OR e.archived = false) "
                "RETURN e.id AS node_id, e.content AS content, "
                "e.tau_initial AS tau_value, e.fact_track AS fact_track "
                "ORDER BY e.created_at DESC "
                "LIMIT $limit",  # 【P1-2】LIMIT 下推数据库侧，避免全库拉取 OOM
                {"limit": self.config.max_bm25_corpus},
            )
        except Exception:
            logger.exception("BM25: failed to fetch corpus from GraphLite")
            return None

        if not rows:
            # 【B-复审】空语料非终态：不置位 _bm25_built/_bm25_ready，
            # 由 _bm25_search 的冷却窗口（bm25_retry_cooldown）自然重试。
            # 【日志降噪】空库是启动期/无数据期的正常状态：进程内首次 warning
            # 提示一次（_bm25_empty_warned 标志），后续只打 debug，避免刷屏。
            if not getattr(self, "_bm25_empty_warned", False):
                self._bm25_empty_warned = True
                logger.warning("BM25: empty corpus from GraphLite")
            else:
                logger.debug("BM25: empty corpus from GraphLite (will retry after cooldown)")
            return None

        doc_ids: list[str] = []
        doc_contents: list[str] = []
        doc_tau: list[float] = []
        doc_fact_track: list[str] = []
        corpus: list[str] = []

        for row in rows:
            if isinstance(row, dict):
                nid = row.get("node_id", "") or ""
                content = row.get("content", "") or ""
                tau = _safe_float_tau(row.get("tau_value", 0.0))
                fact_track = row.get("fact_track", "active") or "active"
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = _safe_float_tau(row[2]) if len(row) > 2 else 0.0
                fact_track = str(row[3]) if len(row) > 3 and row[3] is not None else "active"
            else:
                continue

            if not nid or not content.strip():
                continue
            doc_ids.append(nid)
            doc_contents.append(content)
            doc_tau.append(tau)
            doc_fact_track.append(fact_track)
            corpus.append(content)

        if not corpus:
            # 【B-复审】同空语料：无有效文档非终态，不置位 → 冷却窗口后重试
            # 【日志降噪】首次 warning 提示（_bm25_empty_warned），后续 debug
            if not getattr(self, "_bm25_empty_warned", False):
                self._bm25_empty_warned = True
                logger.warning("BM25: no valid documents in corpus")
            else:
                logger.debug("BM25: no valid documents in corpus (will retry after cooldown)")
            return None

        # 限制语料大小，防 OOM
        max_corpus = self.config.max_bm25_corpus
        if len(corpus) > max_corpus:
            logger.warning("BM25: corpus too large (%d), truncating to %d",
                           len(corpus), max_corpus)
            corpus = corpus[:max_corpus]
            doc_ids = doc_ids[:max_corpus]
            doc_contents = doc_contents[:max_corpus]
            doc_tau = doc_tau[:max_corpus]
            doc_fact_track = doc_fact_track[:max_corpus]

        try:
            # 中文检索：使用字符级 bigram/trigram/4-gram（与 TfidfEncoder 一致），
            # 避免 \b 词边界对连续汉字失效导致中文语义词无法匹配
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                lowercase=True,
                max_features=50000,
            )
            tf_matrix = vectorizer.fit_transform(corpus)  # (n_docs, n_features)
            idf = np.array(vectorizer.idf_)  # (n_features,)

            # 预计算文档长度（term 数）
            doc_lens = tf_matrix.sum(axis=1).A1  # (n_docs,)
            avgdl = float(doc_lens.mean()) if doc_lens.size > 0 else 1.0

            logger.info(
                "BM25 index built",
                num_docs=len(corpus),
                features=len(idf),
                avgdl=round(avgdl, 2),
            )
            return (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
                    tf_matrix, idf, doc_lens, avgdl)
        except Exception:
            logger.exception("BM25: TfidfVectorizer failed")
            return None

    def _swap_bm25_index(self, state) -> None:
        """短临界区：持 _bm25_build_lock 一次性 swap BM25 索引状态。

        state 由 _build_bm25_index_core 本地计算返回，此处只做赋值，
        锁不横跨 GQL 拉取 + fit（prewarm 超时 zombie 不会永久钉死锁）。
        """
        (vectorizer, doc_ids, doc_contents, doc_tau, doc_fact_track,
         term_matrix, idf, doc_lens, avgdl) = state

        def _assign() -> None:
            self._bm25_vectorizer = vectorizer
            self._bm25_doc_ids = doc_ids
            self._bm25_doc_contents = doc_contents
            self._bm25_doc_tau = doc_tau
            self._bm25_doc_fact_track = doc_fact_track
            self._bm25_doc_term_matrix = term_matrix
            self._bm25_idf = idf
            self._bm25_doc_lens = doc_lens
            self._bm25_avgdl = avgdl
            # 【M1】成功路径末尾置位：失败/空语料（core 返回 None）不置位，保留重试机会
            self._bm25_ready = True
            self._bm25_built = True

        lock = getattr(self, "_bm25_build_lock", None)
        if lock is not None:
            with lock:
                _assign()
        else:
            _assign()

    async def prewarm_bm25(self) -> None:
        """异步预热 BM25 索引（启动时调用，不阻塞事件循环）。

        - 构建经 _build_bm25_index_async：fit_transform 放线程池（asyncio.to_thread）
        - 受 bm25_build_timeout 超时保护；超时/失败静默降级（_bm25_ready=False），
          首个查询走延迟构建或直接返回空，不影响启动
        - 超时取消后 _bm25_building 在 finally 复位，zombie 构建不持锁（_swap 短临界区）
        """
        try:
            await asyncio.wait_for(
                self._build_bm25_index_async(),
                timeout=self.config.bm25_build_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("BM25: prewarm timed out after %.1fs, degrading silently",
                           self.config.bm25_build_timeout)
        except Exception:
            logger.exception("BM25: prewarm failed, degrading silently")

    async def prewarm_reranker(self) -> None:
        """【P3a R1】启动异步预热 bge-reranker（冷启动护栏）。

        _get_reranker 双重检查锁懒加载模型（CPU ~5-15s），若留待 FUSION 首次
        检索才触发，会撞上 REST 3s 检索超时（_RETRIEVE_TIMEOUT）导致 FUSION
        请求超时降级。启动段预热让首次 FUSION 检索即就绪；模型加载放线程池
        不阻塞事件循环。幂等：已加载（_reranker 非 None）或已失败
        （_rerank_failed=True）直接返回；失败由 _get_reranker 置 _rerank_failed
        永久标记，_rerank_results 自动降级原列表（零超时风险）。
        """
        if getattr(self, "_reranker", None) is not None or getattr(self, "_rerank_failed", False):
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._get_reranker),
                timeout=_RERANK_PREWARM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self._rerank_failed = True
            logger.warning("reranker prewarm timed out after %.1fs, rerank disabled",
                           _RERANK_PREWARM_TIMEOUT)
        except Exception:
            self._rerank_failed = True
            logger.warning("reranker prewarm failed (non-fatal), rerank disabled",
                           exc_info=True)

    # ──────────────────────────────
    # 【P2-a V-Mem】视觉检索通道（VisualNode → 384d 索引 → 模态路由补充召回）
    # ──────────────────────────────

    async def prewarm_visual(self) -> None:
        """异步预热视觉索引（启动时调用，不阻塞事件循环）。

        链路：GraphLite 拉取 VisualNode（LIMIT visual_limit）→ 解析 embedding
        （JSON 字符串 → json.loads → f32；维度非 384 防御性跳过 + warning）→
        FaissStore(384).add + _visual_id_map/_visual_meta/_visual_vecs。
        【P1-1】与写路径增量（add_visual_node）合并：重建时携带内存中已增量
        索引的节点（_visual_vecs），防「prewarm DB 快照覆盖丢失并发写入节点」。
        【P2-1】合并 + 构建 + swap 在 _visual_lock 内原子完成（与 add_visual_node
        增量、_visual_snapshot 读侧同锁，杜绝重建期间读侧跨结构错配）。
        【P3-1 known-limitation】存量旧版 512d（bge 文本向量直落）VisualNode
        被跳过不迁移（两空间 bge vs CLIP 本质不可比，v5.46.0 已改写路径落 384d
        CLIP 投影空间；生产当前 VisualNode=0 无存量，仅标注不迁移）。
        【P2-1】构建成功后预热 CLIP（to_thread + 30s 超时），首次检索即就绪，
        检索路径绝不触发模型加载（3s 检索预算保护，见 _visual_recall 冷启动守卫）。
        全程 try/except 静默降级：失败/空库/无有效 384d 向量 → _visual_index
        保持 None，_visual_recall 空通道短路（检索零开销）。
        """
        if getattr(self, "_visual_built", False):
            return
        try:
            cfg = get_settings().retrieval.visual_recall
            if not getattr(cfg, "enabled", True):
                return
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "get_visual_nodes"):
                return
            visual_limit = int(getattr(cfg, "visual_limit", 10000))
            rows = await asyncio.to_thread(store.get_visual_nodes, visual_limit)
            rows = rows if isinstance(rows, list) else []
            embs: list[np.ndarray] = []
            id_map: dict[int, str] = {}
            meta: dict[str, dict] = {}
            vecs: dict[str, np.ndarray] = {}
            seen: set[str] = set()
            fid = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                nid = str(row.get("id", "") or "")
                if not nid:
                    continue
                emb = self._parse_visual_embedding(row.get("embedding"))
                if emb is None:
                    continue
                if emb.shape[0] != 384:
                    # 【P3-1 known-limitation】旧版 bge 512d 文本向量直落节点跳过
                    # 不迁移（生产无存量；改路径后新节点均为 384d CLIP 投影空间）
                    logger.warning(
                        "Visual prewarm: node %s embedding dim %d != 384, skipping "
                        "(known-limitation: legacy 512d nodes not migrated)",
                        nid, emb.shape[0],
                    )
                    continue
                emb_384 = emb.astype(np.float32)
                embs.append(emb_384)
                id_map[fid] = nid
                vecs[nid] = emb_384
                meta[nid] = {
                    "caption": str(row.get("caption", "") or ""),
                    "created_at": row.get("created_at"),
                    "image_path": str(row.get("image_path", "") or ""),
                }
                seen.add(nid)
                fid += 1
            # 【P1-1】合并写路径增量节点（_visual_vecs）：prewarm DB 快照之后、
            # swap 之前提交的节点不被覆盖丢失（按 nid 去重，两处入索引幂等）
            # 【P2-1】合并 + 构建 + swap 持 _visual_lock 原子完成：与 add_visual_node
            # 增量互斥（杜绝重建期间 fid 错配），与 _visual_snapshot 读侧同锁一致。
            with self._visual_lock:
                existing_vecs = getattr(self, "_visual_vecs", {}) or {}
                existing_meta = getattr(self, "_visual_meta", {}) or {}
                for nid, vec in existing_vecs.items():
                    if nid in seen:
                        continue
                    vec_384 = vec.astype(np.float32)
                    embs.append(vec_384)
                    id_map[fid] = nid
                    vecs[nid] = vec_384
                    meta[nid] = existing_meta.get(nid, {
                        "caption": "", "created_at": None, "image_path": "",
                    })
                    seen.add(nid)
                    fid += 1
                if not embs:
                    logger.debug("Visual prewarm: no valid 384d embeddings, channel stays empty")
                    return
                index = FaissStore(dimension=384)
                index.add(np.stack(embs), np.arange(len(embs), dtype=np.int64))
                self._visual_index = index
                self._visual_id_map = id_map
                self._visual_meta = meta
                self._visual_vecs = vecs
                self._visual_built = True
            logger.info("Visual index prewarmed", nodes=len(embs))
            # 【P2-1】CLIP 冷启动预热：启动时后台触发模型加载，首次检索即就绪。
            # 30s 超时静默降级（zombie 线程继续加载，加载完成后 available=True）；
            # 未加载期间 _visual_recall 冷启动守卫跳过视觉通道，不拖累文本检索。
            try:
                clip = self._get_clip_embedder()
                if clip is not None:
                    await asyncio.wait_for(
                        asyncio.to_thread(clip.embed_text, "__shm_visual_warmup__"),
                        timeout=30.0,
                    )
            except Exception:
                logger.debug("Visual prewarm: CLIP warmup failed/timed out, channel activates on load")
        except Exception:
            logger.exception("Visual prewarm failed, degrading silently")

    @staticmethod
    def _parse_visual_embedding(raw) -> Optional[np.ndarray]:
        """解析 VisualNode.embedding 字段（GraphLite 落库为 JSON 字符串），失败返回 None。"""
        if raw is None:
            return None
        try:
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, (list, tuple)):
                data = list(raw)
            else:
                return None
            return np.asarray(data, dtype=np.float32).reshape(-1)
        except Exception:
            return None

    def add_visual_node(self, node: dict) -> bool:
        """【P1-1】写路径增量入索引：VisualNode 创建成功后立即索引，无需重启/prewarm。

        与 prewarm_visual 同空间约束（384d CLIP 投影空间）：非 384d 跳过 +
        warning（两处策略一致，索引空间纯净）。索引未构建（None）时惰性引导
        构建——prewarm 空库/未跑时写入节点立即可检索。全程 try/except 静默
        降级（失败不抛异常、不阻断写路径）；幂等（已索引节点跳过）。
        【P2-1】查重 → index.add → 三字典 swap 全部在 _visual_lock 内原子完成，
        与 prewarm 重建、_visual_snapshot 读侧串行化（防 fid 碰撞/中间态）。

        Args:
            node: create_visual_node 同款 dict（含 id/embedding/caption 等）。

        Returns:
            True 入索引成功；False 跳过（降级/重复/维度不符/异常）。
        """
        try:
            cfg = get_settings().retrieval.visual_recall
            if not getattr(cfg, "enabled", True):
                return False
            nid = str((node or {}).get("id", "") or "")
            if not nid:
                return False
            # 【P2-1】锁内完成「查重 → index.add → 三字典 swap」：串行化并发增量，
            # 防 fid 碰撞（两并发调用同读 index.count 取到同一 fid）与读侧看到
            # 中间态（新 index + 旧 map）。_visual_snapshot 同锁 → 读侧永远拿到
            # 一致三元组。锁内仅内存操作，写路径（事件循环线程）临界区微秒级。
            with self._visual_lock:
                id_map = getattr(self, "_visual_id_map", {}) or {}
                if nid in id_map.values():
                    return False  # 已索引（prewarm 已含 / 重复调用）
                emb = self._parse_visual_embedding((node or {}).get("embedding"))
                if emb is None:
                    return False
                if emb.shape[0] != 384:
                    logger.warning(
                        "Visual add: node %s embedding dim %d != 384, skipping",
                        nid, emb.shape[0],
                    )
                    return False
                index = getattr(self, "_visual_index", None)
                if index is None:
                    index = FaissStore(dimension=384)
                    self._visual_index = index
                fid = index.count  # append-only：下一个可用 faiss id（同 prewarm 序号）
                emb_384 = emb.astype(np.float32)
                index.add(emb_384.reshape(1, -1), np.array([fid], dtype=np.int64))
                # copy-on-write swap：_visual_snapshot 读侧锁内快照，始终看到完整字典
                new_id_map = dict(id_map)
                new_id_map[fid] = nid
                new_meta = dict(getattr(self, "_visual_meta", {}) or {})
                new_meta[nid] = {
                    "caption": str((node or {}).get("caption", "") or ""),
                    "created_at": (node or {}).get("created_at"),
                    "image_path": str((node or {}).get("image_path", "") or ""),
                }
                new_vecs = dict(getattr(self, "_visual_vecs", {}) or {})
                new_vecs[nid] = emb_384
                self._visual_id_map = new_id_map
                self._visual_meta = new_meta
                self._visual_vecs = new_vecs
                logger.info("Visual node added to index", node=nid, fid=fid)
                return True
        except Exception:
            logger.debug("Visual add_visual_node failed, write path unaffected", exc_info=True)
            return False

    def _get_clip_embedder(self):
        """惰性获取 CLIP 嵌入器：优先共享写路径实例（services._clip_embedder）。"""
        svc = getattr(self, "_services", None)
        clip = getattr(svc, "_clip_embedder", None) if svc is not None else None
        if clip is None:
            from multimodal.embedders import ClipEmbedder
            clip = ClipEmbedder()
            if svc is not None:
                svc._clip_embedder = clip
        return clip

    def _get_projection(self) -> Optional[np.ndarray]:
        """512→384 随机投影（复用写路径 _clip_projection，query 与索引同空间）。

        与 api/routes/write.py 投影公式逐元素一致：default_rng(42).
        standard_normal((512, 384)) 后列归一——同一进程内 numpy RNG 序列确定，
        两处生成矩阵逐元素相等（投影一致性测试保障）。复用优先级：
        services._clip_projection → 本实例 _visual_projection → 创建并回写。
        """
        svc = getattr(self, "_services", None)
        proj = getattr(svc, "_clip_projection", None) if svc is not None else None
        if proj is None:
            proj = getattr(self, "_visual_projection", None)
        if proj is None:
            rng = np.random.default_rng(42)
            proj = rng.standard_normal((512, 384), dtype=np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self._visual_projection = proj
            if svc is not None:
                svc._clip_projection = proj
        return proj

    def _visual_snapshot(self) -> tuple[Optional[FaissStore], dict, dict]:
        """【P2-1】原子快照 (index, id_map, meta)：锁内一次性读取三个结构。

        写侧（add_visual_node / prewarm 重建）在同一把锁内完成「index 变更 +
        三个字典 swap」，故读侧快照必然是全旧或全新的**一致三元组**——避免分步
        getattr 读到「新 index + 旧 map」（fid 错配）或「新节点已入 index 未入
        map」（漏召回）。锁内仅三次属性读，微秒级，读路径性能无损。
        """
        with self._visual_lock:
            index = getattr(self, "_visual_index", None)
            id_map = getattr(self, "_visual_id_map", {}) or {}
            meta = getattr(self, "_visual_meta", {}) or {}
        return index, id_map, meta

    def _visual_recall(
        self,
        results: list[dict],
        query: str,
        raw_query: Optional[str],
    ) -> list[dict]:
        """【P2-a V-Mem】视觉通道补充召回：VisualNode append，补充非替代。

        链路（对齐 _community_expansion 的 try/except + append + 相对尾分缩放）：
          1. cfg.visual_recall.enabled 关闭 → 原样返回
          2. 空通道短路：_visual_index is None / ntotal==0 → 直接返回（无 GQL、无 CLIP）
          3. CLIP 惰性获取（共享写路径实例）；不可用 → 原样返回
          4. embed_text(raw_query or query) → 512d（CLIP multilingual 原生支持
             中文，zh→en 映射是 MiniLM 专用反而有害 → 必须用 raw_query）
          5. emb @ _get_projection() → 384d（与写路径同投影空间）
          6. idx.search(emb_384[None], max_results)；score = 1/(1+dist)；
             score *= min_seed_score * boost（相对尾分缩放，严格低于文本种子）
          7. append {node_id, content=caption, score, modality="visual",
             level="visual", _source="visual", created_at, image_path}
        异常 → 静默 return results（主检索零回归，永不抛异常）。
        """
        try:
            cfg = get_settings().retrieval.visual_recall
            if not getattr(cfg, "enabled", True):
                return results
            # 【P2-1】读侧一次性快照：锁内取 (index, id_map, meta) 一致三元组，
            # 后续检索全程使用快照，不再次触碰实例属性（写侧并发换字典不影响本次）。
            # 空通道用快照 id_map 判断（index.count 在锁外读会随并发 add 漂移）
            index, id_map, meta = self._visual_snapshot()
            if index is None or not id_map:
                return results
            clip = self._get_clip_embedder()
            if clip is None or not getattr(clip, "available", False):
                return results
            # 【P2-1】CLIP 冷启动隔离：真实 ClipEmbedder 模型未加载（_model is None）
            # → 跳过视觉通道——检索路径绝不触发模型下载/加载（外层 3s 检索超时内），
            # 由 prewarm_visual 启动预热 / 写路径（visual.py CLIP 编码）负责加载。
            if getattr(clip, "_model", None) is None:
                from multimodal.embedders import ClipEmbedder
                if isinstance(clip, ClipEmbedder):
                    logger.debug("Visual recall: CLIP not loaded yet, channel skipped")
                    return results
            emb = clip.embed_text(raw_query or query)
            if emb is None:
                return results
            emb_512 = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb_512.shape[0] != 512:
                return results
            proj = self._get_projection()
            if proj is None:
                return results
            emb_384 = emb_512 @ proj  # (512,) @ (512, 384) → (384,)
            distances, indices = index.search(
                emb_384[None].astype(np.float32),
                int(getattr(cfg, "max_results", 5)),
            )
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results[:5] if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            boost = float(getattr(cfg, "boost", 0.6))
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for i in range(indices.shape[1]):
                fid = int(indices[0][i])
                if fid < 0:
                    continue
                nid = id_map.get(fid, "")
                if not nid or nid in existing_ids:
                    continue
                dist = float(distances[0][i])
                score = round(1.0 / (1.0 + max(dist, 0.0)) * min_seed_score * boost, 6)
                if score <= 0.0:
                    continue
                m = meta.get(nid, {})
                extra.append({
                    "node_id": nid,
                    "content": m.get("caption", ""),
                    "score": score,
                    "modality": "visual",
                    "level": "visual",
                    "_source": "visual",
                    "created_at": m.get("created_at"),
                    "image_path": m.get("image_path", ""),
                })
            if extra:
                logger.info("Visual recall appended", candidates=len(extra), boost=boost)
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Visual recall degraded, returning original results", exc_info=True
            )
            return results

    def _bm25_search(self, query: str, k: int = 20) -> list[dict]:
        """BM25 关键词检索。

        使用 sklearn TfidfVectorizer 的 IDF 权重实现 BM25 评分。
        BM25(q, d) = Σ IDF(qᵢ) * (TF(qᵢ, d) * (k1 + 1)) / (TF(qᵢ, d) + k1 * (1 - b + b * |d| / avgdl))

        Args:
            query: 查询文本
            k: 返回 top-K 结果

        Returns:
            [{"node_id", "content", "score", "tau_value", "level": "bm25"}, ...]
        """
        # 【M1】懒构建：失败不置位 _bm25_built → 保留重试机会；
        # 以 bm25_retry_cooldown 为冷却窗口，避免失败后每次检索都触发全量重建
        # 【P4】构建进行中（_bm25_building）→ 不阻塞等待，返回旧索引/None；
        # 旧索引可用（_bm25_ready）时直接使用，否则返回空。
        if not self._bm25_built:
            if getattr(self, "_bm25_building", False):
                if not self._bm25_ready:
                    logger.debug("BM25: index building in progress, returning empty")
                    return []
            else:
                last_attempt = getattr(self, "_bm25_last_attempt", 0.0)
                if (last_attempt == 0.0
                        or time.time() - last_attempt >= self.config.bm25_retry_cooldown):
                    self._build_bm25_index()
        if not self._bm25_ready or self._bm25_vectorizer is None:
            # 【日志降噪】索引未就绪（空库/构建中）是正常状态，debug 即可
            logger.debug("BM25: index not ready")
            return []

        cfg = self.config
        k1, b = cfg.bm25_k1, cfg.bm25_b

        query_vec = self._bm25_vectorizer.transform([query])  # (1, n_features)
        query_indices = query_vec.indices  # non-zero feature ids
        if query_indices.size == 0:
            return []

        n_docs = self._bm25_doc_term_matrix.shape[0]
        scores = np.zeros(n_docs, dtype=np.float64)

        for qf_idx in query_indices:
            idf = self._bm25_idf[qf_idx]
            col = self._bm25_doc_term_matrix[:, qf_idx]
            # 【CSR 行索引修复】列切片 .indices 是列号（恒 0），须用 nonzero()[0] 取行号；
            # 修复前多文档共享 term 时 BM25 分全部累加到 docs[0]，其余文档恒 0 被跳过
            rows = col.nonzero()[0]
            if rows.size == 0:
                continue
            vals = col.data
            numerator = vals * (k1 + 1.0)
            denominator = vals + k1 * (1.0 - b + b * self._bm25_doc_lens[rows] / self._bm25_avgdl)
            scores[rows] += idf * numerator / denominator

        # 找出 top-k
        top_k = min(k, n_docs)
        if top_k == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        # 【Core-Boost】旧索引缓存兼容：无 _bm25_doc_fact_track 时缺省 active
        fact_tracks = getattr(self, "_bm25_doc_fact_track", [])
        results: list[dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append(
                {
                    "node_id": self._bm25_doc_ids[idx],
                    "content": self._bm25_doc_contents[idx],
                    "score": round(score, 6),
                    "tau_value": self._bm25_doc_tau[idx],
                    "fact_track": fact_tracks[idx] if idx < len(fact_tracks) else "active",
                    "level": "bm25",
                }
            )

        return results

    def _entity_match(self, query: str, k: int = 20) -> list[dict]:
        """实体匹配检索。

        从查询中提取候选实体名（unigram + bigram），
        逐一匹配 GraphLite 中的 EpisodeNode 内容。

        【P0-3】中文子串匹配不可用：GraphLite Rust lexer 不支持 UTF-8，
        _gql_value 对非 ASCII 文本做 b64 块编码——无子串保持性，
        CONTAINS 对中文不保证命中。中文检索主通道依赖向量（FAISS）/ BM25。

        Args:
            query: 查询文本
            k: 每个实体的匹配上限，总结果聚合去重

        Returns:
            [{"node_id", "content", "score", "tau_value", "level": "entity_match"}, ...]
        """
        if self.graphlite_store is None:
            logger.warning("Entity match: graphlite_store unavailable")
            return []

        tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 1]
        if not tokens:
            return []

        # 生成候选实体：bigram → unigram（bigram 优先，更精准）
        candidates: list[str] = []
        seen_phrases: set[str] = set()
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i + 1]}"
            if phrase not in seen_phrases:
                candidates.append(phrase)
                seen_phrases.add(phrase)
        for t in tokens:
            if t not in seen_phrases:
                candidates.append(t)
                seen_phrases.add(t)

        if not candidates:
            return []

        # 合并所有候选词为单个 OR CONTAINS 查询，避免 N+1 模式
        try:
            params: dict[str, str] = {}
            conditions: list[str] = []
            for i, term in enumerate(candidates):
                pkey = f"t{i}"
                params[pkey] = term.lower()
                conditions.append(f"e.content CONTAINS ${pkey}")
            where_clause = " OR ".join(conditions)
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE ({where_clause}) "
                f"AND (e.archived IS NULL OR e.archived = false) "
                f"RETURN e.id AS node_id, e.content AS content, "
                f"e.tau_initial AS tau_value, e.fact_track AS fact_track "
                f"LIMIT $limit"
            )
            params["limit"] = k * 2
            rows = self.graphlite_store.query_cypher(cypher, params)
        except Exception:
            # 【P1-1】SDK 通道故障（QueryError/ConnectionError）向上抛，不静默吞成 []：
            # _fusion_retrieve 的 per-channel except Exception 会捕获并置
            # fusion_channel_skipped=True（entity 通道降级信号不漏报）。
            # 正常无匹配是 query_cypher 返回空 rows（非异常），仍走下方正常 return []。
            logger.exception("Entity match OR query failed")
            raise

        # 【P3a R7】thread-local 降级信号：query_cypher 永不抛异常契约下，基础设施
        # 降级（熔断 open / 重试耗尽）表现为返回 []（非异常）。读同线程标志区分
        # 「正常无匹配」与「基础设施降级」，后者抛 _EntityChannelDegraded →
        # _fusion_retrieve per-channel handler 置 fusion_channel_skipped=True。
        degraded_fn = getattr(self.graphlite_store, "last_query_infra_degraded", None)
        if not rows and degraded_fn is not None and degraded_fn():
            raise _EntityChannelDegraded("entity channel infra degraded")

        seen_ids: set[str] = set()
        results: list[dict] = []
        candidates_lower = set(c.lower() for c in candidates)

        for row in rows:
            if isinstance(row, dict):
                nid = row.get("node_id", "") or ""
                content = row.get("content", "") or ""
                tau = _safe_float_tau(row.get("tau_value", 0.0))
                fact_track = row.get("fact_track", "active") or "active"
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                nid = str(row[0]) if row[0] is not None else ""
                content = str(row[1]) if row[1] is not None else ""
                tau = _safe_float_tau(row[2]) if len(row) > 2 else 0.0
                fact_track = str(row[3]) if len(row) > 3 and row[3] is not None else "active"
            else:
                continue

            if not nid or nid in seen_ids:
                continue
            seen_ids.add(nid)

            # 评分：取所有匹配候选中的最高分
            content_lower = content.lower()
            best_score = 0.0
            for term_lower in candidates_lower:
                if term_lower in content_lower:
                    match_ratio = len(term_lower) / max(len(content), 1)
                    score = min(1.0, 0.3 + 0.7 * match_ratio)
                    if score > best_score:
                        best_score = score

            if best_score == 0.0:
                # fallback: 部分词匹配
                term_tokens = content_lower.split()
                match_count = sum(1 for t in candidates_lower if t in content_lower)
                best_score = 0.1 + 0.2 * (match_count / max(len(candidates_lower), 1))

            results.append({
                "node_id": nid,
                "content": content,
                "score": round(best_score, 4),
                "tau_value": tau,
                "fact_track": fact_track,
                "level": "entity_match",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    @staticmethod
    def _apply_time_decay(results: list[dict], now_ts: Optional[float] = None) -> list[dict]:
        """时序衰减加权。

        对每个结果的 τ 值做时间衰减加权：
        score = score * (1 + 1 / (1 + exp(-τ / 60)))

        τ 值越高（越重要/新鲜），衰减因子的 boost 越大（1x ~ 2x）。

        now_ts: session 时间锚（P0-2 now 下沉参数；当前衰减基于 learnable τ 参数
        （tau_value），与墙钟无关——now_ts 仅为签名一致性透传，供单轮/agentic 多轮
        路径同签名调用，不改变现有 τ 衰减语义）。
        """
        for r in results:
            tau = float(r.get("tau_value", 0.0))
            boost = 1.0 + 1.0 / (1.0 + np.exp(-tau / 60.0))
            r["score"] = round(r["score"] * boost, 6)
        return results

    def _fuse_results(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        entity_results: list[dict],
        now_ts: Optional[float] = None,
    ) -> list[dict]:
        """三路结果加权融合。

        1. 各路内部归一化得分到 [0, 1]
        2. 加权混合：vector * w_v + bm25 * w_b + entity * w_e
        3. 时序衰减：基于 τ 值的时间衰减
        4. 去重：按 content[:100] 保留最高分

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            entity_results: 实体匹配结果

        Returns:
            融合后的结果列表（按 score 降序）
        """
        cfg = self.config
        w_v = cfg.weight_fusion_vector
        w_b = cfg.weight_fusion_bm25
        w_e = cfg.weight_fusion_entity

        def _normalize_scores(items: list[dict]) -> None:
            """就地 min-max 归一化得分到 [0, 1]"""
            if not items:
                return
            scores = [r["score"] for r in items]
            lo, hi = min(scores), max(scores)
            if hi - lo < 1e-9:
                for r in items:
                    r["score"] = 1.0
            else:
                for r in items:
                    r["score"] = (r["score"] - lo) / (hi - lo)

        _normalize_scores(vector_results)
        _normalize_scores(bm25_results)
        _normalize_scores(entity_results)

        # 加权融合
        fused: dict[str, dict] = {}
        for results, weight, source in [
            (vector_results, w_v, "vector"),
            (bm25_results, w_b, "bm25"),
            (entity_results, w_e, "entity"),
        ]:
            for r in results:
                key = r.get("node_id", "")
                if not key:
                    continue
                weighted_score = r["score"] * weight
                if key not in fused:
                    r["score"] = weighted_score
                    r["_source"] = source
                    r["level"] = f"fusion_{source}"
                    fused[key] = r
                else:
                    fused[key]["score"] += weighted_score
                    # 合并来源标记
                    existing_src = fused[key].get("_source", "")
                    if source not in existing_src:
                        fused[key]["_source"] = f"{existing_src}+{source}"
                        fused[key]["level"] = "fusion_multi"

        all_results = list(fused.values())

        # 时序衰减
        self._apply_time_decay(all_results, now_ts)

        # 去重 + 排序收敛到 retrieve() _finish 统一出口（此处不再重复，
        # 避免与 _finish 的 _deduplicate_and_sort 双重 boost）
        return all_results

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        level: RetrievalLevel = RetrievalLevel.HYPERGRAPH,
        include_archived: bool = False,
        session_ts: Optional[float] = None,
        rerank: Optional[bool] = None,
        hyde: Optional[bool] = None,
    ) -> list[dict]:
        """多信号检索融合入口。

        支持两种模式：
          - 降级链模式（HYPERGRAPH/VECTOR/KEYWORD）— 向后兼容
          - 融合模式（FUSION）— 三路并行融合（向量+BM25+实体匹配）

        Args:
            query: 查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            level: 检索级别（默认从 L1 开始，传入 FUSION 使用并行融合）
            include_archived: 是否包含已归档节点（默认 False，排除 archived=true）
            session_ts: session 时间锚（P0-2 时间推理根治；None 回落墙钟）。注入
                到相对时间词解析（_relative_time_at_ts/_property_time_mode），
                使"昨天/last year"等相对词对历史 session 按 session_ts 而非墙钟换算。
            rerank: bge-reranker 重排开关（P3a）。None → 读 config.rerank_enabled；
                仅 level==FUSION 时生效；False 显式关闭（关闭路径逐字节等价旧行为）。
            hyde: HyDE 假设文档增强开关（P3b）。None → 读 config.hyde_enabled
                （默认关）；仅 level==FUSION 时生效；True 时 LLM 生成假设段落
                参与检索（dual 双路合并 / replace 仅假设向量），生成失败静默
                降级现状单路；False 显式关闭（逐字节等价旧行为）。

        Returns:
            检索结果列表 [...]
        """
        # 查询归一化：中文标点统一 + 中文技术术语→英文
        raw_query = query  # 保留原始查询（BM25 通道需要未归一化的中文原文）
        query = self._normalize_query(query)

        def _finish(results: list[dict]) -> list[dict]:
            # 【Core-Boost】统一出口应用 core 轨 ×1.1 boost + 去重排序——
            # 覆盖 L1/L2/L3/L4/FUSION/graph_expansion 全部路径（原仅 FUSION/L1
            # 经 _deduplicate_and_sort，降级链 L2/L3/L4 直接返回原始列表无 boost）。
            # 【P2】非 include_archived 时先过滤归档再去重：避免同一 content[:100]
            # 的 archived 高分项在去重时挤掉 active 低分项，再过滤后结果为空。
            # 【v5.41 社区扩召回】补充非替代：统一去重/排序/boost 前 append 社区成员
            # （闭包可捕获 query/query_embedding/raw_query；候选经
            # _deduplicate_and_sort 单点去重 + core/画像 boost + 钳制，不双重放大）
            results = self._community_expansion(results, query, query_embedding, raw_query)
            # 【v5.49.0 MESA 记忆增强检索】补充非替代：社区摘要合成节点 append
            # （node_id=community_id 可回溯，content=summary；无 archived 字段恒保留）
            results = self._mesa_synthesis(results, query, raw_query)
            # 【P2-a V-Mem 视觉召回】补充非替代：_filter_archived 前 append VisualNode
            # （视觉节点无 archived 字段，走 _filter_archived 恒保留；候选同样经
            # _deduplicate_and_sort 单点去重 + score 钳制，相对尾分缩放严格低于种子）
            results = self._visual_recall(results, query, raw_query)
            # 【P0-1 属性时间版本链】补充非替代：实体属性版本 append（PropertyVerNode
            # 无 archived 字段，走 _filter_archived 恒保留；同样经单点去重 + score 钳制）
            # 【P0-2】session_ts 透传：相对时间词按 session 时间锚解析（None 回落墙钟）
            results = self._property_temporal_retrieve(results, query, raw_query, now_ts=session_ts)
            # 【P3c 跨消息多跳增强】补充非替代：查询专名实体跨会话召回 append
            # （EpisodeNode 带 archived 字段，_filter_archived 正常过滤；score = max(种子分)
            # × boost(0.9) 仅低于最高种子，经 _deduplicate_and_sort 单点去重不双重放大）
            results = self._entity_expansion(results, query, raw_query, now_ts=session_ts)
            # 【阶段3 图作用域检索】补充非替代：种子 EpisodeNode → 邻域向量检索
            # append（仅 overgraph 后端；EpisodeNode 带 archived 字段，
            # _filter_archived 正常过滤；score = max(种子分) × boost(0.9) 仅低于
            # 最高种子，经 _deduplicate_and_sort 单点去重不双重放大）
            results = self._scope_retrieve(results, query, query_embedding, now_ts=session_ts)
            # 【阶段4-1 Schema 模式蒸馏】补充非替代：Schema 节点（:Conceptual 标签）
            # 命中 → append 聚合线索上下文（尾分缩放；无 archived 字段恒保留）
            results = self._schema_recall(results, raw_query or query)
            if not include_archived:
                results = self._filter_archived(results)
            sorted_results = self._deduplicate_and_sort(results)
            # 【P3a】bge-reranker 重排：仅在 FUSION 且 rerank 开启时触发，在去重+boost+钳制
            # 之后重排头部覆盖 score，尾部原序保留。rerank=None → 读 config.rerank_enabled；
            # 异常/模型失败静默降级原列表（零回归）。
            if level == RetrievalLevel.FUSION:
                rerank_enabled = self.config.rerank_enabled if rerank is None else rerank
                sorted_results = self._rerank_results(
                    sorted_results, raw_query, bool(rerank_enabled)
                )
            return sorted_results

        strategy = self.detect_strategy(query)
        logger.info("Retrieval started", query=query[:80], level=level.value, strategy=strategy)

        # 【P0-2 Agentic】多步锚点检索编排（默认关）：agentic_enabled=True 且
        # level==FUSION 才走新路径（【P1-4】不再劫持 HYPERGRAPH/VECTOR/KEYWORD），
        # 首轮 = _route_channels(plan) + 三路融合（现有 FUSION 全路径）；False 时
        # 完全走下方既有单轮路径，字节级等价。
        if self.config.agentic_enabled and level == RetrievalLevel.FUSION:
            return self._agentic_retrieve(
                query, raw_query, query_embedding, session_ts, include_archived,
            )

        # F — 三路并行融合（向量 + BM25 + 实体匹配）
        if level == RetrievalLevel.FUSION:
            # 【P3b】HyDE 假设文档增强（默认关零回归）：仅 FUSION 生效；
            # hyde=None → 读 config.hyde_enabled；生成失败/未启用 → 现状单路。
            hyde_enabled = self.config.hyde_enabled if hyde is None else hyde
            if hyde_enabled:
                hypo = generate_hypothesis(raw_query, timeout=self.config.hyde_timeout)
                if hypo:
                    hypo_emb = self._encode_query(hypo)
                    if hypo_emb is not None:
                        if self.config.hyde_mode == "dual":
                            base = self._fusion_retrieve(
                                query, query_embedding, raw_query, now_ts=session_ts)
                            extra = self._fusion_retrieve(
                                hypo, hypo_emb, raw_query, now_ts=session_ts)
                            return _finish(base + extra)  # _deduplicate_and_sort 天然去重合并
                        # replace：单路，query_embedding 替换为假设向量
                        return _finish(self._fusion_retrieve(
                            query, hypo_emb, raw_query, now_ts=session_ts))
            return _finish(self._fusion_retrieve(query, query_embedding, raw_query, now_ts=session_ts))

        # 从指定级别开始，逐级尝试（空结果自动级联）
        results: list[dict] = []
        if level == RetrievalLevel.HYPERGRAPH:
            try:
                results = self._hypergraph_retrieve(query, query_embedding)
            except CircuitBreakerOpen:
                logger.warning("L1 circuit breaker open, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived, session_ts=session_ts)
                self._tag_degraded(r, level="l1_circuit_breaker")
                return r
            except FAISSUnavailable:
                logger.warning("L1 FAISS unavailable, cascading to L2")
                r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived, session_ts=session_ts)
                self._tag_degraded(r, level="l1_faiss_unavailable")
                return r
            if results:
                return _finish(results)
            logger.info("L1 empty, cascading to L2")
            r = self.retrieve(query, query_embedding, RetrievalLevel.VECTOR, include_archived=include_archived, session_ts=session_ts)
            self._tag_degraded(r, level="l1_empty")
            return r

        if level == RetrievalLevel.VECTOR:
            try:
                results = self._vector_retrieve(query, query_embedding)
            except FAISSUnavailable:
                logger.warning("L2 FAISS unavailable, cascading to L3")
                r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD, include_archived=include_archived, session_ts=session_ts)
                self._tag_degraded(r, level="l2_faiss_unavailable")
                return r
            if results:
                return _finish(results)
            logger.info("L2 empty, cascading to L3")
            r = self.retrieve(query, query_embedding, RetrievalLevel.KEYWORD, include_archived=include_archived, session_ts=session_ts)
            self._tag_degraded(r, level="l2_empty")
            return r

        # L3 keyword + L4 GraphLite fallback
        try:
            results = self._keyword_retrieve(query)
        except Exception as e:
            r = _finish(self._graphlite_text_fallback(query, str(e)))
            self._tag_degraded(r, level="l3_error")
            return r
        if results:
            return _finish(results)
        logger.info("L3 empty, trying L4 GraphLite fallback")
        r = _finish(self._graphlite_text_fallback(query, "L3 empty"))
        self._tag_degraded(r, level="l3_empty")
        return r

    def _fusion_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        raw_query: Optional[str] = None,
        now_ts: Optional[float] = None,
    ) -> list[dict]:
        """三路并行融合检索。

        同时运行向量、BM25、实体匹配三条通道，输出加权融合结果。

        【P6】vector/bm25/entity 三路经 ThreadPoolExecutor(max_workers=3) 并行
        执行（docstring 的"并行"变真）；逐通道收结果保持现有 try/except 降级，
        结果合并逻辑（_fuse_results）不变。【M4】CJK 跳过实体通道逻辑不破坏：
        CJK 查询不提交 entity 任务（省一次全表扫描），一次性 warning 标志保留。

        Args:
            query: 归一化后的查询文本
            query_embedding: 预计算的查询向量（None 则通过 encoder 编码）
            raw_query: 未归一化的原始查询（语料为原始中文，BM25 通道必须用它）

        Returns:
            融合检索结果列表
        """
        cfg = self.config

        # 【M4】CJK 查询跳过实体通道：GraphLite lexer 不支持 UTF-8，CONTAINS 对
        # 中文无子串保持性（b64 块编码），通道恒空——纯省一次全表扫描。
        # 在提交任务前判断：CJK 时根本不提交 entity 任务。
        skip_entity = any("一" <= ch <= "鿿" for ch in query)
        if skip_entity and not self._cjk_warned:
            self._cjk_warned = True
            logger.warning("Fusion: CJK query detected, skipping entity channel (CONTAINS not UTF-8 safe)")

        vector_results: list[dict] = []
        bm25_results: list[dict] = []
        entity_results: list[dict] = []

        def _run_vector():
            return self._vector_retrieve(query, query_embedding)

        def _run_bm25():
            # BM25 通道用未归一化的原始查询：语料为原始中文，归一化后无交集
            bm25_query = raw_query if raw_query is not None else query
            return self._bm25_search(bm25_query, cfg.top_k_vector)

        def _run_entity():
            return self._entity_match(query, cfg.top_k_keyword)

        fusion_channel_skipped = False
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures: dict = {}
            futures["vector"] = pool.submit(_run_vector)
            futures["bm25"] = pool.submit(_run_bm25)
            if not skip_entity:
                futures["entity"] = pool.submit(_run_entity)

            for channel in ("vector", "bm25", "entity"):
                fut = futures.get(channel)
                if fut is None:
                    continue
                try:
                    result = fut.result()
                except FAISSUnavailable:
                    logger.warning("Fusion: %s channel unavailable, skipping", channel)
                    fusion_channel_skipped = True
                    continue
                except Exception:
                    logger.exception("Fusion: %s channel failed", channel)
                    fusion_channel_skipped = True
                    continue
                if channel == "vector":
                    vector_results = result
                elif channel == "bm25":
                    bm25_results = result
                else:
                    entity_results = result

        logger.info(
            "Fusion retrieval results",
            vector=len(vector_results),
            bm25=len(bm25_results),
            entity=len(entity_results),
        )

        fused = self._fuse_results(vector_results, bm25_results, entity_results, now_ts)
        # 【P3】通道级降级也算降级信号：FAISSUnavailable/异常跳过某通道时打标
        # （部分通道降级语义；正常全通道不打标，不误报）。
        if fusion_channel_skipped:
            self._tag_degraded(fused, level="fusion_channel_skip")
        return fused

    # ──────────────────────────────
    # 【P0-2 Agentic】多步锚点检索编排（规则原语 + 编排器，全私有）
    # ──────────────────────────────

    @staticmethod
    def _classify_property_terms(query: str) -> list[str]:
        """轻量属性词检测：中英文属性词 → 英文 attr_name（无需 store 的 attr_names 集合）。

        与 _extract_property_terms 的区别：后者需已知 attr_names 过滤英文词；
        此处用中英文属性词映射做意图分类（分类先于检索，此时无 attr_names 可用）。
        【P1-3】补英文属性词表（revenue/market_cap 等）——"Apple revenue" 不再漏判属性意图。
        【R2 N2-P2】词边界匹配（\\b）——"age" 不再命中 "agent"/"manager"、"sales"
        不再命中 "salesforce"；下划线属性名归一化（market_cap ↔ market cap）后 \\b 判断。
        """
        terms: set[str] = set()
        for zh, en in _PROPERTY_QUERY_TERM_MAP.items():
            if zh in query:
                terms.add(en)
        ql = query.lower()
        for en in _PROPERTY_EN_TERMS:
            variants = {en}
            if "_" in en:
                variants.add(en.replace("_", " "))
            if any(re.search(r'\b' + re.escape(v) + r'\b', ql) for v in variants):
                terms.add(en)
        return sorted(terms)

    def _classify_intent(self, query: str, session_ts: Optional[float]) -> _IntentPlan:
        """规则分类查询意图 → _IntentPlan（cat=1 跨消息 / cat=2 时间推理 分流）。

        判定优先级（首匹配）：
          - time:      有时间意图且无实体/属性词 → 纯时间推理（property_temporal）
          - attribute: 有属性词 → 实体属性查询（property_temporal + fusion）
          - event:     有时间意图且有实体 → 事件回忆（fusion + hypergraph）
          - identity:  有实体无时间/属性 → 身份/实体回忆（entity + fusion）
          - multi_hop: 无明确信号 → 跨消息关联（fusion，首轮证据不足才 refine）
        """
        time_mode, at_ts = self._property_time_mode(query, session_ts)
        time_key = self._time_anchor_key(query) if at_ts is not None else None
        entities = self._extract_query_entities(query)
        property_terms = self._classify_property_terms(query)
        has_time = time_mode != "current"
        if has_time and not entities and not property_terms:
            intent = "time"
        elif property_terms:
            intent = "attribute"
        elif has_time and entities:
            intent = "event"
        elif entities:
            intent = "identity"
        else:
            intent = "multi_hop"
        return _IntentPlan(
            intent=intent,
            time_mode=time_mode,
            at_ts=at_ts,
            time_key=time_key,
            entities=entities,
            property_terms=property_terms,
        )

    @staticmethod
    def _route_channels(plan: _IntentPlan) -> list[str]:
        """意图 → 检索通道列表（fusion 为三路融合基础通道，恒执行）。

          time      → property_temporal（时间版本链，cat=2）
          identity  → entity + fusion（实体精确匹配，融合已含 entity）
          attribute → property_temporal + fusion
          event     → fusion + hypergraph（图扩散，事件关联）
          multi_hop → fusion（跨消息关联，首轮证据不足经锚点 refine）
        """
        mapping: dict[str, list[str]] = {
            "time": ["property_temporal"],
            "identity": ["entity", "fusion"],
            "attribute": ["property_temporal", "fusion"],
            "event": ["fusion", "hypergraph"],
            "multi_hop": ["fusion"],
        }
        return list(mapping.get(plan.intent, ["fusion"]))

    @staticmethod
    def _channels_from_anchors(
        anchors: Optional[_AnchorSet], plan: _IntentPlan
    ) -> list[str]:
        """由锚点集派生下一轮补充通道（cat=1 跨消息 refine）。

        新实体 → entity；属性词/时间锚 → property_temporal；event 意图保持 hypergraph。
        fusion 基础通道恒执行，故不重复列入。
        """
        if anchors is None:
            return []
        channels: list[str] = []
        if anchors.entities:
            channels.append("entity")
        if anchors.property_terms or anchors.time_anchor is not None:
            channels.append("property_temporal")
        if plan.intent == "event":
            channels.append("hypergraph")
        return channels

    def _sufficiency_check(self, results: list[dict], plan: _IntentPlan) -> bool:
        """证据充分性判定：top-k 归一化分差 ≥ agentic_score_gap 且 distinct 节点数 ≥ agentic_min_new。

        - 归一化分差 = (max - min) / max（top-1 相对 top-k 的区分优势幅度）；
          分差过小 → 结果拥挤、区分度不足 → 判证据不足（需 refine）
        - distinct 节点数过少 → 证据稀薄 → 判证据不足
        """
        top = results[: self.config.agentic_top_k]
        if not top:
            return False
        scores = [float(r.get("score") or 0.0) for r in top]
        hi, lo = max(scores), min(scores)
        if hi <= 0.0:
            return False
        gap = (hi - lo) / hi
        distinct = len({r.get("node_id") for r in top if r.get("node_id")})
        return gap >= self.config.agentic_score_gap and distinct >= self.config.agentic_min_new

    def _extract_anchors(
        self, top_results: list[dict], plan: _IntentPlan,
        session_ts: Optional[float] = None,
    ) -> _AnchorSet:
        """从证据消息（top 结果 content）提取实体 + 时间锚 + 属性词（cat=1 锚点 refine）。

        时间锚：优先 plan.at_ts（分类阶段已按 session_ts 解析）；否则从证据文本解析
        相对时间词/年份（【P1-2】按 session_ts 而非墙钟）。all = 实体 ∪ 属性词 ∪
        时间锚哨兵（全部锚点），由编排器求差得 new。
        """
        entities: list[str] = []
        property_terms: list[str] = []
        for r in top_results:
            text = r.get("content", "") or ""
            if not text:
                continue
            entities.extend(self._extract_query_entities(text))
            property_terms.extend(self._classify_property_terms(text))

        time_anchor = plan.at_ts
        time_key = plan.time_key  # 相对时间锚稳定语义键（绝对时间/无锚 → None）
        if time_anchor is None:
            for r in top_results:
                text = r.get("content", "") or ""
                if not text:
                    continue
                _, ats = self._property_time_mode(text, session_ts)
                if ats is not None:
                    time_anchor = ats
                    time_key = self._time_anchor_key(text)
                    break

        def _dedup(items: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for it in items:
                k = it.lower()
                if not it or k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        entities = _dedup(entities)[:5]
        property_terms = _dedup(property_terms)
        all_anchors = list(dict.fromkeys(entities + property_terms))
        if time_anchor is not None:
            # 【R2 N3-P2】时间锚 key 唯一化：不同时间锚（"yesterday" vs "2023"）值
            # 唯一，可各自作为新锚点计数（修复前统一哨兵 → 永不新增）。
            # 【R3 P3-1】相对时间锚用稳定语义键（time_key），绝对时间保留 timestamp。
            anchor_key = time_key if time_key is not None else time_anchor
            all_anchors.append(f"{_TIME_ANCHOR_SENTINEL}:{anchor_key}")
        return _AnchorSet(
            all=all_anchors, entities=entities, time_anchor=time_anchor,
            property_terms=property_terms,
        )

    def _hypergraph_supplement(self, results: list[dict]) -> list[dict]:
        """超图扩散补充：从融合种子沿超边扩散邻居（event 意图通道）。

        复用 _graph_expansion（相对尾分缩放，append 非替代）；失败静默返回原 results。
        """
        try:
            if not results or self.graphlite_store is None:
                return results
            # 【P2-1】先按 score 降序再取头尾——未排序时 results[:5]/[-1]
            # 非真实 top/lowest（_fuse_results 输出为 dict 无序集合）
            ranked = sorted(results, key=lambda r: r.get("score") or 0.0, reverse=True)
            seeds = [r.get("node_id") for r in ranked[:5] if r.get("node_id")]
            if not seeds:
                return results
            existing_ids = {r.get("node_id") for r in results}
            tail_score = float(ranked[-1].get("score") or 0.0)
            expansion = self._graph_expansion(seeds, existing_ids, tail_score)
            if expansion:
                return results + expansion
        except Exception:
            logger.debug("Agentic hypergraph supplement degraded", exc_info=True)
        return results

    def _agentic_round(
        self,
        channels: list[str],
        query: str,
        query_embedding: Optional[np.ndarray],
        raw_query: Optional[str],
        session_ts: Optional[float],
        include_archived: bool,
        at_ts: Optional[float] = None,
    ) -> list[dict]:
        """单轮检索：三路融合（基础）+ 路由补充通道（property_temporal/hypergraph）。

        与单轮 FUSION 全路径（retrieve _finish）对齐：社区 + 视觉补充 append、
        归档过滤；去重/排序/boost 统一收敛到编排器末尾 _deduplicate_and_sort
        （单点 boost，避免多轮双重放大）。"entity" 通道已由三路融合的实体匹配覆盖。
        at_ts：编排器已解析的时间锚（plan.at_ts / 证据时间锚），透传给
        _property_temporal_retrieve 不重算。
        """
        results = self._fusion_retrieve(query, query_embedding, raw_query, now_ts=session_ts)
        if "property_temporal" in channels:
            results = self._property_temporal_retrieve(
                results, query, raw_query, now_ts=session_ts, at_ts=at_ts,
            )
        if "hypergraph" in channels:
            results = self._hypergraph_supplement(results)
        results = self._community_expansion(results, query, query_embedding, raw_query)
        # 【P1-1】MESA 合成节点补充：与 retrieve() _finish 顺序一致（community → mesa → visual）。
        # 修复前 agentic 路径缺此步 → MESA 静默跳过 + 自演化统计 mesa_hit_count=0 误判。
        results = self._mesa_synthesis(results, query, raw_query)
        results = self._visual_recall(results, query, raw_query)
        # 【P3c】实体扩召回补充：与 retrieve() _finish 顺序一致（property → entity_expansion）。
        # 防 MESA 历史教训：agentic 路径缺接线会导致通道静默失效。
        results = self._entity_expansion(results, query, raw_query, now_ts=session_ts)
        if not include_archived:
            results = self._filter_archived(results)
        return results

    def _agentic_retrieve(
        self,
        query: str,
        raw_query: Optional[str],
        query_embedding: Optional[np.ndarray],
        session_ts: Optional[float],
        include_archived: bool,
    ) -> list[dict]:
        """P0-2 多步锚点检索编排器（agentic_enabled=True 且 level=FUSION 时替代单轮）。

        三重防死循环：
          1. seen 集合 —— 每轮过滤已见 node_id（同节点不跨轮重复累积）
          2. agentic_max_steps —— 硬上限（含首轮最多 N 轮）
          3. agentic_min_new —— 锚点枯竭（真正未见过的新锚点 < min_new）提前停

        首轮 = _route_channels(plan)（cat=2 时间锚注入 + cat=1 意图分流），
        后续轮 = _channels_from_anchors（证据消息锚点 refine）。
        """
        plan = self._classify_intent(query, session_ts)
        seen: set[str] = set()
        seen_anchors: set[str] = set()
        results: list[dict] = []
        anchors: Optional[_AnchorSet] = None

        for step in range(1, self.config.agentic_max_steps + 1):
            channels = (
                self._route_channels(plan)
                if step == 1
                else self._channels_from_anchors(anchors, plan)
            )
            at_ts = plan.at_ts if step == 1 else (anchors.time_anchor if anchors else None)
            round_results = self._agentic_round(
                channels, query, query_embedding, raw_query, session_ts,
                include_archived, at_ts=at_ts,
            )
            # 三重防护 1：去已见节点（防跨轮重复累积）
            fresh = [r for r in round_results if r.get("node_id") not in seen]
            results.extend(fresh)
            seen.update(r.get("node_id") for r in fresh if r.get("node_id"))
            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

            if step == self.config.agentic_max_steps:
                break  # 三重防护 2：硬上限
            if self._sufficiency_check(results[: self.config.agentic_top_k], plan):
                break  # 证据充分，不再 refine
            anchors = self._extract_anchors(results[: self.config.agentic_top_k], plan, session_ts)
            # 【P1-1】增量枯竭判定：新锚点 = all - seen_anchors（差集），
            # 相同锚点不再重复满足 min_new。
            # 【R2 N5-P3】大小写归一：Apple vs APPLE 计为同一锚点（lower 后差集）。
            new_anchors = [a for a in anchors.all if a.lower() not in seen_anchors]
            anchors.new = new_anchors
            if len(new_anchors) < self.config.agentic_min_new:
                break  # 三重防护 3：锚点枯竭
            seen_anchors.update(a.lower() for a in anchors.all)

        return self._deduplicate_and_sort(results)

    def _hypergraph_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L1 超图检索 + L2 向量检索融合降级（GraphLite + FAISS 联合）。

        1. 编码查询为向量
        2. FAISS 搜索最相关的 episode
        3. 通过 GraphLite 回查获取节点详情
        4. τ 值加权排序
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_l1)
            episode_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        valid_pairs = [
            (ep_id, score) for ep_id, score in episode_scores if ep_id >= 0
        ]
        uuid_map = {
            ep_id: self.faiss_id_map.get(ep_id, str(ep_id))
            for ep_id, _ in valid_pairs
        }

        # GraphLite 批量回查获取节点详情
        node_uuids = list(uuid_map.values())
        episodes_dict: dict[str, dict] = {}
        missing = [u for u in node_uuids if u not in self._episode_cache]
        if missing and self.graphlite_store is not None and hasattr(self.graphlite_store, 'get_episodes_batch'):
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in self.graphlite_store.get_episodes_batch(missing)
                }
            except CircuitBreakerOpen:
                raise  # 熔断跳闸：向上传播，由 retrieve() L613 级联到 L2
            except Exception:
                episodes_dict = {}

        results: list[dict] = []
        for faiss_id, score in valid_pairs:
            node_uuid = uuid_map[faiss_id]
            # episode_cache 优先
            if node_uuid in self._episode_cache:
                episode = dict(self._episode_cache[node_uuid])
                episode["id"] = node_uuid
            elif node_uuid in episodes_dict:
                episode = episodes_dict[node_uuid]
            else:
                continue
            if episode:
                episode["score"] = round(1.0 / (1.0 + float(score)), 4)
                episode["level"] = "l1_faiss"
                episode["_source"] = "faiss"
                episode["fact_track"] = episode.get("fact_track", "active")
                episode["node_id"] = episode.pop("id", "")
                results.append(episode)

        # Graph expansion (v5.26.0): 从向量种子沿超边扩散
        if results and self.graphlite_store is not None:
            try:
                seeds = [r["node_id"] for r in results[:5] if r.get("node_id")]
                existing_ids = {r["node_id"] for r in results}
                tail_score = results[-1]["score"] if results else 0.0
                expansion = self._graph_expansion(seeds, existing_ids, tail_score)
                if expansion:
                    results = results + expansion
            except Exception:
                pass  # 扩散失败静默回退，纯向量结果不受影响

        # 去重 + 排序收敛到 retrieve() _finish 统一出口（含 core boost）
        return results

    def _graph_expansion(
        self, seeds: list[str], existing_ids: set[str], tail_score: float
    ) -> list[dict]:
        """从向量种子沿超边扩散获取邻居节点。整个方法 try/except 包裹，异常返回 []。"""
        if not seeds:
            return []
        try:
            all_neighbors = self.graphlite_store.get_hypergraph_neighbors(
                seeds, self.config.graph_expansion_max
            )
        except CircuitBreakerOpen:
            return []
        except Exception as e:
            logger.error("Graph expansion failed: %s", e)
            return []

        results: list[dict] = []
        alpha = self.config.graph_expansion_alpha

        # 邻居 dict 无 archived 字段（get_hypergraph_neighbors 只回 id/content/co_occurrence）
        # 且 HYPEREDGE_MEMBER 边在归档后仍保留——回查补 archived 交 _filter_archived 过滤。
        neighbor_ids: list[str] = []
        for sid in seeds:
            for nb in all_neighbors.get(sid, []):
                nid = nb.get("id", "")
                if nid:
                    neighbor_ids.append(nid)
        archived_ids = self._lookup_archived_ids(neighbor_ids)
        # 【Core-Boost】邻居回查补 fact_track（get_hypergraph_neighbors 只回
        # id/content/co_occurrence），core 邻居在 _finish 统一出口同样获 ×1.1 boost
        fact_tracks = self._lookup_fact_track(neighbor_ids)

        for sid in seeds:
            for nb in all_neighbors.get(sid, []):
                nid = nb.get("id", "")
                content = nb.get("content", "")
                if not nid or nid in existing_ids:
                    continue
                if not content:
                    continue
                cooc = nb.get("co_occurrence", 0)
                score = round(1.0 / (1.0 + max(cooc, 0)) * tail_score * alpha, 6)
                results.append({
                    "node_id": nid,
                    "content": content,
                    "score": score,
                    "archived": nid in archived_ids,
                    "fact_track": fact_tracks.get(nid, "active"),
                    "level": "graph_expansion",
                    "_source": "graph",
                })

        # 跨种子截断：汇总所有种子的邻居后，按 score 降序取 top graph_expansion_max 条
        if len(results) > self.config.graph_expansion_max:
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:self.config.graph_expansion_max]

        return results

    def _community_expansion(
        self,
        results: list[dict],
        query: str,
        query_embedding: Optional[np.ndarray],
        raw_query: str,
    ) -> list[dict]:
        """【v5.41 社区扩召回】补充非替代：种子 → 社区（BM25-on-summary）→ 成员 append。

        链路（对齐超边扩召回）：
          1. seeds = 前 5 个检索结果的 node_id
          2. get_communities_by_seeds(seeds) → 所属社区；relevance = BM25(query, summary)
          3. relevance < threshold(0.5) 丢弃；相关社区 → get_community_members → 成员（排除种子）
          4. 扩展分 = relevance × min(种子分) × boost(0.6)（相对尾分缩放，严格低于种子）

        插入点在 retrieve() _finish 去重/排序前——候选经 _deduplicate_and_sort 统一
        去重、core/画像 boost、score 钳制（不双重放大）。GraphLite 失败/开关关闭 →
        静默返回原 results（主检索零回归，永不抛异常）。
        """
        try:
            cfg = get_settings().retrieval.community_expansion
            if not getattr(cfg, "enabled", True) or not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "get_communities_by_seeds"):
                return results
            seeds = [r.get("node_id") for r in results[:5] if r.get("node_id")]
            if not seeds:
                return results
            communities = store.get_communities_by_seeds(seeds)
            if not isinstance(communities, list) or not communities:
                return results
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results[:5] if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            boost = float(getattr(cfg, "boost", 0.6))
            threshold = float(getattr(cfg, "threshold", 0.5))
            max_members = int(getattr(cfg, "max_members", 10))
            relevance = self._community_relevance(
                raw_query or query,
                [c.get("summary", "") or "" for c in communities],
            )
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for comm, rel_score in zip(communities, relevance):
                if rel_score < threshold:
                    continue
                members = store.get_community_members(
                    comm.get("community_id", ""), limit=max_members
                )
                if not isinstance(members, list):
                    continue
                for m in members:
                    mid = m.get("member_id", "") or ""
                    content = m.get("content", "") or ""
                    if not mid or mid in existing_ids or not content:
                        continue
                    score = round(rel_score * min_seed_score * boost, 6)
                    if score <= 0.0:
                        continue
                    extra.append({
                        "node_id": mid,
                        "content": content,
                        "score": score,
                        "tau_value": float(m.get("tau_value") or 0.0),
                        "archived": m.get("archived", False),
                        "fact_track": m.get("fact_track") or "active",
                        "level": "community_expansion",
                        "_source": "community",
                    })
            if extra:
                logger.info(
                    "Community expansion appended",
                    candidates=len(extra),
                    boost=boost,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Community expansion degraded, returning original results", exc_info=True
            )
            return results

    def _community_relevance(self, query: str, summaries: list[str]) -> list[float]:
        """BM25-on-summary 相关度（[0,1]）：社区 summary 语料上的 BM25 + 单调归一。

        CC 修正 #1：keywords 未落库（GraphLite 只写 id/name/summary/leiden_score/
        created_at 5 字段）→ 相关度用 summary 词法（≤800 字散文含 Keywords 行）。
        与主 BM25 同特征空间（char_wb 2-4gram + IDF，k1=1.5/b=0.75）；归一化
        rel = bm25/(1+bm25)：无词法重叠 → 0（阈值闸口生效），强匹配 → 趋近 1。
        """
        if not summaries or not query:
            return [0.0] * len(summaries)
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), lowercase=True,
                max_features=50000,
            )
            tf = vectorizer.fit_transform(summaries)
            idf = np.array(vectorizer.idf_)
            doc_lens = tf.sum(axis=1).A1
            avgdl = float(doc_lens.mean()) if doc_lens.size > 0 else 1.0
            q_vec = vectorizer.transform([query])
            k1, b = 1.5, 0.75
            scores = np.zeros(len(summaries), dtype=np.float64)
            for qf_idx in q_vec.indices:
                idf_w = idf[qf_idx]
                col = tf[:, qf_idx]
                # 【CSR 行索引修复】列切片 .indices 是列号（恒 0），须用 nonzero()[0] 取行号；
                # 修复前多社区时所有 BM25 分累加到 summaries[0]，其余社区恒 0
                vals = col.data
                rows = col.nonzero()[0]
                if rows.size == 0:
                    continue
                numerator = vals * (k1 + 1.0)
                denominator = vals + k1 * (1.0 - b + b * doc_lens[rows] / avgdl)
                scores[rows] += idf_w * numerator / denominator
            return (scores / (scores + 1.0)).tolist()
        except Exception:
            return [0.0] * len(summaries)

    def _mesa_synthesis(
        self,
        results: list[dict],
        query: str,
        raw_query: str,
    ) -> list[dict]:
        """【v5.49.0 MESA 记忆增强检索】补充非替代：种子 → 社区摘要 → 合成节点 append。

        链路（对齐 _community_expansion 的 try/except + append + 相对尾分缩放）：
          1. seeds = 前 5 个检索结果的 node_id
          2. get_communities_by_seeds(seeds) → 所属社区；relevance = BM25(query, summary)
          3. relevance < threshold(0.5) 丢弃；相关社区 → 合成节点（社区摘要 = 梦境产物）
          4. 合成分 = relevance × min(种子分) × boost(0.4)（相对尾分缩放，严格低于
             种子，也低于 community_expansion.boost=0.6 的社区原始成员）
        【数学保证】mesa_boost=0.4 < community_expansion.boost=0.6 → 合成节点低于本社区
        原始成员；0.4 < 1 → 低于种子。合成节点 node_id=community_id（跨查询可回溯），
        content=summary（社区摘要），fact_track="active"（不给 core 标记避免误吃 ×1.1）。
        开关关闭/GraphLite 异常 → 静默返回原 results（默认关零回归，主检索永不抛异常）。
        """
        try:
            cfg = self.config
            if not getattr(cfg, "mesa_enabled", False) or not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "get_communities_by_seeds"):
                return results
            seeds = [r.get("node_id") for r in results[:5] if r.get("node_id")]
            if not seeds:
                return results
            communities = store.get_communities_by_seeds(seeds)
            if not isinstance(communities, list) or not communities:
                return results
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results[:5] if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            boost = float(getattr(cfg, "mesa_boost", 0.4))
            # 【P2-3】运行时 clamp：mesa_boost 不得超过 community boost 的 95%，
            # 确保合成节点严格低于社区原始成员（配置期校验只挡默认 0.6 的常见误配，
            # community boost 调低如 0.5 时 0.59 仍超 → 此处兜底钳制）。
            community_boost = float(
                getattr(get_settings().retrieval.community_expansion, "boost", 0.6)
            )
            boost = min(boost, community_boost * 0.95)
            threshold = float(getattr(cfg, "mesa_threshold", 0.5))
            max_nodes = int(getattr(cfg, "mesa_max_nodes", 5))
            # 【P2-2】下界守卫：max_nodes<=0 不合成任何节点——修复前循环先 append 再
            # break（append 后 `if len(extra) >= max_nodes: break`），0/-1 时首个候选仍合成 1 条。
            if max_nodes <= 0:
                return results
            relevance = self._community_relevance(
                raw_query or query,
                [c.get("summary", "") or "" for c in communities],
            )
            extra: list[dict] = []
            for comm, rel_score in zip(communities, relevance):
                if rel_score < threshold:
                    continue
                comm_id = comm.get("community_id", "") or ""
                summary = comm.get("summary", "") or ""
                if not comm_id or not summary:
                    continue
                score = round(rel_score * min_seed_score * boost, 6)
                if score <= 0.0:
                    continue
                extra.append({
                    "node_id": comm_id,
                    "content": summary,
                    "score": score,
                    "level": "mesa_synthesis",
                    "_source": "mesa",
                    "fact_track": "active",
                })
                if len(extra) >= max_nodes:
                    break
            if extra:
                logger.info(
                    "Mesa synthesis appended",
                    candidates=len(extra),
                    boost=boost,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Mesa synthesis degraded, returning original results", exc_info=True
            )
            return results

    # ──────────────────────────────
    # 【P0-1 实体-属性-时间】属性时间版本链检索通道（PropertyVerNode）
    # ──────────────────────────────

    def _extract_query_entities(self, text: str) -> list[str]:
        """从查询文本提取候选实体名（对齐 relation_extractor 的 subject 形态）。

        英文首字母大写词序列（含空格分隔多词）+ 中文组织/机构后缀词 + 小写英文词
        （【R2 N1-P1】"apple 收入" → "apple"，经 normalize_entity_name 与写侧
        "Apple Inc" 对齐，防属性检索静默失效）。        停用词 + 时间词/动词停用词过滤 +
        去重保序，最多 5 个候选。
        """
        text = _expand_contractions(text)
        candidates: list[str] = []
        for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', text):
            candidates.append(m.group(1))
        for m in re.finditer(
            r'([\u4e00-\u9fff]{2,8}(?:公司|集团|科技|有限|大学|银行|研究院))', text
        ):
            candidates.append(m.group(1))
        # 【R2 N1-P1】恢复小写英文词提取（P1-3 删过头 → "apple 收入" 属性检索静默
        # 失效）；时间词/动词（happened/year 等）仍由 _TIME_WORD_STOPWORDS 滤除。
        for m in re.finditer(r'\b([a-z]{2,})\b', text):
            candidates.append(m.group(1))
        seen: set[str] = set()
        out: list[str] = []
        for c in candidates:
            cl = c.lower()
            if cl in seen or cl in _PROPERTY_CANDIDATE_STOPWORDS:
                continue
            # 【R2 N4-P3】删年份过滤死分支：候选仅来自字母/中文正则，永不产生
            # 4 位数字（年份 "2023" 本就不会被提取），re.fullmatch 恒 False。
            if cl in _TIME_WORD_STOPWORDS:
                continue
            seen.add(cl)
            out.append(c)
            if len(out) >= 5:
                break
        return out

    def _property_time_mode(self, query: str, now_ts: Optional[float] = None) -> tuple[str, Optional[float]]:
        """判定查询的时间意图（P0-1 属性时间检索）。

        - latest: 含"最近/现在/最新/刚刚/当前"或 recent/now/latest/current
          （绝对最近词，非相对时间词）
        - at_time: 含 4 位年份（"2014"/"2020年"）→ 返回 (mode, 该年 12 月 31 日
          23:59:59 ts，P2-1 年末语义——年中生效版本不丢)；相对时间词（昨天/今天/
          earlier 等）→ 换算成 at_ts 走 at_time（P2-2，不再误归 latest）
        - current: 无时间意图 → 取全部未过期版本

        now_ts: session 时间锚（P0-2 下沉；None 回落墙钟 time.time()）。
        """
        q = query.lower()
        latest_hints = ("最近", "现在", "最新", "刚刚", "当前",
                        "recent", "now", "latest", "current", "just now")
        if any(kw in q for kw in latest_hints):
            return "latest", None
        # P2-2: 相对时间词换算 at_ts 走 at_time；不可换算的（上一条/之前说的等）
        # 不命中 latest —— 回落年份/current 判定
        rel_ts = self._relative_time_at_ts(q, now_ts)
        if rel_ts is not None:
            return "at_time", rel_ts
        m = re.search(r'(?:19|20)\d{2}', q)
        if m:
            # P2-1: 年份查询取年末（Dec 31 23:59:59），非年初
            ts = datetime(int(m.group(0)), 12, 31, 23, 59, 59).timestamp()
            return "at_time", ts
        return "current", None

    @staticmethod
    def _relative_time_at_ts(q: str, now_ts: Optional[float] = None) -> Optional[float]:
        """相对时间词 → at_ts（可换算的走 at_time）；不可换算返回 None。

        可换算（P0-1-R2 N4 修复）:
        - 数字 + 单位 + 前/ago（"5分钟前" / "5 minutes ago" / "3 days ago"）
        - 昨天/yesterday/earlier（1 天前）
        - 今天/today（当前时刻——当日 0 点会漏掉当天稍晚生效的版本）
        - last/previous + 单位（last year/month/week/day → 对应时长）
        - 裸 last/previous（1 天前）与字面"几分钟前"（5 分钟前）向后兼容
        "上一条/之前说的"等无明确锚点的相对词返回 None → 不误归 latest（P2-2）。

        now_ts: session 时间锚（P0-2 下沉参数）——对历史 session 用 session_ts 而非
        墙钟，根治 cat=2 时间推理（否则"昨天"对历史 session 恒按当前墙钟错算）。
        None 时回落 time.time()。
        """
        now = now_ts if now_ts is not None else time.time()
        ql = q.lower()
        # 数字 + 单位 + 前/ago（须有明确相对后缀，防 "2021 年" 被误判成 2021 年前）
        m = re.search(
            r'(\d+)\s*(分钟|小时|天|周|月|年|minute|hour|day|week|month|year)'
            r's?\s*(前|ago)', ql,
        )
        if m:
            n = int(m.group(1))
            seconds_per_unit = {
                "分钟": 60, "小时": 3600, "天": 86400, "周": 7 * 86400,
                "月": 30 * 86400, "年": 365 * 86400,
                "minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400,
                "month": 30 * 86400, "year": 365 * 86400,
            }
            return now - n * seconds_per_unit[m.group(2)]
        if any(k in ql for k in ("今天", "today")):
            return now
        if any(k in ql for k in ("昨天", "yesterday", "earlier")):
            return now - 86400.0
        # last/previous + 单位（先于裸 last/previous 判定）
        unit_seconds = {
            "year": 365 * 86400, "month": 30 * 86400,
            "week": 7 * 86400, "day": 86400,
        }
        for unit, secs in unit_seconds.items():
            if f"last {unit}" in ql or f"previous {unit}" in ql:
                return now - secs
        if any(k in ql for k in ("last", "previous")):
            return now - 86400.0
        if "几分钟前" in ql:
            return now - 300.0
        return None

    @staticmethod
    def _time_anchor_key(q: str) -> Optional[str]:
        """相对时间词的稳定语义键（跨轮稳定）；绝对时间（年份/日期）返回 None。

        【R3 P3-1】无 session_ts 时 _relative_time_at_ts 用 time.time()，每轮
        timestamp 不同 → 同一 "today" 被当新锚点，可能空转到 max_steps。按语义词
        规范化后（__time_anchor__:today）跨轮恒等；绝对时间仍保留 timestamp。
        判定顺序与 _relative_time_at_ts 一致（数字+单位/今天/昨天/last+N/裸 last）。
        """
        ql = q.lower()
        m = re.search(
            r'(\d+)\s*(分钟|小时|天|周|月|年|minute|hour|day|week|month|year)'
            r's?\s*(前|ago)', ql,
        )
        if m:
            unit = {
                "分钟": "minute", "小时": "hour", "天": "day", "周": "week",
                "月": "month", "年": "year",
                "minute": "minute", "hour": "hour", "day": "day",
                "week": "week", "month": "month", "year": "year",
            }[m.group(2)]
            return f"{m.group(1)}_{unit}_ago"
        if any(k in ql for k in ("今天", "today")):
            return "today"
        if any(k in ql for k in ("昨天", "yesterday", "earlier")):
            return "yesterday"
        for unit in ("year", "month", "week", "day"):
            if f"last {unit}" in ql or f"previous {unit}" in ql:
                return f"last_{unit}"
        if any(k in ql for k in ("last", "previous")):
            return "last"
        if "几分钟前" in ql:
            return "few_minutes_ago"
        return None

    @staticmethod
    def _is_property_expired(v: dict) -> bool:
        """属性版本是否已过期（GraphLite 缺失属性可能返回 'Null' 字符串，统一归一）。"""
        raw = v.get("expired_at")
        return raw not in (None, "", "Null", False)

    @staticmethod
    def _pick_property_versions(rows: list[dict], mode: str,
                                at_ts: Optional[float]) -> list[dict]:
        """按时间意图筛选属性版本：每 (entity_id, attr_name) 取 1 个版本。

        rows 已按 valid_from DESC（store 契约）：
        - latest/current → 首个未过期版本（即最新有效版）
        - at_time → 首个 valid_from ≤ at_ts 且目标时点未过期（expired_at IS NULL
          或 > at_ts，P2-1）的版本（DESC 序 = 该时点前最新）；无匹配 → 该属性跳过
        """
        picked: dict[tuple[str, str], dict] = {}
        for v in rows:
            eid = v.get("entity_id", "")
            attr = v.get("attr_name", "")
            if not eid or not attr:
                continue
            key = (eid, attr)
            if key in picked:
                continue
            if mode == "at_time":
                vf = v.get("valid_from")
                try:
                    vf_ts = float(vf) if vf is not None else None
                except (TypeError, ValueError):
                    vf_ts = None
                if vf_ts is None or vf_ts > at_ts:
                    continue
                # P2-1: 目标时点该版本必须尚未过期（expired_at IS NULL 或 > at_ts）
                if QueryRouter._is_property_expired(v):
                    try:
                        exp_ts = float(v.get("expired_at"))
                    except (TypeError, ValueError):
                        exp_ts = None
                    if exp_ts is None or exp_ts <= at_ts:
                        continue
            else:
                if QueryRouter._is_property_expired(v):
                    continue
            picked[key] = v
        return list(picked.values())

    @staticmethod
    def _property_content(v: dict) -> str:
        """属性版本 → 检索内容描述（供 _deduplicate_and_sort 去重 + 展示）。"""
        eid = v.get("entity_id", "")
        attr = v.get("attr_name", "")
        value = v.get("value", "")
        vf = v.get("valid_from")
        ts = ""
        try:
            vf_f = float(vf) if vf is not None else None
            if vf_f is not None:
                ts = time.strftime("%Y-%m-%d", time.localtime(vf_f))
        except (TypeError, ValueError):
            pass
        return f"[属性] {eid} 的 {attr}: {value}（自 {ts} 生效）"

    @staticmethod
    def _attr_name_matches(term: str, attr_name: str) -> bool:
        """属性词 ↔ attr_name 词边界匹配（下划线归一：market_cap ↔ market cap）。

        【R3 P2-2】"age" 不命中 "manager"/"agent"、"sales" 不命中 "salesforce"
        （子串假阳性根治）；下划线属性名与空格写法视作同词。
        """
        attr = attr_name.lower().replace("_", " ")
        t = term.lower().replace("_", " ")
        return bool(re.search(r"\b" + re.escape(t) + r"\b", attr))

    def set_attr_aliases(self, aliases: Optional[dict]) -> None:
        """运行时替换属性别名表（v5.50.0 P1-5）。

        梦境 attr_op 写盘后经 retrieval_guard 更新内层 _qr，新学别名无需重启即生效。
        空/None → 清空（属性通道检索逐字节等价零回归）。
        非 dict（list/string 等手工损坏或旧文件）→ 降级空 dict（【R3 P3-2】）。
        """
        if not isinstance(aliases, dict):
            aliases = {}
        self._attr_aliases = aliases

    @staticmethod
    def _expand_attr_aliases(terms: list[str], aliases: dict) -> list[str]:
        """【v5.50.0 P2】属性词归一：term 命中 alias 表 → 扩展出 canonical。

        纯增量（只可能多命中）；空表/无命中 → 返回原 terms。去重保序，原 term
        恒保留，命中 alias 时追加 canonical（下划线/大小写归一后反查）。
        """
        if not terms or not aliases:
            return list(terms)
        alias_to_canonical: dict[str, str] = {}
        for canonical, alias_list in aliases.items():
            if not isinstance(alias_list, (list, tuple, set)):
                continue
            for a in alias_list:
                k = str(a).lower().replace("_", " ")
                if k:
                    alias_to_canonical[k] = canonical
        out: list[str] = []
        seen: set[str] = set()
        for t in terms:
            tk = str(t).lower()
            if tk not in seen:
                seen.add(tk)
                out.append(t)
            canon = alias_to_canonical.get(str(t).lower().replace("_", " "))
            if canon is not None and canon.lower() not in seen:
                seen.add(canon.lower())
                out.append(canon)
        return out

    @staticmethod
    def _extract_property_terms(query: str, attr_names: set[str],
                                aliases: Optional[dict] = None) -> list[str]:
        """从查询提取属性词（attr_name 匹配片段）；无属性意图 → 空列表（不过滤）。

        - 中文属性词 → 英文 attr_name 同义词（"收入" → "revenue"，P0-1-R2 N5）
        - 英文词仅保留出现在已知 attr_name 中的（实体名等非属性词不误当属性词）
        - 【v5.50.0 P1-1】英文词命中 alias 表（即便不是现存 attr_name）也纳入，
          否则"只存 revenue、查 income"时 income 永不进 terms → 别名学习缺口。
        - 【v5.50.0 P1-4】alias 直接子串匹配：中文 alias（"营业额"）与多词英文
          alias 非 [a-z]{2,} 单 token，英文 token 提取收不到 → 直接
          `a in query.lower()` 覆盖（query 已 lower + 空格归一）。
        """
        # 【R4 P3-3】先还原撇号缩写（与 _extract_query_entities 一致），
        # 否则 [a-z]{2,} 永不产出撇号 token，停用表中 "don't" 等是死条件。
        query = _expand_contractions(query)
        terms: set[str] = set()
        for zh, en in _PROPERTY_QUERY_TERM_MAP.items():
            if zh in query:
                terms.add(en)
        alias_words: set[str] = set()
        for canonical, alias_list in (aliases or {}).items():
            if not isinstance(alias_list, (list, tuple, set)):
                continue
            for a in alias_list:
                w = str(a).lower().replace("_", " ")
                if w:
                    alias_words.add(w)
        # 【v5.50.0 P1-4】alias 直接子串匹配（覆盖中文 + 多词英文 alias）。
        # 下划线已归一为空格；query.lower() 使英文 alias 大小写不敏感。
        # 【R3 P1-6】子串匹配仅限含 CJK 或空格的 alias：纯 ASCII 单 token alias
        # （income/age/sales）走下方精确 token + 词边界通道，防 "Apple incoming"
        # 误命中 income（→ 扩出 revenue 并错误过滤属性版本）。
        ql = query.lower()
        for a in alias_words:
            if (" " in a or any(ord(ch) > 127 for ch in a)) and a in ql:
                terms.add(a)
        for m in re.finditer(r'[a-z]{2,}', ql):
            w = m.group(0)
            if w in _PROPERTY_CANDIDATE_STOPWORDS:
                continue
            if any(QueryRouter._attr_name_matches(w, an) for an in attr_names):
                terms.add(w)
            elif w in alias_words:
                terms.add(w)
        return sorted(terms)

    def _property_temporal_retrieve(self, results: list[dict], query: str,
                                    raw_query: Optional[str],
                                    now_ts: Optional[float] = None,
                                    at_ts: Optional[float] = None) -> list[dict]:
        """【P0-1 属性时间版本链】补充非替代：查询实体 → PropertyVerNode 版本 append。

        链路（对齐 _community_expansion 的 try/except + append + 相对尾分缩放）：
          1. raw_query 提取候选实体（_extract_query_entities）
          2. store.get_property_versions_for_entities(候选) → 全部版本（valid_from DESC）
          3. 时间意图（_property_time_mode）：latest（最近/现在）→ 最新未过期版；
             at_time（含年份）→ 该时点前最新版；current → 全部未过期版
          4. 评分 = min(种子分) × boost(0.6)（相对尾分缩放，严格低于种子）
          5. append {node_id, content, score, level="property_temporal",
             _source="property", attr_name, entity_id, valid_from}
        异常/无候选/无命中/无种子分 → 静默返回原 results（主检索零回归）。

        at_ts：编排器已按 session_ts 解析的时间锚（plan.at_ts / 证据时间锚），
        非 None 时直接按 at_time 取该时点前版本、不重算（【P1-2】）。
        """
        try:
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "get_property_versions_for_entities"):
                return results
            candidates = self._extract_query_entities(raw_query or query)
            if not candidates:
                return results
            rows = store.get_property_versions_for_entities(candidates)
            if not isinstance(rows, list) or not rows:
                return results
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results[:5] if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            if at_ts is not None:
                mode = "at_time"
            else:
                mode, at_ts = self._property_time_mode(query, now_ts)
            versions = self._pick_property_versions(rows, mode, at_ts)
            if not versions:
                return results
            # N5: 属性词过滤 —— query 出现属性词 → 只保留匹配 attr_name 的版本
            # （"Apple 收入" 不再返回 acquired_value 等无关属性）
            # 【v5.50.0 P1-1】alias 表传入提取阶段：alias 非现存 attr_name 时也能
            # 被识别为候选 term（否则 _expand_attr_aliases 收不到该 term）。
            aliases = getattr(self, "_attr_aliases", None) or {}
            attr_terms = self._extract_property_terms(
                query, {str(v.get("attr_name", "")).lower() for v in versions}, aliases
            )
            # 【v5.50.0 P2】属性词别名归一：term 命中 alias 表 → 扩展出 canonical
            # （空表/None → 恒等短路，检索零回归）
            attr_terms = self._expand_attr_aliases(attr_terms, aliases)
            if attr_terms:
                versions = [
                    v for v in versions
                    if any(QueryRouter._attr_name_matches(t, str(v.get("attr_name", ""))) for t in attr_terms)
                ]
                if not versions:
                    return results
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for v in versions:
                vid = v.get("id", "")
                if not vid or vid in existing_ids:
                    continue
                score = round(min_seed_score * _PROPERTY_BOOST, 6)
                if score <= 0.0:
                    continue
                extra.append({
                    "node_id": vid,
                    "content": self._property_content(v),
                    "score": score,
                    "level": "property_temporal",
                    "_source": "property",
                    "attr_name": v.get("attr_name", ""),
                    "entity_id": v.get("entity_id", ""),
                    "valid_from": v.get("valid_from"),
                })
            if extra:
                extra = extra[:_PROPERTY_MAX_RESULTS]
                logger.info("Property temporal appended",
                            candidates=len(extra), boost=_PROPERTY_BOOST)
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Property temporal retrieve degraded, returning original results",
                exc_info=True,
            )
            return results

    @staticmethod
    def _extract_proper_nouns(query: str) -> list[str]:
        """P3c：提取大写专名实体（连续大写词序列），保留原始大小写。

        复用 _extract_query_entities 的英文大写序列正则；句首 What/How/Where 等
        经 _PROPERTY_CANDIDATE_STOPWORDS 过滤（非实体）。上限 3 个实体。返回
        保留原始大小写（仅去首词停用词 + 去尾词后缀，不再小写化）的实体名，
        直接作 GQL CONTAINS 词——GraphLite CONTAINS 大小写敏感，小写化后打
        不进大写专名存储（"Melanie"→"melanie" 恒 0 命中），必须保留原始形式
        供 _entity_expansion 生成 orig/lower 双变体条件覆盖两种存储库。
        【R1 P2-2】token 级首词停用词剥离：句首 The/In 等混入多词序列时
        （"The Apple Store" 整段不在单词停用词集合 → 旧逻辑漏滤出 "the apple
        store"），逐 token 剥离首词停用词；"The" 单独成段剥离后为空 → 跳过。
        中间大写词与后置实体名保留（"The Apple Store" → "Apple Store"）。
        """
        from core.entity_resolver import (
            _ENTITY_NORMALIZE_SUFFIX_RE,
            normalize_entity_name,
        )
        seen: set[str] = set()
        out: list[str] = []
        for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', query):
            tokens = m.group(1).split()
            while tokens and tokens[0].lower() in _PROPERTY_CANDIDATE_STOPWORDS:
                tokens.pop(0)
            if not tokens:
                continue
            raw = " ".join(tokens)
            # 去重键用规范化（小写 + 去尾词后缀）结果（"Apple Inc" 与 "APPLE Inc"
            # 视为同一实体）；输出保留原始大小写 + 去尾词后缀（"Apple Inc"→"Apple"，
            # 与 lower 变体同源同后缀剥离，仅大小写不同）
            norm = normalize_entity_name(raw)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(_ENTITY_NORMALIZE_SUFFIX_RE.sub('', raw).strip())
            if len(out) >= 3:
                break
        return out

    def _entity_expansion(
        self,
        results: list[dict],
        query: str,
        raw_query: Optional[str],
        now_ts: Optional[float] = None,
    ) -> list[dict]:
        """【P3c 跨消息多跳增强】补充非替代：查询专名实体 → EpisodeNode 跨会话 append。

         链路（对齐 _community_expansion 的 try/except + append + 相对尾分缩放）：
          1. _extract_proper_nouns 提取大写专名实体（停用词过滤，保留原始大小写，
             top-3）
          2. 单条 OR CONTAINS GQL 合并查询（避免 N+1）：content 含任一实体的
             未归档 EpisodeNode，ORDER BY created_at DESC LIMIT 每实体×实体数；
             每实体生成 orig（原始大小写）/lower（小写）双变体条件——GraphLite
             CONTAINS 大小写敏感（实测 'Melanie' 3 rows / 'melanie' 0 rows），
             仅小写条件打不进大写专名存储 → P3c 恒空转，双变体覆盖大写/小写
             两种存储库（R2 P0）
          3. 时间锚上界过滤：now_ts（session 时间锚）非空且 config.time_filter
             开启 → AND e.created_at <= $at_ts（created_at 为时间戳秒数，int 转换）
          4. 扩展分 = max(种子分) × boost(0.9)（仅低于最高种子）
             append {node_id, content, score, level="entity_expansion",
             _source="entity_expansion"}

        【R1 P0】不再用全查询 CJK 判定短路：纯中文查询自然提取不到大写 ASCII
        专名 → 空实体跳过（原 CJK 早退语义保留）；中英混合查询（"Apple 最近做了
        什么"）先提取实体（apple）再走 CONTAINS——v5.31.4+ 中文原生直写（fork
        4452a96 UTF-8 lexer 修复），英文词 CONTAINS 对混合内容可用（对齐
        _entity_match 主通道；仅 v5.31.4 前遗留 {b64} 数据不命中，属历史数据
        迁移边界，非本通道回归）。
        【CC P3c】max(种子分) 取全部含 score 键的种子：推翻 R1-P1 的 min 锚契约。
        R1-P1 原契约（min 锚防扩展分反超低分种子）过度保守：cat1 聚合场景要求
        跨会话证据进 LLM 上下文（评测 docs[:40]→rerank top-12），min 锚使扩展分
        ≈0.25 沉底进不了 top-40；max 锚 + boost(0.9) 使扩展分 ≈0.81 仅低于最高
        种子，稳进 top-40，由内部/外部 rerank 双兜底收敛语义相关性。
        【R1 P2-3】与 FUSION _entity_match 通道存在重复全表 OR CONTAINS 扫描：
        本通道以专名实体词跨会话聚合（P3c 目的），_entity_match 以查询 token
        匹配当前会话——语义互补；单条合并查询 + ORDER BY created_at DESC
        LIMIT 下推（LIMIT 经 _interpolate 直插 GQL，引擎侧截断非 Python 全量
        拉取）控制成本，扩展候选经 _deduplicate_and_sort 单点去重不重复输出。
        异常/无实体/无种子分 → 静默返回原 results（主检索零回归，永不抛异常）。
        """
        try:
            ecfg = getattr(getattr(self, "config", None), "entity_expansion", None)
            if ecfg is None or not getattr(ecfg, "enabled", True) or not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "query_cypher"):
                return results
            entities = self._extract_proper_nouns(raw_query or query)
            if not entities:
                return results
            entities = entities[:max(1, int(getattr(ecfg, "max_entities", 3)))]
            # 【CC P3c】全部种子（score 非 None）的最大分（max 锚）；无有效种子分
            # → default 0.0 → 下方 max_seed_score <= 0.0 直接返回原 results（不扩展）。
            max_seed_score = max(
                (float(r.get("score") or 0.0) for r in results if r.get("score") is not None),
                default=0.0,
            )
            if max_seed_score <= 0.0:
                return results
            boost = float(getattr(ecfg, "boost", 0.9))
            per_entity = int(getattr(ecfg, "max_results", 10))
            if per_entity <= 0:
                return results
            # 单条 OR CONTAINS GQL（避免 N+1）：每实体生成 orig（原始大小写）+
            # lower（小写）双变体条件——GraphLite CONTAINS 大小写敏感，仅小写
            # 条件打不进大写专名存储（"Melanie"→"melanie" 恒 0 命中），双变体
            # 覆盖大写存储（Melanie）与小写存储（melanie）两种库
            params: dict = {}
            conditions: list[str] = []
            for i, ent in enumerate(entities):
                pkey_orig = f"t{i}_orig"
                pkey_lower = f"t{i}_lower"
                params[pkey_orig] = ent
                params[pkey_lower] = ent.lower()
                conditions.append(
                    f"(e.content CONTAINS ${pkey_orig} OR e.content CONTAINS ${pkey_lower})"
                )
            where_clause = " OR ".join(conditions)
            at_clause = ""
            if now_ts is not None and bool(getattr(ecfg, "time_filter", True)):
                at_clause = " AND e.created_at <= $at_ts"
                params["at_ts"] = int(now_ts)
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE ({where_clause}) "
                f"AND (e.archived IS NULL OR e.archived = false){at_clause} "
                f"RETURN e.id AS node_id, e.content AS content, "
                f"e.tau_initial AS tau_value, e.fact_track AS fact_track "
                f"ORDER BY e.created_at DESC LIMIT $limit"
            )
            params["limit"] = per_entity * len(entities)
            rows = store.query_cypher(cypher, params)
            if not isinstance(rows, (list, tuple)) or not rows:
                return results
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for row in rows:
                if isinstance(row, dict):
                    nid = row.get("node_id", "") or ""
                    content = row.get("content", "") or ""
                    tau = _safe_float_tau(row.get("tau_value", 0.0))
                    fact_track = row.get("fact_track", "active") or "active"
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    nid = str(row[0]) if row[0] is not None else ""
                    content = str(row[1]) if row[1] is not None else ""
                    tau = _safe_float_tau(row[2]) if len(row) > 2 else 0.0
                    fact_track = str(row[3]) if len(row) > 3 and row[3] is not None else "active"
                else:
                    continue
                if not nid or nid in existing_ids or not content:
                    continue
                score = round(max_seed_score * boost, 6)
                if score <= 0.0:
                    continue
                extra.append({
                    "node_id": nid,
                    "content": content,
                    "score": score,
                    "tau_value": tau,
                    "fact_track": fact_track,
                    "level": "entity_expansion",
                    "_source": "entity_expansion",
                })
            if extra:
                # 总 append 硬上限 20（实体 top-3 × 每实体 max-10 的钳制）
                extra = extra[:_ENTITY_EXPANSION_MAX_APPEND]
                logger.info(
                    "Entity expansion appended",
                    entities=len(entities), candidates=len(extra), boost=boost,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Entity expansion degraded, returning original results", exc_info=True
            )
            return results

    def _scope_retrieve(
        self,
        results: list[dict],
        query: str,
        query_embedding: Optional[np.ndarray],
        now_ts: Optional[float] = None,
    ) -> list[dict]:
        """【阶段3 图作用域检索】补充非替代：种子 EpisodeNode 邻域向量检索 append。

         链路（对齐 _entity_expansion 的 try/except + append + max 锚 × boost）：
          1. 仅 overgraph 后端：hasattr(store, "vector_search_scoped") 守卫
             （graphlite 后端假 no-op）
          2. 种子 = 首个 get_node_internal_id 成功的检索结果（跳过
             community/mesa/visual/property/entity_expansion 合成节点——其
             node_id 非真实 EpisodeNode elementKey）
          3. vector_search_scoped(seed, k=max_results, query_vec=query_embedding,
             max_depth=cfg.max_depth, at_ts=now_ts) → [(ep_id, score)]（引擎内
             direction 硬编码 both + depth 两跳，共享超边 co-member 可达，D7）
          4. get_episodes_batch 富化（content/archived/fact_track/tau 一次取回，
             免 N+1 回查）；扩展分 = max(种子分) × boost(0.9)（仅低于最高种子）
             append {node_id, content, score, level="scope", _source="scope"}
          5. 硬上限 cfg.max_results；无种子/无向量/异常 → 静默返回原 results
             （主检索零回归，永不抛异常）

        【D8】scope_at_epoch=int(at_ts×1000) 仅时间锚透传（SHM 边无时序）；
        已归档候选经 _finish _filter_archived 正常过滤（EpisodeNode 带 archived）。
        """
        try:
            scfg = getattr(getattr(self, "config", None), "scope_recall", None)
            if scfg is None or not getattr(scfg, "enabled", True) or not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "vector_search_scoped"):
                return results
            if query_embedding is None:
                # 【P1-1】生产入口（self_evolving.py:608 / api/routes/search.py:129）
                # 调 retrieve() 均不传 query_embedding → 外层恒 None → scope 通道
                # 生产永不生效（恒 no-op）。与 _vector_retrieve 同构：None 时先
                # 编码（query 已是归一化 query），编码失败才静默降级。
                query_embedding = self._encode_query(query)
                if query_embedding is None:
                    return results
            # 【P1-1】scope 引擎 dense_query 需 1D 向量；_encode_query 产出 2D
            # (1, dim)（FAISS 检索契约），reshape 为 1D 再传入（显式 1D 直通）。
            query_vec = (
                query_embedding.reshape(-1)
                if getattr(query_embedding, "ndim", 0) == 2
                else query_embedding
            )
            # 【P2-1】skip 集与实际合成 level 对齐（原 "community"/"mesa"/"property"
            # 是死名字，实际 level 为 community_expansion/mesa_synthesis/…）：
            #   合成节点（node_id 非真实 EpisodeNode elementKey，不可作种子）：
            #     - mesa_synthesis: node_id=community_id（CommunityNode，非 EpisodeNode）
            #     - visual: node_id=VisualNode id
            #     - property_temporal: node_id=PropertyVerNode id
            #     - schema: node_id=Conceptual id（阶段4-1 节点）
            #     - scope: 自身 level（防自递归种子）
            #   node_id 实为真实 EpisodeNode elementKey 的扩召回（community_expansion
            #   社区成员 / entity_expansion 实体召回均为 EpisodeNode）本可作种子，但
            #   阶段3 契约「种子取首个基础通道真实 EpisodeNode」统一跳过——基础通道
            #   （vector/bm25/entity）恒有真实 EpisodeNode，零种子损失。
            skip_levels = {
                "community_expansion", "mesa_synthesis", "visual",
                "property_temporal", "schema", "scope", "entity_expansion",
            }
            seed_id: Optional[str] = None
            for r in results:
                nid = r.get("node_id", "")
                if not nid or r.get("level") in skip_levels:
                    continue
                if store.get_node_internal_id(str(nid)) is not None:
                    seed_id = str(nid)
                    break
            if seed_id is None:
                return results
            max_seed_score = max(
                (float(r.get("score") or 0.0) for r in results if r.get("score") is not None),
                default=0.0,
            )
            if max_seed_score <= 0.0:
                return results
            boost = float(getattr(scfg, "boost", 0.9))
            max_depth = int(getattr(scfg, "max_depth", 2))
            max_results = int(getattr(scfg, "max_results", 10))
            if max_results <= 0:
                return results
            hits = store.vector_search_scoped(
                seed_id,
                k=max_results,
                query_vec=query_vec,
                max_depth=max_depth,
                at_ts=now_ts,
            )
            if not isinstance(hits, (list, tuple)) or not hits:
                return results
            hit_ids = [h[0] for h in hits if h and h[0]]
            if not hit_ids:
                return results
            episodes = store.get_episodes_batch(list(dict.fromkeys(hit_ids)))
            by_id = {
                str(ep.get("id", "")): ep
                for ep in episodes if isinstance(ep, dict) and ep.get("id")
            }
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for ep_id, hit_score in hits:
                if not ep_id or ep_id in existing_ids:
                    continue
                ep = by_id.get(str(ep_id))
                if ep is None:
                    continue
                content = ep.get("content", "") or ""
                if not content:
                    continue
                score = round(max_seed_score * boost, 6)
                if score <= 0.0:
                    continue
                extra.append({
                    "node_id": str(ep_id),
                    "content": content,
                    "score": score,
                    "tau_value": _safe_float_tau(ep.get("tau_initial", 0.0)),
                    "fact_track": ep.get("fact_track", "active") or "active",
                    "archived": ep.get("archived"),
                    "level": "scope",
                    "_source": "scope",
                    "_scope_sim": round(float(hit_score), 4),
                })
            if extra:
                extra = extra[:max_results]
                logger.info(
                    "Scope recall appended",
                    seed=seed_id[:12], candidates=len(extra), boost=boost,
                    max_depth=max_depth,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Scope recall degraded, returning original results", exc_info=True
            )
            return results

    def _schema_recall(self, results: list[dict], query: str) -> list[dict]:
        """【阶段4-1 Schema 模式蒸馏】补充非替代：查询术语 → Schema 节点 append。

        链路（对齐 MESA 合成节点通道的补充非替代语义）：
          1. 查询术语 = extract_terms(query)（拉丁词 + CJK 双字 gram）
          2. store.query_schema_nodes(terms)（两后端均有；缺失 → hasattr 守卫 no-op）
          3. 扩展分 = min(种子分) × _SCHEMA_BOOST(0.5)（相对尾分缩放，严格低于种子）
             append {node_id=schema_id 可回溯, content=description, level="schema",
             _source="schema"}
          4. 硬上限 _SCHEMA_MAX_RESULTS；无种子分/无 Schema/异常 → 静默返回原
             results（主检索零回归，永不抛异常）
        """
        try:
            if not results:
                return results
            store = getattr(self, "graphlite_store", None)
            if store is None or not hasattr(store, "query_schema_nodes"):
                return results
            terms = [t for t in extract_terms(query) if t]
            if not terms:
                return results
            min_seed_score = min(
                (float(r.get("score") or 0.0) for r in results if r.get("score")),
                default=0.0,
            )
            if min_seed_score <= 0.0:
                return results
            nodes = store.query_schema_nodes(terms, limit=_SCHEMA_MAX_RESULTS * 2)
            if not isinstance(nodes, (list, tuple)) or not nodes:
                return results
            existing_ids = {r.get("node_id") for r in results if r.get("node_id")}
            extra: list[dict] = []
            for node in nodes:
                nid = node.get("id", "")
                if not nid or nid in existing_ids:
                    continue
                content = node.get("summary", "") or ""
                if not content:
                    continue
                score = round(min_seed_score * _SCHEMA_BOOST, 6)
                if score <= 0.0:
                    continue
                extra.append({
                    "node_id": str(nid),
                    "content": content,
                    "score": score,
                    "level": "schema",
                    "_source": "schema",
                    "schema_name": node.get("schema_name", ""),
                    "schema_support": node.get("support", 0),
                })
            if extra:
                extra = extra[:_SCHEMA_MAX_RESULTS]
                logger.info(
                    "Schema recall appended",
                    schemas=len(extra), boost=_SCHEMA_BOOST,
                )
                results = results + extra
            return results
        except Exception:
            logger.debug(
                "Schema recall degraded, returning original results", exc_info=True
            )
            return results

    def _vector_retrieve(
        self, query: str, query_embedding: Optional[np.ndarray] = None
    ) -> list[dict]:
        """
        L2 纯向量检索（FAISS + GraphLite 回查）。

        在节点向量空间中检索，通过 GraphLite 回查补充节点内容、tau 值等字段。
        参考 /search/vector 路由的实现方式。

        Raises:
            FAISSUnavailable: FAISS 索引不可用
        """
        if query_embedding is None:
            query_embedding = self._encode_query(query)
        if query_embedding is None:
            raise FAISSUnavailable("No encoder available for query embedding")

        try:
            distances, indices = self.faiss_index.search(query_embedding, self.config.top_k_vector)
            node_scores = list(zip(indices[0], distances[0]))
        except Exception as e:
            raise FAISSUnavailable(f"FAISS search failed: {e}") from e

        valid_pairs = [(faiss_id, score) for faiss_id, score in node_scores if faiss_id >= 0]

        # 通过 faiss_id_map 将 FAISS int ID → GraphLite UUID string
        uuid_map: dict[int, str] = {}
        for faiss_id, _ in valid_pairs:
            uuid = self.faiss_id_map.get(faiss_id)
            if uuid:
                uuid_map[faiss_id] = uuid

        # 批量回查 GraphLite 获取节点详情（content、tau_value 等）
        episodes_dict: dict[str, dict] = {}
        if uuid_map and self.graphlite_store is not None and hasattr(self.graphlite_store, 'get_episodes_batch'):
            try:
                episodes_dict = {
                    ep["id"]: ep
                    for ep in self.graphlite_store.get_episodes_batch(list(uuid_map.values()))
                }
            except CircuitBreakerOpen:
                episodes_dict = {}  # 熔断降级：静默跳过回查（内容为空，不刷异常日志）
            except Exception:
                logger.exception(
                    "_vector_retrieve: GraphLite batch lookup failed, results will have empty content"
                )

        # 构建结果集：优先使用 GraphLite 回查的数据，fallback 到空 content
        results: list[dict] = []
        for faiss_id, score in valid_pairs:
            node_uuid = uuid_map.get(faiss_id, str(faiss_id))
            episode = episodes_dict.get(node_uuid, {})
            results.append({
                "node_id": node_uuid,
                "content": episode.get("content", ""),
                "score": round(1.0 / (1.0 + float(score)), 4),
                "tau_value": episode.get("tau_initial", 0.0),
                "archived": episode.get("archived", False),
                "fact_track": episode.get("fact_track", "active"),
                "level": RetrievalLevel.VECTOR.value,
            })

        return results

    def _keyword_retrieve(self, query: str) -> list[dict]:
        """
        L3 关键词检索（TF-IDF 最终兜底）。

        不依赖向量编码或图数据库，纯文本关键词匹配。
        """
        if self.tfidf_index is None:
            raise RuntimeError("TF-IDF index not available, all retrieval levels failed")

        try:
            keyword_results = self.tfidf_index.search(query, self.config.top_k_keyword)
        except Exception as e:
            raise RuntimeError(f"TF-IDF keyword search failed: {e}") from e

        items: list[tuple[str, str, float]] = []
        node_ids: list[str] = []
        for item in keyword_results:
            if isinstance(item, tuple) and len(item) >= 2:
                doc_id, score = item[0], item[1]
                content = item[2] if len(item) > 2 else ""
            else:
                doc_id, score = str(item), 0.0
                content = ""
            nid = str(doc_id)
            items.append((nid, str(content), float(score)))
            node_ids.append(nid)
        # TF-IDF 快照语料在 app.py/system.py fit，无法在此构建期过滤——回查补 archived + fact_track
        archived_ids = self._lookup_archived_ids(node_ids)
        fact_tracks = self._lookup_fact_track(node_ids)
        return [
            {
                "node_id": doc_id,
                "content": content,
                "score": score,
                "tau_value": 0.0,
                "archived": doc_id in archived_ids,
                "fact_track": fact_tracks.get(doc_id, "active"),
                "level": RetrievalLevel.KEYWORD.value,
            }
            for doc_id, content, score in items
        ]

    def _graphlite_text_fallback(self, query: str, error_context: str = "") -> list[dict]:
        """
        L4 GraphLite GQL 全文兜底检索。

        当 FAISS 和 TF-IDF 均不可用时，直接查询 GraphLite 数据库，
        使用 Cypher CONTAINS 做文本匹配。

        【P0-3】中文子串匹配不可用：b64 块编码无子串保持性，
        CONTAINS 对中文不保证命中。中文检索主通道依赖向量/BM25，L4 仅英文有效。

        Returns:
            检索结果列表 [{"node_id", "content", "score", "level": "graphlite_fallback"}, ...]
             失败时返回空列表（不抛异常）。
        """
        if self.graphlite_store is None:
            logger.warning("L4 fallback: graphlite_store unavailable")
            return []

        # 【M4】CJK 查询直接返回 []：GraphLite CONTAINS 对中文无子串保持性
        # （b64 块编码），L4 兜底通道对中文恒空——纯省一次全表扫描。
        if any("一" <= ch <= "鿿" for ch in query):
            if not self._cjk_warned:
                self._cjk_warned = True
                logger.warning("L4 fallback: CJK query skipped (CONTAINS not UTF-8 safe)")
            return []

        logger.info("L4 fallback: querying GraphLite directly", query=query[:80])
        try:
            # 提取关键词（取前5个有意义的词）
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
            search_terms = words[:5]

            if not search_terms:
                return []

            # 构建多个 CONTAINS 条件（大小写不敏感）
            params = {f"w{i}": w.lower() for i, w in enumerate(search_terms)}
            conditions = " OR ".join(
                f"e.content CONTAINS $w{i}" for i in range(len(search_terms))
            )
            cypher = (
                f"MATCH (e:EpisodeNode) WHERE ({conditions}) "
                f"AND (e.archived IS NULL OR e.archived = false) "
                f"RETURN e.id AS node_id, e.content AS content, e.tau_initial AS tau_value, "
                f"e.fact_track AS fact_track "
                "LIMIT $limit"
            )
            params["limit"] = self.config.top_k_keyword
            rows = self.graphlite_store.query_cypher(cypher, params)
            results = []
            for row in rows:
                if isinstance(row, dict):
                    results.append(
                        {
                            "node_id": row.get("node_id", ""),
                            "content": row.get("content", ""),
                            "score": 0.5,
                            "tau_value": row.get("tau_value", 0.0),
                            "fact_track": row.get("fact_track", "active"),
                            "level": "graphlite_fallback",
                        }
                    )
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    results.append(
                        {
                            "node_id": str(row[0]),
                            "content": str(row[1]),
                            "score": 0.5,
                            "tau_value": _safe_float_tau(row[2]) if len(row) > 2 else 0.0,
                            "fact_track": str(row[3]) if len(row) > 3 else "active",
                            "level": "graphlite_fallback",
                        }
                    )
            logger.info("L4 fallback results", count=len(results))
            return results
        except Exception:
            logger.exception("L4 GraphLite text fallback failed")
            return []

    def detect_strategy(self, query_text: str) -> RetrievalStrategy:
        """
        检测查询的时间敏感性，选择检索策略。

        Args:
            query_text: 用户查询文本

        Returns:
            推荐的检索策略
        """
        has_time_keyword = any(kw in query_text.lower() for kw in self._time_keywords)
        if has_time_keyword:
            return RetrievalStrategy.TAU_FIRST
        if len(query_text) > 50:
            return RetrievalStrategy.VECTOR_FIRST
        # 【PERF 2026-08-07】纯英文/ASCII 查询直接走 VECTOR_FIRST:
        # HYBRID 的实体匹配通道(CONTAINS 全表扫描)对英文 bigram 收益低且大库下
        # 拖慢整条检索(1946 节点实测卡死, benchmark 验证 vector 通道 recall 0.9)。
        # 中文查询保持 HYBRID(实体匹配对中文有区分度)。
        if not any('\u4e00' <= ch <= '\u9fff' for ch in query_text):
            return RetrievalStrategy.VECTOR_FIRST
        return RetrievalStrategy.HYBRID

    def hybrid_score(self, tau_score: float, vector_score: float) -> float:
        """
        计算混合得分。

        final_score = tau_weight * tau_score + vector_weight * vector_score
        """
        return self.config.tau_weight * tau_score + self.config.vector_weight * vector_score

    def fuse_results(
        self,
        results: list[dict],
        time_field: str = "created_at",
        score_field: str = "score",
    ) -> list[dict]:
        """
        融合检索结果并应用时间衰减因子。

        时间衰减公式: score = score * (1 + 1/(1 + exp(-age_hours/24)))
        效果: 24小时内新数据权重接近翻倍, 7天前衰减到0.5左右

        Args:
            results: 检索结果列表，每项需包含 score_field 和可选的 time_field
            time_field: 创建时间字段名（unix timestamp）
            score_field: 得分字段名

        Returns:
            时间衰减加权后的结果列表（按新 score 降序）
        """
        now = time.time()
        for r in results:
            try:
                created_at = r.get(time_field)
                if created_at is None:
                    continue
                age_hours = (now - float(created_at)) / 3600.0
                # 时间衰减因子: score * (1 + sigmoid(-age_hours/24))
                decay = 1.0 + 1.0 / (1.0 + math.exp(-age_hours / 24.0))
                r[score_field] = round(r.get(score_field, 0.0) * decay, 4)
            except (ValueError, TypeError, KeyError):
                continue  # 降级：异常时返回原分

        # 按新 score 降序排列
        results.sort(key=lambda x: x.get(score_field, 0.0), reverse=True)
        return results

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """编码查询文本为 2D 向量 (1, dim)，FAISS 要求 2D 输入"""
        if self.encoder is None:
            return None
        try:
            emb = self.encoder.embed(query)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb
        except Exception:
            logger.exception("Query encoding failed")
            return None

    @staticmethod
    def _tag_degraded(results, level: str) -> None:
        """给结果逐条打降级标记（防御式：仅对 list[dict] 生效，非 list 静默跳过）。

        【P3a R5】FUSION 通道级降级打标新增后，旧测试/极端路径可能让 results
        非 list（如 mock 返回字符串）；打标是元数据增强，不应对非 list 结果
        抛 TypeError（会破坏既有契约）。

        【P3a R6 P2-1】标签保留语义：`_degradation_level not in r` 是「先打先留」
        （first-writer-wins）。多级级联时内层先打标（如 l3_empty 比 l1_empty 更具体），
        外层后打不覆盖——保留最内层具体原因，诊断价值最高。
        """
        if not isinstance(results, list):
            return
        for r in results:
            if isinstance(r, dict) and "_degradation_level" not in r:
                r["_degradation_level"] = level

    def _get_reranker(self):
        """【P3a】懒加载 bge-reranker CrossEncoder（双重检查锁 + 失败永久标记）。

        仅 FUSION + rerank_enabled 时被 _rerank_results 调用。模型加载较慢
        （CPU ~5-15s），放 __init__ 会拖慢启动，故懒加载；sentence_transformers
        为可选依赖，函数内 import（未安装时 ImportError → _rerank_failed=True
        永久跳过，不再重试）。并发线程经双重检查锁保证只加载一次。
        """
        if getattr(self, "_reranker", None) is not None:
            return self._reranker
        if getattr(self, "_rerank_failed", False):
            return None
        lock = getattr(self, "_rerank_lock", None) or threading.Lock()
        self._rerank_lock = lock
        with lock:
            if getattr(self, "_reranker", None) is not None:
                return self._reranker
            if getattr(self, "_rerank_failed", False):
                return None
            try:
                import os as _os

                # 离线加载（同 embedding/encoder.py 模式）：进程内强制离线防网络挂起。
                # bge-reranker-base 是标准 XLMRobertaForSequenceClassification，无需
                # trust_remote_code（该参数会强制网络拉取远程代码，离线环境每次失败
                # 重试 ~30s 后才报错）。模型名命中本地 HF 缓存 snapshot 直接加载。
                _os.environ.setdefault("HF_HUB_OFFLINE", "1")
                _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    "BAAI/bge-reranker-base",
                    device="cpu",
                )
                return self._reranker
            except Exception:
                self._rerank_failed = True
                logger.warning(
                    "bge-reranker unavailable, rerank disabled (permanent)",
                    exc_info=True,
                )
                return None

    def _rerank_results(
        self,
        results: list[dict],
        raw_query: Optional[str],
        enabled: bool,
    ) -> list[dict]:
        """【P3a】bge-reranker 重排（仅 FUSION 头部，尾部原序保留）。

        - enabled=False / _rerank_failed / len<2 → 原列表直接返回
        - 取 top min(rerank_input_k, len) 头部候选，空 content 跳过（视觉节点无文本）
        - CrossEncoder 打分 → sigmoid 归一化 → 覆盖头部 score 重排；尾部原序 append
        - 【F4】unscorable（空 content）保持其在 head 中的原始相对位置（不统一 append
          到可打分节点之后）
        - 【F5】predict 返回长度失配 → 静默降级原列表（防部分候选被丢）
        - 【F6】NaN/Inf 防护：nan_to_num 后 sigmoid，防 SDK 返回 NaN/Inf 破坏 score 契约
        - 任何异常 → 原列表返回（静默降级，主检索零回归）

        顺序契约：调用方（_finish）先执行 _deduplicate_and_sort（去重+boost+钳制），
        再调用本方法；返回后顺序即最终顺序，不再二次排序。
        """
        if not enabled or getattr(self, "_rerank_failed", False) or len(results) < 2:
            return results
        try:
            reranker = self._get_reranker()
            if reranker is None:
                return results
            k = min(int(self.config.rerank_input_k), len(results))
            head = results[:k]
            tail = results[k:]
            scorable: list[tuple[str, dict]] = []
            for r in head:
                content = str(r.get("content", "") or "").strip()
                if content:
                    scorable.append((content, r))
            if len(scorable) < 2:
                return results
            query_text = raw_query or ""
            pairs = [(query_text, c[:3000]) for c, _ in scorable]
            scores = reranker.predict(pairs)
            logits = np.asarray(scores, dtype=np.float64).reshape(-1)
            # 【F5】长度失配 → 静默降级原列表（防部分候选被丢）
            if len(logits) != len(scorable):
                return results
            # 【F6】NaN/Inf 防护：SDK 返回 NaN/Inf 会破坏 EpisodicResult score 契约
            # （nan→0.0，±inf→±50，经下方 clip+sigmoid 收敛到 (0,1) 有效分）
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            # sigmoid 归一化：logit 可能极大 → clip 防 exp 溢出（numpy 溢出 warning）
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            order = np.argsort(-probs)
            reranked_scorable: list[dict] = []
            for idx in order:
                i = int(idx)
                r = scorable[i][1]
                r["score"] = float(probs[i])
                reranked_scorable.append(r)
            # 【F4】按 head 顺序归并：scorable 位置填入重排后的下一候选，
            # 空 content 位置原样保留其原始相对位置（而非 append 到所有可打分节点后）。
            merged: list[dict] = []
            sc_iter = iter(reranked_scorable)
            for r in head:
                if str(r.get("content", "") or "").strip():
                    merged.append(next(sc_iter))
                else:
                    merged.append(r)
            return merged + tail
        except Exception:
            logger.debug(
                "rerank degraded, returning original results", exc_info=True
            )
            return results

    @staticmethod
    def _deduplicate_and_sort(results: list[dict]) -> list[dict]:
        """按 content[:100] 去重并按 score 降序排列（core 轨 ×1.1 / 画像命中 ×1.2）。

        【P1】分数契约：boost 后钳制 score ∈ [0, 1]——EpisodicResult.score 约束
        le=1.0（api/models.py），越界会让 /memories/retrieve 构造响应抛
        ValidationError → 500（如 1.0×1.2=1.2）。钳制不破坏排序语义（单调）。
        """
        # 【Core-Boost】core 事实温和加分 + 【Profile-Boost】画像值命中加分
        profile_vals = profile_values(_USER_PROFILE)
        for r in results:
            if r.get("fact_track") == "core":
                r["score"] *= 1.1
            if profile_hit(r.get("content", ""), profile_vals):
                r["score"] *= 1.2
            # 【P1】钳制上界 1.0：core ×1.1 / 画像 ×1.2 可能突破 1.0，
            # 契约越界 → EpisodicResult ValidationError → 生产检索 500
            r["score"] = min(1.0, r["score"])
        seen: set[str] = set()
        unique: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            content = r.get("content", "")
            key = (content or r.get("node_id") or "")[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def search_profile(self, query: str) -> dict:
        """旁路：返回与 query 相关的画像上下文块（prepend 到 prompt，不参与主排序）。

        消费点：api/routes/search.py /memories/retrieve 注入响应
        RetrieveResponse.profile_context（SelfEvolvingRetrieval 包装时经 _qr 解包调用）。
        """
        if not _USER_PROFILE:
            return {"matched": False, "context": ""}
        hits = []
        for group, entries in _USER_PROFILE.items():
            if not isinstance(entries, dict):
                continue
            for value, info in entries.items():
                if value and str(value) in query:
                    weight = info.get("weight", 0.0) if isinstance(info, dict) else 0.0
                    hits.append({"group": group, "value": value, "weight": weight})
        if not hits:
            return {"matched": False, "context": ""}
        lines = "\n".join(f"- {h['group']}: {h['value']} (weight {h['weight']:.1f})" for h in hits)
        return {"matched": True, "context": f"【用户画像】\n{lines}", "hits": hits}

    @staticmethod
    def _filter_archived(results: list[dict]) -> list[dict]:
        """排除已归档节点（archived=true）。旧节点无 archived 字段视为未归档，不排除。"""
        return [r for r in results
                if r.get("archived") not in (True, "true", 1)]

    def _lookup_archived_ids(self, node_ids: list[str]) -> set[str]:
        """批量回查节点归档状态，返回已归档 id 集合（回查失败返回空集，不误过滤）。"""
        if not node_ids or self.graphlite_store is None:
            return set()
        try:
            if not hasattr(self.graphlite_store, "get_episodes_batch"):
                return set()
            eps = self.graphlite_store.get_episodes_batch(list(dict.fromkeys(node_ids)))
            if not isinstance(eps, (list, tuple)):
                return set()
            return {
                str(ep.get("id", ""))
                for ep in eps
                if isinstance(ep, dict) and ep.get("archived") in (True, "true", 1)
            }
        except Exception:
            return set()

    def _lookup_fact_track(self, node_ids: list[str]) -> dict[str, str]:
        """批量回查节点 fact_track，返回 {node_id: fact_track}（回查失败返回空，缺省 active）。"""
        if not node_ids or self.graphlite_store is None:
            return {}
        try:
            if not hasattr(self.graphlite_store, "get_episodes_batch"):
                return {}
            eps = self.graphlite_store.get_episodes_batch(list(dict.fromkeys(node_ids)))
            if not isinstance(eps, (list, tuple)):
                return {}
            return {
                str(ep.get("id", "")): ep.get("fact_track", "active")
                for ep in eps
                if isinstance(ep, dict)
            }
        except Exception:
            return {}
