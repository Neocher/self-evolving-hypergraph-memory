"""
YAML 配置加载器 + 环境变量覆盖
=============================
从 config/defaults.yaml 加载默认配置，支持 SHM_<SECTION>__<KEY> 环境变量覆盖。

示例:
    export SHM_TAU__TAU_DECAY_SECONDS=3600
    export SHM_KUZU__DATABASE_PATH=/data/shm_prod
    export SHM_API__PORT=8080
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from typing import get_type_hints

from core.ontology_validator import OntologyConfig
# 【v5.45.0 P2-2】复用 core.defense 版 DefenseConfig —— 消除双定义永久漂移
# （同 v5.44.1 OntologyConfig 方案 A）。core.defense 不依赖 config.settings,
# 无循环 import; 字段一致由 _build_settings 的 __dataclass_fields__ 过滤保证。
from core.defense import DefenseConfig


def _get_defaults_path() -> Path:
    """获取 defaults.yaml 的绝对路径（相对于本文件所在目录）。"""
    return Path(__file__).resolve().parent / "defaults.yaml"


@dataclass
class TauConfig:
    tau_initial: float = 1.0
    tau_decay_seconds: float = 1800.0
    decay_threshold: float = 0.1
    refresh_on_access: bool = True


@dataclass
class HebbianConfig:
    k_sparsity: int = 8
    learning_rate: float = 0.1
    decay_constant: float = 0.01
    activation_threshold: float = 0.3
    max_connections_per_node: int = 64


@dataclass
class SSMConfig:
    hidden_dim: int = 128
    input_dim: int = 8
    gate_threshold: float = 0.5
    state_decay: float = 0.9
    seed: int = 42
    feat_mean_activation: int = 0
    feat_age_hours: int = 1
    feat_access_freq: int = 2
    feat_member_count: int = 3
    feat_community_density: int = 4
    feat_tau_mean: int = 5
    feat_tau_variance: int = 6
    feat_connection_entropy: int = 7


@dataclass
class DreamConfig:
    idle_timeout_seconds: int = 300
    accum_threshold: int = 100
    min_interval_seconds: int = 60
    max_dream_duration_seconds: int = 450
    cpu_affinity_low_priority: bool = True
    memory_limit_mb: int = 256


@dataclass
class GraphLiteConfig:
    database_path: str = "./data/shm_graphlite_db"
    buffer_pool_size_mb: int = 256
    max_threads: int = 4


@dataclass
class HNSWConfig:
    """OverGraph HNSW 索引参数（D11 graph.hnsw.*）。

    仅 ef_search 生效（vector_search(ef_search=) 透传，见 OverGraphStore.
    vector_search_dense）。m/ef_construction 无 SDK 设置 API（open() 拒绝
    未知选项，实证）→ 死配置已移除。
    """
    ef_search: int = 64


@dataclass
class GraphConfig:
    """图后端单开关（v6.0.0 OverGraph 迁移，设计 A8/D11）"""
    backend: str = "graphlite"      # graphlite|overgraph（默认 graphlite → 存量零感知）
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)


@dataclass
class OverGraphConfig:
    """OverGraph 引擎配置（design_overgraph_store.md 任务书 8）"""
    database_path: str = "./data/shm_overgraph_db"
    dense_vector_dimension: int = 512
    # R1 P2#9 实证：vector_search score 恒为 cosine —— l2/cosine 双开库同向量
    # 对拍 score 逐位一致（引擎忽略选项）。字段保留仅为引擎 open() 参数兼容。
    dense_vector_metric: str = "cosine"


@dataclass
class FAISSConfig:
    dimension: int = 512
    index_type: str = "IVFFlat"
    nlist: int = 100
    nprobe: int = 10


@dataclass
class CommunityExpansionConfig:
    """社区扩召回配置（v5.41.0 Community-Expansion）"""
    enabled: bool = True        # 开关：关闭时行为 = 现状（bit 级一致）
    boost: float = 0.6          # 扩展分 = relevance × min(种子分) × boost（相对尾分缩放）
    threshold: float = 0.5      # 社区相关度闸口（BM25-on-summary relevance < threshold 丢弃）
    max_members: int = 10       # 每社区最大成员召回数


@dataclass
class VisualRecallConfig:
    """视觉检索配置（v5.46.0 P2-a V-Mem 模态路由）"""
    enabled: bool = True        # 开关：关闭时行为 = 现状（bit 级一致）
    boost: float = 0.6          # 视觉分 = 1/(1+dist) × min(种子分) × boost（相对尾分缩放，严格低于文本种子）
    max_results: int = 5        # 每查询最多追加的视觉结果数
    visual_limit: int = 10000   # prewarm 拉取 VisualNode 上限


@dataclass
class MesaConfig:
    """MESA 记忆增强检索配置（v5.49.0 MESA 合成节点通道）"""
    enabled: bool = False       # 开关：关闭时行为 = 现状（bit 级一致；默认关，与 community 默认开不同）
    boost: float = 0.4          # 合成分 = relevance × min(种子分) × boost（严格 < community_expansion.boost=0.6）
    threshold: float = 0.5      # BM25-on-summary 相关度阈值（对齐 community_expansion）
    max_nodes: int = 5          # 每查询最多合成节点数

    def __post_init__(self) -> None:
        # 【P2-2】max_nodes 下界校验：0/负数会让 _mesa_synthesis 先 append 再
        # break 仍合成 1 条——配置期 fail-fast 拒绝非法值。
        if self.max_nodes < 1:
            raise ValueError(f"MesaConfig.max_nodes={self.max_nodes} 必须 >= 1")
        # 【P2-3】boost 上界校验：严格 < community_expansion.boost=0.6（取 0.59
        # 与 self_evolving._MESA_BOOST_MAX 对齐，float 精度下不越 0.6）——配置期
        # fail-fast 拒绝非法值；community boost 调低场景由 _mesa_synthesis 运行时
        # clamp（min(boost, community_boost*0.95)）兜底。
        if not 0.0 <= self.boost <= 0.59:
            raise ValueError(f"MesaConfig.boost={self.boost} 超出 [0, 0.59]")


@dataclass
class EntityExpansionConfig:
    """实体扩召回配置（v5.53.0 P3c 跨消息多跳增强）"""
    enabled: bool = True        # 开关：关闭时行为 = 现状（bit 级一致）
    boost: float = 0.9          # 扩展分 = max(种子分) × boost（仅低于最高种子）
    max_results: int = 10       # 每实体最大召回数
    max_entities: int = 3       # 每查询最多提取实体数
    time_filter: bool = True    # 时间锚上界过滤（created_at <= session 时间锚）

    def __post_init__(self) -> None:
        # 【R1 P2-1】boost/max_results/max_entities 校验（镜像 MesaConfig 做法）：
        # boost>=1 让扩展分反超最高种子（违反「仅低于最高种子」契约）、max<=0 让扩展
        # 静默空返回——配置期 fail-fast 拒绝非法值。
        if not 0.0 <= self.boost < 1.0:
            raise ValueError(f"EntityExpansionConfig.boost={self.boost} 必须 ∈ [0, 1)")
        if self.max_results < 1:
            raise ValueError(f"EntityExpansionConfig.max_results={self.max_results} 必须 >= 1")
        if self.max_entities < 1:
            raise ValueError(f"EntityExpansionConfig.max_entities={self.max_entities} 必须 >= 1")


@dataclass
class RetrievalConfig:
    top_k_l1: int = 5              # L1 FAISS 检索 top-K
    top_k_vector: int = 20         # L2 向量检索 top-K
    top_k_keyword: int = 20        # L3 关键词检索 top-K
    top_k_episodes: int = 20       # 中粒度：展开的情节数
    top_k_facts: int = 50          # 细粒度：最终返回的事实数
    score_threshold: float = 0.3   # 得分阈值过滤
    use_tau_rerank: bool = True    # 是否使用 τ 值重排序
    tau_weight: float = 0.4        # τ 值权重（混合模式）
    vector_weight: float = 0.6     # 向量相似度权重（混合模式）
    community_expansion: CommunityExpansionConfig = field(default_factory=CommunityExpansionConfig)
    visual_recall: VisualRecallConfig = field(default_factory=VisualRecallConfig)
    mesa: MesaConfig = field(default_factory=MesaConfig)
    entity_expansion: EntityExpansionConfig = field(default_factory=EntityExpansionConfig)
    agentic_enabled: bool = False   # P0-2 多步锚点检索编排开关（默认关）
    rerank_enabled: bool = True     # bge-reranker 重排开关（默认开；仅 FUSION 生效）
    rerank_input_k: int = 40        # 送入 reranker 的头部候选数（尾部保持原序 append）
    hyde_enabled: bool = False      # P3b HyDE 假设文档增强开关（默认关；仅 FUSION 生效）
    hyde_mode: str = "dual"         # P3b HyDE 模式：dual（双路合并）/ replace（仅假设向量）
    hyde_timeout: float = 1.5       # P3b HyDE LLM 生成超时（秒），失败静默降级单路

    def __post_init__(self) -> None:
        # rerank_input_k 下界校验：0/负数会让 _rerank_results 取空头部分支，
        # 静默跳过重排（配置期 fail-fast 拒绝非法值）。
        if self.rerank_input_k < 1:
            raise ValueError(f"RetrievalConfig.rerank_input_k={self.rerank_input_k} 必须 >= 1")
        # 【P3b】hyde_mode 枚举校验：检索路径只分 dual/replace 两分支，非法值会
        # 静默落回单路与操作者意图不符（配置期 fail-fast 拒绝非法值）。
        if self.hyde_mode not in ("dual", "replace"):
            raise ValueError(f"RetrievalConfig.hyde_mode={self.hyde_mode} 必须 ∈ {{dual, replace}}")


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "auto"


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "WARNING"
    cors_origins: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8000", "http://localhost:8000"])


@dataclass
class CircuitBreakerConfig:
    failure_threshold: float = 0.5
    recovery_timeout: float = 30.0
    half_open_max_requests: int = 1
    window_size: int = 10


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    backoff: float = 2.0
    max_total_timeout: float = 0.0


@dataclass
class CommunityConfig:
    template_threshold: int = 5
    jaccard_threshold: float = 0.8
    max_keywords: int = 10


@dataclass
class LLMConfig:
    """LLM 客户端配置（梦境·本体·关系抽取）"""
    endpoint: str = "http://127.0.0.1:11434/v1/chat/completions"
    model: str = "llama3"
    api_key: str = ""
    timeout: float = 60.0
    max_retries: int = 3
    fallback_endpoints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """空 API key 时发出警告（防止从 Ollama 切换到云服务时静默无认证）。"""
        if not self.api_key and self.endpoint and "ollama" not in self.endpoint.lower() and "127.0.0.1" not in self.endpoint:
            import logging
            logging.getLogger(__name__).warning(
                "LLMConfig: api_key is empty but endpoint '%s' is not a local service. "
                "Set SHM_LLM__API_KEY or configure api_key in defaults.yaml.",
                self.endpoint,
            )

    def __repr__(self) -> str:
        masked = (self.api_key[:6] + "..." + self.api_key[-4:]) if len(self.api_key) > 10 else ("*****" if self.api_key else "")
        return (
            f"LLMConfig(endpoint={self.endpoint!r}, model={self.model!r}, "
            f"api_key={masked!r}, timeout={self.timeout}, "
            f"max_retries={self.max_retries}, "
            f"fallback_endpoints={self.fallback_endpoints!r})"
        )


@dataclass
class SHMClientConfig:
    """SHM 服务端点（供 MCP/CLI/SDK 客户端连接）"""
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 15.0
    mcp_http_port: int = 8222


@dataclass
class HealthConfig:
    """健康检查阈值（内存监控）"""
    memory_warning_mb: int = 2048
    memory_critical_mb: int = 3072


@dataclass
class Settings:
    """SHM v4.0 全局配置聚合。"""

    tau: TauConfig = field(default_factory=TauConfig)
    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    ssm: SSMConfig = field(default_factory=SSMConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    graphlite: GraphLiteConfig = field(default_factory=GraphLiteConfig)
    overgraph: OverGraphConfig = field(default_factory=OverGraphConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    shm_client: SHMClientConfig = field(default_factory=SHMClientConfig)
    health: HealthConfig = field(default_factory=HealthConfig)


def _env_override(raw: dict[str, Any], prefix: str = "SHM") -> dict[str, Any]:
    """
    用环境变量覆盖 YAML 配置值。

    环境变量格式: SHM_<SECTION>__<KEY>
    示例: SHM_TAU__TAU_DECAY_SECONDS=3600 覆盖 raw["tau"]["tau_decay_seconds"]

    支持类型推断：字符串 'true'/'false' → bool，纯数字 → int/float。
    """
    result: dict[str, Any] = dict(raw)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}_"):
            continue
        parts = env_key[len(prefix) + 1 :].lower().split("__")
        if len(parts) != 2:
            continue
        section, key = parts[0], parts[1]
        if section not in result:
            result[section] = {}
        typed_val = _coerce_value(env_val)
        result[section][key] = typed_val
    return result


def _coerce_value(raw: str) -> Any:
    """将环境变量字符串推断为适当的 Python 类型。"""
    low = raw.lower()
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _build_settings(raw: dict[str, Any]) -> Settings:
    """从原始字典构建 Settings 实例。"""
    section_map: dict[str, type] = {
        "tau": TauConfig,
        "hebbian": HebbianConfig,
        "ssm": SSMConfig,
        "dream": DreamConfig,
        "graph": GraphConfig,
        "graphlite": GraphLiteConfig,
        "overgraph": OverGraphConfig,
        "faiss": FAISSConfig,
        "retrieval": RetrievalConfig,
        "embedding": EmbeddingConfig,
        "api": APIConfig,
        "circuit_breaker": CircuitBreakerConfig,
        "retry": RetryConfig,
        "community": CommunityConfig,
        "ontology": OntologyConfig,
        "llm": LLMConfig,
        "shm_client": SHMClientConfig,
        "defense": DefenseConfig,
        "health": HealthConfig,
    }
    kwargs: dict[str, Any] = {}
    for section, cls in section_map.items():
        section_data = raw.get(section, {})
        flds = {k: v for k, v in section_data.items()
                if k in cls.__dataclass_fields__}
        # 嵌套 dataclass 字段（如 retrieval.community_expansion）：YAML dict → 实例。
        # 用 get_type_hints 解析真实类型（settings.py 有 from __future__ import
        # annotations，field.type 是字符串，不能直接 hasattr）。
        hints = get_type_hints(cls)
        for fname, fval in flds.items():
            ftype = hints.get(fname)
            if isinstance(fval, dict) and hasattr(ftype, "__dataclass_fields__"):
                flds[fname] = ftype(**{k: v for k, v in fval.items()
                                       if k in ftype.__dataclass_fields__})
        kwargs[section] = cls(**flds)
    return Settings(**kwargs)


def load_settings(yaml_path: Optional[Path] = None) -> Settings:
    """
    加载配置：YAML 文件 + 环境变量覆盖。

    Args:
        yaml_path: YAML 文件路径（None = 使用 config/defaults.yaml）

    Returns:
        构建好的 Settings 实例
    """
    path = yaml_path or _get_defaults_path()
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    overridden = _env_override(raw)
    return _build_settings(overridden)


# 模块级单例：首次导入时自动加载
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局 Settings 单例（惰性加载）。

    首次调用时自动从 config/defaults.yaml 加载并应用环境变量覆盖。
    """
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings(yaml_path: Optional[Path] = None) -> Settings:
    """
    强制重新加载配置（不读取缓存）。

    测试或运行时热更新时使用。
    """
    global _settings
    _settings = load_settings(yaml_path)
    return _settings
