"""
Pydantic 请求/响应模型
=====================
所有 FastAPI 路由的请求体和响应体数据结构。
Hyperedge 成员数 ≥2 由 field_validator 校验。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── 枚举 ──────────────────────────────────────────────────

# SourceTag → 废弃 (v5.8.3 relaxed to str for multi-agent support)


class HyperedgeType(str, Enum):
    EPISODE = "episode"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"


class TriggerMode(str, Enum):
    IDLE = "idle"
    ACCUMULATED = "accum"
    EXPLICIT = "explicit"


class SourceType(str, Enum):
    """来源信任分级（写时来源类型，P3）。

    direct   — 用户直接观察/直述（仅 source == "user" 允许）
    tool     — 工具/系统桥接写入（MCP/CLI 等显式工具调用）
    inferred — agent 推理/系统提升内容（LLM 生成、梦境提升）
    """
    DIRECT = "direct"
    TOOL = "tool"
    INFERRED = "inferred"


def resolve_source_type(source: str, declared_source_type: str) -> str:
    """防洗白：agent 来源（非 "user"）不得声明 direct。

    规则：只有 source == "user" 才能落 direct；agent（hermes/codex/claude/
    opencode 等）写入的 LLM 内容若声明 direct → 强制降级 inferred。
    tool / inferred 声明及 source == "user" 的 direct 原样放行。
    """
    st = getattr(declared_source_type, "value", declared_source_type)
    if st == "direct" and source != "user":
        return "inferred"
    return st


# ─── Episode 请求/响应 ─────────────────────────────────────

class EpisodeCreate(BaseModel):
    """创建情节节点请求（Layer2）。"""
    content: str = Field(..., min_length=1, max_length=100_000,
                         description="情节内容文本")
    source: str = Field(default="user", description="来源标识（agent名称，如 hermes/codex/claude）")
    source_type: SourceType = Field(default=SourceType.DIRECT, description="来源信任分级: direct(用户直述) / tool(工具桥接) / inferred(agent推理/系统提升)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")
    force_promote: bool = Field(default=False, description="是否绕过 τ 阈值强制提升")
    namespace: Optional[str] = Field(default=None, description="命名空间（用于图隔离，如 mirofish_xxx）")
    visibility: str = Field(default="private", description="可见性: private(仅当前namespace) / shared(所有Agent可检索)")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v.strip()


class EpisodeResponse(BaseModel):
    """情节节点创建响应。"""
    episode_id: str = Field(..., description="新创建的情节节点 ID")
    content: str = Field(..., description="情节内容（裁剪后）")
    tau_initial: float = Field(default=1.0, description="初始 τ 值")
    tau_decay_seconds: float = Field(default=1800.0, description="τ 衰减时间常数（秒）")
    created_at: float = Field(..., description="创建时间戳（Unix 秒）")
    source: str = Field(default="user", description="来源标签")
    status: str = Field(default="created", description="创建状态: created/filtered/error")
    error: Optional[str] = Field(default=None, description="拒绝原因（写入被拒绝时非空）")


# ─── Hyperedge 请求/响应 ───────────────────────────────────

class HyperedgeCreate(BaseModel):
    """创建超边请求（Layer4）。"""
    type: HyperedgeType = Field(..., description="超边类型")
    member_ids: List[str] = Field(..., min_length=2,
                                  description="成员节点 ID 列表（至少 2 个）")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")
    # 以下字段按超边类型选用
    topic: Optional[str] = Field(default=None, description="情节超边主题")
    conclusion: Optional[str] = Field(default=None, description="语义超边结论")
    start_time: Optional[float] = Field(default=None, description="时态超边起始")
    end_time: Optional[float] = Field(default=None, description="时态超边结束")

    @field_validator("member_ids")
    @classmethod
    def at_least_two_members(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("Hyperedge must have at least 2 member nodes")
        return v


class HyperedgeResponse(BaseModel):
    """超边查询/创建响应。"""
    id: str = Field(..., description="超边 ID")
    type: HyperedgeType = Field(..., description="超边类型")
    member_ids: List[str] = Field(..., description="成员节点 ID 列表")
    created_at: float = Field(..., description="创建时间戳（Unix 秒）")
    gate_value: float = Field(default=1.0, description="SSM 门控值 (0~1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class HyperedgeListResponse(BaseModel):
    """超边列表响应。"""
    hyperedges: List[HyperedgeResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="超边总数")


# ─── 感觉缓冲区 ────────────────────────────────────────────

class SensoryRecord(BaseModel):
    """Layer1 感觉缓冲区写入请求。"""
    content: str = Field(..., min_length=1, max_length=100_000,
                         description="原始文本内容")
    source: str = Field(default="user", description="来源标识（agent名称，如 hermes/codex/claude）")
    source_type: SourceType = Field(default=SourceType.DIRECT, description="来源信任分级: direct(用户直述) / tool(工具桥接) / inferred(agent推理/系统提升)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")
    namespace: Optional[str] = Field(default=None, description="命名空间（用于图隔离）")
    visibility: str = Field(default="private", description="可见性: private(仅当前namespace) / shared(所有Agent可检索)")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v.strip()


class SensoryResponse(BaseModel):
    """感觉记录写入响应。"""
    status: str = Field(default="ok")
    record_id: str = Field(..., description="分配的记录 ID")
    buffer_usage: int = Field(default=0, description="当前缓冲区使用量")
    error: Optional[str] = Field(default=None, description="拒绝原因（写入被拒绝时非空）")


# ─── 提升 ──────────────────────────────────────────────────

class PromoteRequest(BaseModel):
    """Layer1 → Layer2 提升请求。"""
    sensory_record_id: str = Field(..., description="Layer1 记录 ID")
    force: bool = Field(default=False, description="是否强制提升")


class PromoteResponse(BaseModel):
    """提升响应。"""
    episode_id: str = Field(..., description="新创建的 Layer2 情节节点 ID")
    tau_initial: float = Field(default=1.0)
    hebbian_connections_updated: int = Field(default=0)


# ─── 检索 ──────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    """检索请求。"""
    query: str = Field(..., min_length=1, max_length=10_000, description="查询文本")
    top_k: int = Field(default=20, ge=1, le=200, description="返回结果数上限")
    strategy: Optional[str] = Field(default="auto", description="检索策略: auto|tau_first|vector_first|hybrid")
    include_audit: bool = Field(default=False, description="是否附带溯源信息")
    namespace: Optional[str] = Field(default=None, description="限定检索的命名空间")
    include_shared: bool = Field(default=True, description="是否同时检索 visibility=shared 的记忆")
    include_archived: bool = Field(default=False, description="是否包含已归档记忆（默认排除）")
    session_ts: Optional[float] = Field(
        default=None,
        description="session 时间锚（相对时间词解析基准，None 回落墙钟）",
    )


class EpisodicResult(BaseModel):
    """单条检索结果。"""
    node_id: str = Field(..., description="节点 ID")
    content: str = Field(default="", description="节点内容")
    score: float = Field(ge=0.0, le=1.0, description="相关性得分")
    tau_value: Optional[float] = Field(default=None, description="τ 值")
    source: str = Field(default="episodic", description="来源层")
    hyperedge_id: Optional[str] = Field(default=None, description="所属超边 ID")
    community_id: Optional[str] = Field(default=None, description="所属社区 ID")
    created_at: Optional[float] = Field(default=None, description="创建时间戳")
    retrieval_level: str = Field(default="hypergraph", description="来自哪级检索")
    risk_level: Optional[str] = Field(default=None, description="R6 内容风险级别 none/high/critical")
    modality: Optional[str] = Field(default=None, description="模态: episodic/text/visual（P2-a V-Mem 视觉通道）")


class RetrieveResponse(BaseModel):
    """检索响应。"""
    query: str = Field(..., description="原始查询文本")
    strategy_used: str = Field(default="auto", description="实际使用的检索策略")
    results: List[EpisodicResult] = Field(default_factory=list, description="检索结果列表")
    total_found: int = Field(default=0, ge=0, description="检索到的结果总数")
    latency_ms: float = Field(default=0.0, description="检索耗时（毫秒）")
    degraded: bool = Field(default=False, description="是否触发了检索降级")
    profile_context: Optional[str] = Field(
        default=None,
        description="用户画像旁路上下文块（search_profile 命中时注入，消费方 prepend 到 prompt，不参与主排序）",
    )


# ─── 梦境 ──────────────────────────────────────────────────

class DreamReport(BaseModel):
    """梦境执行报告（由 DreamPipeline.run() 返回）。"""
    dream_id: str = Field(..., description="梦境周期 ID")
    trigger_mode: TriggerMode = Field(..., description="触发模式")
    timestamp: float = Field(..., description="启动时间戳")
    duration_seconds: float = Field(..., description="执行耗时（秒）")
    stats: Dict[str, int] = Field(default_factory=dict, description="{created, updated, deleted}")
    community_count: int = Field(default=0, ge=0)
    prune_count: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)
    audit_block_hash: str = Field(default="", description="溯源区块哈希")
    compressed_topics: int = Field(default=0)
    compressed_episodes: int = Field(default=0)
    compressed_facts: int = Field(default=0)
    keywords_extracted: int = Field(default=0)


class DreamTriggerResponse(BaseModel):
    """显式梦境触发响应。"""
    accepted: bool = Field(..., description="是否接受触发请求")
    message: str = Field(default="", description="状态说明")


# ─── 梦境候选（非破坏性梦境审查） ──────────────────────────


class DreamCandidateSummary(BaseModel):
    """梦境候选列表项（供 review 列表使用）。"""
    dream_id: str = Field(..., description="梦境 ID")
    created_at: float = Field(..., description="创建时间戳")
    trigger_mode: str = Field(default="unknown", description="触发模式")
    community_count: int = Field(default=0, description="社区数")
    prune_count: int = Field(default=0, description="剪枝节点数")
    conflict_count: int = Field(default=0, description="冲突解决数")
    stats: Dict[str, int] = Field(default_factory=dict, description="{created, updated, deleted}")


class DreamCandidateListResponse(BaseModel):
    """梦境候选列表响应。"""
    candidates: List[DreamCandidateSummary] = Field(default_factory=list)
    total: int = Field(default=0, description="待审查候选总数")


class DreamCandidateDetail(BaseModel):
    """梦境候选详情（供 review）。"""
    dream_id: str = Field(..., description="梦境 ID")
    created_at: float = Field(..., description="创建时间戳")
    trigger_mode: str = Field(default="unknown", description="触发模式")
    stats: Dict[str, int] = Field(default_factory=dict)
    community_count: int = Field(default=0)
    prune_count: int = Field(default=0)
    conflict_count: int = Field(default=0)
    community_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="社区摘要列表（含 report/keywords/patterns/contradictions）",
    )
    prune_ops: List[Dict[str, str]] = Field(
        default_factory=list,
        description="将要删除的节点操作",
    )
    merge_ops: List[Dict[str, str]] = Field(
        default_factory=list,
        description="将要合并的节点操作",
    )
    applied: bool = Field(default=False)
    discarded: bool = Field(default=False)


class DreamApplyResponse(BaseModel):
    """梦境候选应用响应。"""
    success: bool = Field(..., description="是否应用成功")
    dream_id: str = Field(..., description="梦境 ID")
    message: str = Field(default="", description="状态说明")


# ─── 溯源 ──────────────────────────────────────────────────

class AuditOperation(BaseModel):
    """单条溯源操作记录。"""
    op_type: str = Field(..., pattern="^(create|update|delete)$", description="操作类型")
    node_id: str = Field(..., description="受影响的节点 ID")
    old_value: Optional[str] = Field(default=None, description="变更前值")
    new_value: Optional[str] = Field(default=None, description="变更后值")
    reason: str = Field(default="explicit", description="操作原因: tau_decay|hebbian_prune|adaptive_gate|community_merge|explicit")
reason: str = ""  # 'tau_decay' | 'hebbian_prune' | 'adaptive_gate' | 'community_merge' | 'explicit'

class AuditTrace(BaseModel):
    """节点溯源链响应。"""
    node_id: str = Field(..., description="查询的节点 ID")
    operations: List[AuditOperation] = Field(default_factory=list,
                                              description="该节点的完整变更历史")
    chain_verified: bool = Field(default=True, description="溯源链完整性验证结果")
    total_blocks: int = Field(default=0, description="溯源链总区块数")


# ─── 健康检查 ──────────────────────────────────────────────

class HealthStatus(BaseModel):
    """深度健康检查响应。"""
    status: str = Field(..., pattern="^(ok|degraded|error)$",
                        description="服务整体状态")
    graph_connected: bool = Field(..., description="图数据库连接状态 (RyuGraph)")
    faiss_loaded: bool = Field(..., description="FAISS 索引加载状态")
    dream_scheduler_running: bool = Field(default=False, description="梦境调度器运行状态")
    stats: Dict[str, Any] = Field(default_factory=dict, description="详细统计数据")
    timestamp: float = Field(default=0.0, description="检查时间戳")


# ─── 社区 ──────────────────────────────────────────────────

class CommunityInfo(BaseModel):
    """社区摘要信息。"""
    id: str = Field(..., description="社区 ID")
    name: str = Field(default="", description="社区名称")
    summary: str = Field(default="", description="社区摘要")
    member_count: int = Field(default=0, description="成员节点数")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    leiden_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Leiden 算法得分")


class CommunityListResponse(BaseModel):
    """社区列表响应。"""
    communities: List[CommunityInfo] = Field(default_factory=list)
    total: int = Field(default=0)


# ─── 配置（API 透出） ──────────────────────────────────────

# ─── 向量搜索 ──────────────────────────────────────────────


class SearchVectorRequest(BaseModel):
    """向量搜索请求。"""
    query: str = Field(..., min_length=1, max_length=10_000, description="查询文本")
    limit: int = Field(default=10, ge=1, le=200, description="返回结果数上限")


class SearchVectorResult(BaseModel):
    """单条向量搜索结果。"""
    node_id: str = Field(..., description="节点 ID")
    content: str = Field(default="", description="节点内容")
    score: float = Field(ge=0.0, le=1.0, description="余弦相似度得分")
    faiss_id: int = Field(..., description="FAISS 内部 ID")


class SearchVectorResponse(BaseModel):
    """向量搜索结果响应。"""
    query: str = Field(..., description="原始查询文本")
    results: List[SearchVectorResult] = Field(default_factory=list, description="搜索结果列表")
    total_found: int = Field(default=0, ge=0, description="结果总数")
    latency_ms: float = Field(default=0.0, description="检索耗时（毫秒）")
    degraded: bool = Field(default=False, description="是否因组件不可用而降级")


class TauDecayConfig(BaseModel):
    """τ 衰减配置（API 可见子集）。"""
    tau_initial: float = Field(default=1.0, gt=0.0, le=1.0)
    tau_decay_seconds: float = Field(default=1800.0, gt=0.0)
    decay_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    refresh_on_access: bool = Field(default=True)


class HebbianConfig(BaseModel):
    """Hebbian 学习配置（API 可见子集）。"""
    k_sparsity: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    decay_constant: float = Field(default=0.01, gt=0.0, le=1.0)
    activation_threshold: float = Field(default=0.3, gt=0.0, le=1.0)


# ─── Ontology v2 ─────────────────────────────────────────────

class AttributeDefModel(BaseModel):
    """属性定义"""
    name: str = Field(..., description="属性名")
    type: str = Field(default="string", description="属性类型: string|integer|float|boolean|date|datetime|string[]|text_embedding|entity_ref")
    required: bool = Field(default=False, description="是否必填")
    indexed: bool = Field(default=False, description="是否建索引")
    description: str = Field(default="", description="属性描述")
    default: Any = Field(default=None, description="默认值")
    min_value: Optional[float] = Field(default=None, description="数值最小值")
    max_value: Optional[float] = Field(default=None, description="数值最大值")
    enum_values: Optional[List[str]] = Field(default=None, description="枚举值列表")


class EntityTypeDefModel(BaseModel):
    """实体类型定义"""
    name: str = Field(..., description="类型名")
    description: str = Field(default="", description="描述")
    parent: Optional[str] = Field(default=None, description="父类型")
    attributes: List[AttributeDefModel] = Field(default_factory=list, description="属性列表")


class EdgeAttributeDefModel(BaseModel):
    """边属性定义"""
    name: str = Field(..., description="属性名")
    type: str = Field(default="string", description="属性类型")
    required: bool = Field(default=False)
    description: str = Field(default="")


class EdgeTypeDefModel(BaseModel):
    """边类型定义"""
    name: str = Field(..., description="边类型名")
    description: str = Field(default="")
    source_types: List[str] = Field(default_factory=list, description="允许的源实体类型（空=全部）")
    target_types: List[str] = Field(default_factory=list, description="允许的目标实体类型（空=全部）")
    attributes: List[EdgeAttributeDefModel] = Field(default_factory=list)
    symmetry: bool = Field(default=False, description="是否对称")


class EntityTypeListResponse(BaseModel):
    entity_types: List[EntityTypeDefModel] = Field(default_factory=list)
    total: int = Field(default=0)


class EdgeTypeListResponse(BaseModel):
    edge_types: List[EdgeTypeDefModel] = Field(default_factory=list)
    total: int = Field(default=0)


class OntologyStatsResponse(BaseModel):
    entity_type_count: int = Field(default=0)
    edge_type_count: int = Field(default=0)
    baseline_loaded: bool = Field(default=True)


# ─── v2.0: 工作记忆（Session Memory）──────────────────────


class SessionMemoryCreate(BaseModel):
    """工作记忆写入请求"""
    content: str = Field(..., min_length=1, max_length=50_000,
                         description="记忆内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据（如agent_id, round_num）")


class SessionMemoryItem(BaseModel):
    """单条工作记忆"""
    id: str = Field(..., description="记忆ID")
    session_id: str = Field(..., description="会话ID")
    content: str = Field(..., description="记忆内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: float = Field(..., description="创建时间戳")


class SessionMemoryListResponse(BaseModel):
    """工作记忆列表响应"""
    session_id: str = Field(..., description="会话ID")
    results: List[SessionMemoryItem] = Field(default_factory=list, description="记忆列表")
    total: int = Field(default=0, description="总数")


# ─── 多模态 ──────────────────────────────────────────────


class MultimodalRecord(BaseModel):
    """多模态记忆写入请求。

    至少提供 text / image / audio / video 中的一项。
    媒体文件以 Base64 字节流形式传入（而非文件路径），
    由服务端解码存储后嵌入路径到 episode metadata。
    """
    text: Optional[str] = Field(default=None, description="文本描述（可选，用于索引和检索）")
    images: List[str] = Field(default_factory=list, description="Base64 编码的图像字节列表")
    audio: List[str] = Field(default_factory=list, description="Base64 编码的音频字节列表")
    video: List[str] = Field(default_factory=list, description="Base64 编码的视频字节列表")
    source: str = Field(default="user", description="来源标识")
    source_type: SourceType = Field(default=SourceType.DIRECT, description="来源信任分级: direct(用户直述) / tool(工具桥接) / inferred(agent推理/系统提升)")
    namespace: Optional[str] = Field(default=None, description="命名空间")
    visibility: str = Field(default="private", description="可见性")


class MultimodalResponse(BaseModel):
    """多模态记忆写入响应。"""
    episode_id: str = Field(..., description="关联的情节节点 ID")
    visual_node_id: Optional[str] = Field(default=None, description="视觉节点 ID（有图像时）")
    text: Optional[str] = Field(default=None, description="文本内容（裁剪后）")
    media_paths: List[str] = Field(default_factory=list, description="存储的媒体文件路径列表")
    unembedded_paths: List[str] = Field(
        default_factory=list,
        description="已保存但嵌入失败/超时的媒体文件（文件保留，未嵌入索引）",
    )
    transcription: Optional[str] = Field(default=None, description="音频转录文本")
    created_at: float = Field(..., description="创建时间戳")
    error: Optional[str] = Field(default=None, description="拒绝原因（写入被拒绝时非空）")
