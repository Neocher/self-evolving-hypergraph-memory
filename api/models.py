"""
Pydantic 请求/响应模型
=====================
所有 FastAPI 路由的请求体和响应体数据结构。
Hyperedge 成员数 ≥2 由 field_validator 校验。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── 枚举 ──────────────────────────────────────────────────

class SourceTag(str, Enum):
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"


class HyperedgeType(str, Enum):
    EPISODE = "episode"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"


class TriggerMode(str, Enum):
    IDLE = "idle"
    ACCUMULATED = "accum"
    EXPLICIT = "explicit"


# ─── Episode 请求/响应 ─────────────────────────────────────

class EpisodeCreate(BaseModel):
    """创建情节节点请求（Layer2）。"""
    content: str = Field(..., min_length=1, max_length=100_000,
                         description="情节内容文本")
    source: SourceTag = Field(default=SourceTag.USER, description="来源标签")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")
    force_promote: bool = Field(default=False, description="是否绕过 τ 阈值强制提升")

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
    source: SourceTag = Field(default=SourceTag.USER, description="来源标签")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")

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


class RetrieveResponse(BaseModel):
    """检索响应。"""
    query: str = Field(..., description="原始查询文本")
    strategy_used: str = Field(default="auto", description="实际使用的检索策略")
    results: List[EpisodicResult] = Field(default_factory=list, description="检索结果列表")
    total_found: int = Field(default=0, ge=0, description="检索到的结果总数")
    latency_ms: float = Field(default=0.0, description="检索耗时（毫秒）")
    degraded: bool = Field(default=False, description="是否触发了检索降级")


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


# ─── 溯源 ──────────────────────────────────────────────────

class AuditOperation(BaseModel):
    """单条溯源操作记录。"""
    op_type: str = Field(..., pattern="^(create|update|delete)$", description="操作类型")
    node_id: str = Field(..., description="受影响的节点 ID")
    old_value: Optional[str] = Field(default=None, description="变更前值")
    new_value: Optional[str] = Field(default=None, description="变更后值")
    reason: str = Field(default="explicit", description="操作原因: tau_decay|hebbian_prune|ssm_gate|community_merge|explicit")


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
    kuzu_connected: bool = Field(..., description="Kuzu 数据库连接状态")
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
