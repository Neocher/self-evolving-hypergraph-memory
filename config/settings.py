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
    graphlite: GraphLiteConfig = field(default_factory=GraphLiteConfig)
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
        "graphlite": GraphLiteConfig,
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
