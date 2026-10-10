"""Schema 级版本化演化（MINDSET 能量 + 迟滞 + 版本链）。

与 ``core/schema_evolver.py`` 的**属性级** ``decide()`` 是不同层：本模块做
**schema 级** 四动作（REINFORCE / SUPERSEDE / SPLIT / NEW）选择，不改动属性级
状态机的任何行为。二者不合并、不替换。

设计依据：MINDSET 论文（arXiv:2610.08586），机制拆解见
``/home/admin/docs/world_model_integration_plan.md`` §1.2（§1.2.1 / §1.2.2 /
§1.2.4）。五项代价与迟滞常量均照论文冻结值，不引入嵌入向量（C 用 Jaccard、
F/I 用计数、D 的余弦用 token 重叠代理）。

纯函数 + 内存态：无 I/O、无第三方依赖；默认关闭（SchemaVersioningConfig.enabled
=False），开关关时既有调用路径零回归。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

# ─── 动作 ──────────────────────────────────────────────────────────

class SchemaAction:
    REINFORCE = "REINFORCE"
    SUPERSEDE = "SUPERSEDE"
    SPLIT = "SPLIT"
    NEW = "NEW"


ALL_ACTIONS = (SchemaAction.REINFORCE, SchemaAction.SUPERSEDE,
               SchemaAction.SPLIT, SchemaAction.NEW)

# ─── 冻结常量（论文拆解值，直接使用）───────────────────────────────

# 能量权重 w = (w_d, w_c, w_h, w_f, w_i) —— 矛盾权重最高，碎片化最低
WEIGHTS = (1.00, 1.25, 0.75, 0.22, 1.00)

LAMBDA_H = 0.35          # 历史损伤系数
LAMBDA_I = 0.50          # 内部不一致系数
ETA = 0.75               # 碎片化 η（系数不是指数）
M_MIN = 2                # 最小 support 数

# 兼容度 K 的权重 (α_d, α_e, α_r, α_c, α_t)
ALPHA_D, ALPHA_E, ALPHA_R, ALPHA_C, ALPHA_T = (0.52, 0.18, 0.12, 0.10, 0.08)
# τ 的权重 (β_c, β_ℓ, β_x, β_r)
BETA_C, BETA_L, BETA_X, BETA_R = (0.35, 0.20, 0.40, 0.25)
# κ 的权重 (γ_t, γ_c, γ_r, γ_d)
GAMMA_T, GAMMA_C, GAMMA_R, GAMMA_D = (0.65, 0.45, 0.20, 0.35)

M_R = 2                  # r 归一化分母
THETA_R = 0.80           # 同槽值 Jaccard 证据阈值（保留供抽取层使用）
O_EMPTY = 0.50

# 可行性门槛
THETA_SUP = 0.35         # SUPERSEDE 要求 p ≥ θ_sup
THETA_SPLIT = 0.12       # SPLIT 要求 κ ≥ θ_split
THETA_NEW = 0.43         # NEW 在 K ≥ θ_new 时被移除
THETA_D = 0.35           # 硬新话题守卫 d 阈值

# 迟滞
M_SWITCH = 0.18
LAMBDA_R = 0.60
RHO = 0.55
THETA_TAU = 0.45
M_STAY = 0.06

# 捷径 / exception 阈值
THETA_P_EASY = 0.15
THETA_KAPPA_EASY = 0.12
THETA_K_EASY = 0.58
THETA_P_ISO = 0.60
THETA_EXC = 0.45

# 多值类别：矛盾只对单值槽计算，这些排除
MULTI_VALUE_SLOTS = frozenset(
    {"preference", "plan", "possession", "experience", "ability", "dislike"}
)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "and", "or",
    "to", "in", "on", "at", "for", "with", "be", "been", "by", "as",
})
_TOKEN_RE = re.compile(r"[0-9a-z]+")


def tokenize(text: str) -> frozenset[str]:
    """归一化去停用词 token 集（小写 + 字母数字词元）。"""
    words = _TOKEN_RE.findall((text or "").lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def token_jaccard(a, b) -> float:
    """归一化 token 集的 Jaccard 相似度 J(a, b) = |a∩b| / |a∪b|（空并 → 0）。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def exception_ratio(exception_count: float) -> float:
    """异常证据计数 M → 归一化 r = min(1, M / m_r)。"""
    return min(1.0, exception_count / M_R) if M_R else 0.0


# ─── 数据结构 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Episode:
    """不可变 episode（provenance）。任何动作下内容不变。"""
    id: str
    raw_text: str
    tokens: frozenset[str] = frozenset()
    slots: tuple[tuple[str, str], ...] = ()   # (槽名, 值 token 串)，有序不可变


@dataclass
class SchemaState:
    """版本化 schema（结构化索引，非摘要）。"""
    id: str
    version: int = 1
    centroid_tokens: set[str] = field(default_factory=set)
    slot_values: dict[str, tuple[frozenset[str], ...]] = field(default_factory=dict)
    support: set[str] = field(default_factory=set)      # support episode id
    exception: set[str] = field(default_factory=set)    # exception episode id
    distortion_mean: float = 0.0                        # D̄_s 运行均值失真
    counter_groups: dict[str, dict[str, int]] = field(default_factory=dict)
    active: bool = True
    predecessor: str | None = None
    successor: str | None = None
    medoid: str | None = None


@dataclass(frozen=True)
class Signals:
    """入站 episode 与 schema 的对齐信号（抽取层提供，本模块只消费）。"""
    m_c: float = 1.0   # context 匹配
    m_t: float = 1.0   # topic 匹配
    o_e: float = 1.0   # entity 特征重叠比
    o_r: float = 1.0   # relation 特征重叠比
    x: float = 0.0     # 显式变更线索
    ell: float = 1.0   # 时间不早于最新 support
    r: float = 0.0     # 异常证据计数 M（≥0；内部 r_norm = min(1, M/m_r)）


@dataclass(frozen=True)
class Potentials:
    p: float
    tau: float
    kappa: float
    k: float
    d: float            # cos 代理 = J(x_tokens, centroid)
    delta_s: float
    d_plus_e: float
    u_s: float
    n_s: int
    distortion_mean: float
    r_norm: float


@dataclass(frozen=True)
class CollectionStats:
    active: int = 0     # A = active schema 数
    thin: int = 0       # T = support < m_min 的 schema 数
    episodes: int = 0   # N = 含 e 的注册 episode 数


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    candidate: str | None = None    # 迟滞带标记的候选重构动作
    as_exception: bool = False      # REINFORCE 且 p ≥ θ_exc → exception
    delta: float | None = None
    m_eff: float | None = None
    feasible: frozenset[str] = frozenset()


# ─── 五项代价 ──────────────────────────────────────────────────────

def compute_contradiction(schema: SchemaState, episode: Episode) -> float:
    """单值槽矛盾 p = max_q (1 − max J(u, v))。多值类别排除；无合格槽 → 0。"""
    best = 0.0
    for q, value_str in episode.slots:
        if q in MULTI_VALUE_SLOTS:
            continue
        vs = schema.slot_values.get(q)
        if not vs:
            continue  # 只对 schema 已存在的单值槽计算
        v_e = tokenize(value_str)
        best = max(best, 1.0 - max(token_jaccard(u, v_e) for u in vs))
    return best


def _internal_inconsistency(schema: SchemaState) -> float:
    """U(s) = (1/|G_s|) Σ_g (1 − max_v n_g(v) / Σ_v n_g(v))；G_s 空 → 0。"""
    vals: list[float] = []
    for g in schema.counter_groups.values():
        if not g:
            continue
        total = sum(g.values())
        if total <= 0:
            continue
        vals.append(1.0 - max(g.values()) / total)
    return sum(vals) / len(vals) if vals else 0.0


def compute_potentials(schema: SchemaState, episode: Episode,
                       signals: Signals) -> Potentials:
    """由 schema + episode + 信号推导全部势函数（纯函数）。"""
    n_s = len(schema.support)
    centroid = set(schema.centroid_tokens)
    ep_tokens = set(episode.tokens)
    union = centroid | ep_tokens
    d = token_jaccard(ep_tokens, centroid)
    if union:
        delta_s = 1.0 - len(centroid) / len(union)
        d_plus_e = 1.0 - len(ep_tokens) / len(union)
    else:
        delta_s = 0.0
        d_plus_e = 0.0
    p = compute_contradiction(schema, episode)
    r_norm = exception_ratio(signals.r)
    tau = (p * min(1.0, BETA_C * signals.m_c + BETA_L * signals.ell)
           + BETA_X * signals.x + BETA_R * r_norm)
    kappa = (max(p, GAMMA_T * signals.m_t)
             * min(1.0, GAMMA_C * (1.0 - signals.m_c)
                   + GAMMA_R * (1.0 - signals.o_r)
                   + GAMMA_D * (1.0 - d)))
    k = clip01(ALPHA_D * d + ALPHA_E * signals.o_e + ALPHA_R * signals.o_r
               + ALPHA_C * signals.m_c + ALPHA_T * signals.m_t)
    return Potentials(p=p, tau=tau, kappa=kappa, k=k, d=d,
                      delta_s=delta_s, d_plus_e=d_plus_e,
                      u_s=_internal_inconsistency(schema),
                      n_s=n_s, distortion_mean=schema.distortion_mean,
                      r_norm=r_norm)


def distortion_cost(action: str, pot: Potentials) -> float:
    """D_a：仅 REINFORCE 计失真，(n_s·(D̄_s+δ_s) + d⁺_e)/(n_s+1)。"""
    if action != SchemaAction.REINFORCE:
        return 0.0
    n = pot.n_s
    return (n * (pot.distortion_mean + pot.delta_s) + pot.d_plus_e) / (n + 1)


def contradiction_cost(action: str, pot: Potentials) -> float:
    """C_a：仅 REINFORCE 计矛盾（= p）。"""
    return pot.p if action == SchemaAction.REINFORCE else 0.0


def historical_damage(action: str, pot: Potentials) -> float:
    """H_a：仅 REINFORCE 计历史损伤 = δ_s + λ_h·p。"""
    if action != SchemaAction.REINFORCE:
        return 0.0
    return pot.delta_s + LAMBDA_H * pot.p


def fragmentation_cost(action: str, *, active: int, thin: int,
                       episodes: int, n_s: int) -> float:
    """F_a：(A + q_a + η·(T + q_a − b_a)) / sqrt(max(1, N+1))。"""
    q_a = 1.0 if action in (SchemaAction.SPLIT, SchemaAction.NEW) else 0.0
    b_a = 1.0 if (action == SchemaAction.REINFORCE
                  and n_s < M_MIN <= n_s + 1) else 0.0
    return (active + q_a + ETA * (thin + q_a - b_a)) / math.sqrt(max(1, episodes + 1))


def inconsistency_cost(action: str, pot: Potentials) -> float:
    """I_a：动作相关。"""
    if action == SchemaAction.REINFORCE:
        return pot.u_s + LAMBDA_I * max(pot.tau, pot.kappa)
    if action == SchemaAction.SUPERSEDE:
        return max(0.0, pot.kappa - pot.tau)
    if action == SchemaAction.SPLIT:
        return max(0.0, pot.tau - pot.kappa)
    if action == SchemaAction.NEW:
        return pot.k
    return 0.0


def energy(action: str, pot: Potentials, stats: CollectionStats) -> float:
    """E(a) = w_d·D_a + w_c·C_a + w_h·H_a + w_f·F_a + w_i·I_a。"""
    w_d, w_c, w_h, w_f, w_i = WEIGHTS
    return (w_d * distortion_cost(action, pot)
            + w_c * contradiction_cost(action, pot)
            + w_h * historical_damage(action, pot)
            + w_f * fragmentation_cost(action, active=stats.active,
                                       thin=stats.thin, episodes=stats.episodes,
                                       n_s=pot.n_s)
            + w_i * inconsistency_cost(action, pot))


# ─── 可行性门槛 + 迟滞决策 ─────────────────────────────────────────

def feasible_actions(pot: Potentials) -> set[str]:
    """比能量前先筛动作：被门槛排除的动作不参与比较。"""
    acts = set(ALL_ACTIONS)
    if pot.p < THETA_SUP:
        acts.discard(SchemaAction.SUPERSEDE)
    if pot.kappa < THETA_SPLIT:
        acts.discard(SchemaAction.SPLIT)
    if pot.k >= THETA_NEW:
        acts.discard(SchemaAction.NEW)
    return acts


def compute_m_eff(pot: Potentials) -> float:
    """有效切换边际 m_eff = m_switch·(1−λ_r·r)·(ρ 若 τ>κ 且 τ>θ_τ 否则 1)。"""
    rho = RHO if (pot.tau > pot.kappa and pot.tau > THETA_TAU) else 1.0
    return M_SWITCH * (1.0 - LAMBDA_R * pot.r_norm) * rho


def decide(schema: SchemaState, episode: Episode, signals: Signals,
           stats: CollectionStats) -> Decision:
    """schema 级四动作决策（能量最小 + 迟滞）。纯函数。"""
    pot = compute_potentials(schema, episode, signals)
    feasible = feasible_actions(pot)
    frozen = frozenset(feasible)

    # 硬新话题守卫：K < θ_new 且 m_t=0 且 d < θ_d → 强制 NEW
    if pot.k < THETA_NEW and signals.m_t == 0 and pot.d < THETA_D:
        return Decision(SchemaAction.NEW, "hard-new-topic", feasible=frozen)

    # 捷径 1：易强化
    if (pot.p < THETA_P_EASY and pot.kappa < THETA_KAPPA_EASY
            and pot.k >= THETA_K_EASY):
        return Decision(SchemaAction.REINFORCE, "easy-reinforce",
                        as_exception=False, feasible=frozen)

    # 捷径 2：孤立强矛盾 → 归 REINFORCE（防孤立矛盾误重写稳定 schema）
    if pot.p >= THETA_P_ISO and signals.r == 0 and signals.x == 0:
        return Decision(SchemaAction.REINFORCE, "isolated-contradiction",
                        as_exception=(pot.p >= THETA_EXC), feasible=frozen)

    e_reinforce = energy(SchemaAction.REINFORCE, pot, stats)
    alts = [a for a in (SchemaAction.SUPERSEDE, SchemaAction.SPLIT,
                        SchemaAction.NEW) if a in feasible]
    if not alts:
        return Decision(SchemaAction.REINFORCE, "no-feasible-alternative",
                        as_exception=(pot.p >= THETA_EXC), feasible=frozen)

    best = min(alts, key=lambda a: energy(a, pot, stats))
    e_best = energy(best, pot, stats)
    delta = e_reinforce - e_best
    m_eff = compute_m_eff(pot)

    if delta > m_eff:
        return Decision(best, "switch", delta=delta, m_eff=m_eff,
                        as_exception=False, feasible=frozen)
    if delta <= M_STAY:
        return Decision(SchemaAction.REINFORCE, "stay", delta=delta, m_eff=m_eff,
                        as_exception=(pot.p >= THETA_EXC), feasible=frozen)
    # 中间带：保持 REINFORCE 但标记候选重构
    return Decision(SchemaAction.REINFORCE, "intermediate", candidate=best,
                    delta=delta, m_eff=m_eff,
                    as_exception=(pot.p >= THETA_EXC), feasible=frozen)


# ─── 版本链 + 回滚（内存态）────────────────────────────────────────

class SchemaRegistry:
    """schema 版本注册表：active/historical + predecessor/successor 链 + 回滚。"""

    def __init__(self) -> None:
        self.schemas: dict[str, SchemaState] = {}
        self._counter = 0

    def add(self, schema: SchemaState) -> SchemaState:
        self.schemas[schema.id] = schema
        return schema

    def get(self, schema_id: str) -> SchemaState:
        return self.schemas[schema_id]

    def _new_id(self, base: str) -> str:
        self._counter += 1
        return f"{base}~v{self._counter}"

    def supersede(self, schema_id: str) -> SchemaState:
        """SUPERSEDE：关旧开新 + 时间链接（predecessor/successor）。"""
        old = self.schemas[schema_id]
        if not old.active:
            raise ValueError(f"schema {schema_id} 非 active，不能 supersede")
        new = SchemaState(
            id=self._new_id(old.id),
            version=old.version + 1,
            centroid_tokens=set(old.centroid_tokens),
            slot_values={k: tuple(v) for k, v in old.slot_values.items()},
            distortion_mean=old.distortion_mean,
            counter_groups={k: dict(v) for k, v in old.counter_groups.items()},
            active=True,
            predecessor=old.id,
            successor=None,
            medoid=old.medoid,
        )
        old.active = False
        old.successor = new.id
        self.schemas[new.id] = new
        return new

    def split(self, schema_id: str) -> SchemaState:
        """SPLIT：保留旧状态 + 加分支（旧 schema 仍 active，不被关闭）。"""
        old = self.schemas[schema_id]
        branch = SchemaState(
            id=self._new_id(old.id),
            version=1,                       # 新分支独立版本线
            centroid_tokens=set(old.centroid_tokens),
            slot_values={k: tuple(v) for k, v in old.slot_values.items()},
            distortion_mean=old.distortion_mean,
            counter_groups={k: dict(v) for k, v in old.counter_groups.items()},
            active=True,
            predecessor=old.id,
            successor=None,
        )
        self.schemas[branch.id] = branch
        return branch

    def lineage(self, schema_id: str) -> list[SchemaState]:
        """沿 predecessor 走到根再反序，返回 root→newest 的有序版本链。"""
        node = self.schemas[schema_id]
        chain: list[SchemaState] = []
        seen: set[str] = set()
        while node is not None and node.id not in seen:
            seen.add(node.id)
            chain.append(node)
            node = self.schemas.get(node.predecessor) if node.predecessor else None
        chain.reverse()
        return chain

    def head(self, schema_id: str) -> SchemaState:
        """沿 successor 走到最新版本（successor=None 的头）。"""
        root = self.lineage(schema_id)[0]
        node = root
        while node.successor is not None and node.successor in self.schemas:
            node = self.schemas[node.successor]
        return node

    def rollback(self, schema_id: str, to_version: int) -> SchemaState:
        """回滚到历史版本：目标版本 active，其余（含更新的）historical，链完整。"""
        lineage = self.lineage(schema_id)
        target = next((s for s in lineage if s.version == to_version), None)
        if target is None:
            raise ValueError(
                f"version {to_version} 不在 {schema_id} 的版本链中: "
                f"{[s.version for s in lineage]}")
        for s in lineage:
            s.active = (s is target)
        return target


# ─── 不可变性指纹（AC6 provenance）─────────────────────────────────

def episode_fingerprint(ep: Episode) -> bytes:
    """episode 的规范字节指纹（任何动作前后应逐字节相等）。"""
    data = {
        "id": ep.id,
        "raw_text": ep.raw_text,
        "tokens": sorted(ep.tokens),
        "slots": [list(s) for s in ep.slots],
    }
    return json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")


# ─── 配置开关（默认 OFF，零回归）───────────────────────────────────

@dataclass
class SchemaVersioningConfig:
    """schema 级版本化演化开关（默认关 → 既有行为逐字节一致）。"""
    enabled: bool = False
