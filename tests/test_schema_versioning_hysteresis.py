"""Schema 级版本化演化 + 迟滞 验收测试（MINDSET 能量函数 / 迟滞 / 版本链）。

覆盖 AC1–AC9：五项代价能量 / 四动作可行性门槛 / 迟滞核心行为 /
版本链回滚 / episode 不可变 / 默认关闭零回归 / 属性级 decide() 不回归。
"""
import sys
sys.path.insert(0, "/home/admin/shm")

import pytest

from core import schema_versioning as sv
from core.schema_versioning import (
    SchemaAction as A,
    CollectionStats,
    Episode,
    Potentials,
    SchemaRegistry,
    SchemaState,
    Signals,
    WEIGHTS,
    M_STAY,
)


# ─── 构造辅助 ──────────────────────────────────────────────────────

def _schema(n_support=2, centroid=("alice", "engineer", "acme"),
            slot_name="role", slot_value="engineer"):
    s = SchemaState(id="s1", version=1)
    s.centroid_tokens = set(centroid)
    s.slot_values = {slot_name: (sv.tokenize(slot_value),)}
    s.support = {f"ep{i}" for i in range(n_support)}
    return s


def _episode(tokens=("alice", "engineer", "acme"), slot_value="engineer"):
    return Episode(id="e", raw_text="the alice engineer acme episode",
                   tokens=frozenset(tokens), slots=(("role", slot_value),))


def _stats(active=1, thin=0, episodes=3):
    return CollectionStats(active=active, thin=thin, episodes=episodes)


# ════════════════════════════════════════════════════════════════
# AC1 — 能量函数：5 项代价 + 权重 + 矛盾项主导
# ════════════════════════════════════════════════════════════════

class TestEnergyFunction:
    def test_weights_frozen(self):
        assert WEIGHTS == (1.00, 1.25, 0.75, 0.22, 1.00)

    def test_five_costs_computable(self):
        pot = Potentials(p=0.4, tau=0.3, kappa=0.2, k=0.5, d=0.6,
                         delta_s=0.1, d_plus_e=0.2, u_s=0.05, n_s=3,
                         distortion_mean=0.02, r_norm=0.5)
        # D_a: REINFORCE=(3*(0.02+0.1)+0.2)/4=0.14；非 REINFORCE=0
        assert sv.distortion_cost(A.REINFORCE, pot) == pytest.approx(0.14)
        assert sv.distortion_cost(A.SUPERSEDE, pot) == 0.0
        # C_a: REINFORCE=p；非 REINFORCE=0
        assert sv.contradiction_cost(A.REINFORCE, pot) == pytest.approx(0.4)
        assert sv.contradiction_cost(A.SPLIT, pot) == 0.0
        # H_a: δ_s + λ_h·p = 0.1 + 0.35*0.4
        assert sv.historical_damage(A.REINFORCE, pot) == pytest.approx(0.24)
        assert sv.historical_damage(A.NEW, pot) == 0.0
        # F_a: (A + q + η(T + q − b)) / sqrt(N+1)
        # REINFORCE(q=0,b=0): (2+0+0.75*(1+0-0))/3 = 2.75/3
        assert sv.fragmentation_cost(A.REINFORCE, active=2, thin=1,
                                     episodes=8, n_s=3) == pytest.approx(2.75 / 3)
        # SPLIT(q=1,b=0): (2+1+0.75*(1+1-0))/3 = (3+1.5)/3 = 1.5
        assert sv.fragmentation_cost(A.SPLIT, active=2, thin=1,
                                     episodes=8, n_s=3) == pytest.approx(1.5)
        # I_a：REINFORCE=u_s+λ_i·max(τ,κ)；SUPERSEDE=max(0,κ−τ)；SPLIT=max(0,τ−κ)；NEW=k
        assert sv.inconsistency_cost(A.REINFORCE, pot) == pytest.approx(0.05 + 0.5 * 0.3)
        assert sv.inconsistency_cost(A.SUPERSEDE, pot) == 0.0
        assert sv.inconsistency_cost(A.SPLIT, pot) == pytest.approx(0.1)
        assert sv.inconsistency_cost(A.NEW, pot) == pytest.approx(0.5)

    def test_energy_linear_combination(self):
        pot = Potentials(p=0.4, tau=0.3, kappa=0.2, k=0.5, d=0.6,
                         delta_s=0.1, d_plus_e=0.2, u_s=0.05, n_s=3,
                         distortion_mean=0.02, r_norm=0.5)
        stats = CollectionStats(active=2, thin=1, episodes=8)
        expect = (1.00 * 0.14 + 1.25 * 0.4 + 0.75 * 0.24
                  + 0.22 * (2.75 / 3) + 1.00 * (0.05 + 0.5 * 0.3))
        assert sv.energy(A.REINFORCE, pot, stats) == pytest.approx(expect)

    def test_contradiction_dominates_energy(self):
        """C 高时 REINFORCE 能量显著升高（矛盾项主导）。"""
        stats = _stats(episodes=3)
        matching = _episode(slot_value="engineer")
        contradicting = _episode(slot_value="astronaut")
        sig = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=1.0, x=0.0, ell=1.0, r=0.0)
        pot_low = sv.compute_potentials(_schema(), matching, sig)
        pot_high = sv.compute_potentials(_schema(), contradicting, sig)
        assert pot_high.p > pot_low.p
        # 矛盾代价项 = p；在 energy() 中按 w_c=1.25 计入（矛盾权重最高）
        assert (sv.contradiction_cost(A.REINFORCE, pot_high)
                - sv.contradiction_cost(A.REINFORCE, pot_low)) == pytest.approx(
                    pot_high.p - pot_low.p)
        e_low = sv.energy(A.REINFORCE, pot_low, stats)
        e_high = sv.energy(A.REINFORCE, pot_high, stats)
        assert e_high - e_low > 1.5   # 显著升高，矛盾主导


# ════════════════════════════════════════════════════════════════
# AC2 — 四动作 + 可行性门槛：被排除动作不参与比较
# ════════════════════════════════════════════════════════════════

class TestFeasibilityGates:
    def test_gates_by_potential(self):
        # p 低 → SUPERSEDE 排除；κ 低 → SPLIT 排除；K 高 → NEW 排除
        pot = Potentials(p=0.1, tau=0.0, kappa=0.05, k=0.9, d=0.5,
                         delta_s=0.0, d_plus_e=0.0, u_s=0.0, n_s=2,
                         distortion_mean=0.0, r_norm=0.0)
        assert sv.feasible_actions(pot) == {A.REINFORCE}

    def test_all_four_feasible(self):
        pot = Potentials(p=0.5, tau=0.3, kappa=0.3, k=0.2, d=0.5,
                         delta_s=0.0, d_plus_e=0.0, u_s=0.0, n_s=2,
                         distortion_mean=0.0, r_norm=0.0)
        assert sv.feasible_actions(pot) == {A.REINFORCE, A.SUPERSEDE, A.SPLIT, A.NEW}

    def test_gated_out_actions_not_compared(self):
        """SUPERSEDE/SPLIT 被门槛排除后，决策只与可行替代比较。"""
        schema = _schema(n_support=2)  # slot role=engineer
        # 单值槽矛盾 p=0.2（<0.35 → SUPERSEDE 排除）；构造 κ≥0.12 使 SPLIT 可行
        schema.slot_values = {"role": (sv.tokenize("alpha beta gamma delta"),)}
        ep = _episode(tokens=("alice", "engineer", "acme"),
                      slot_value="alpha beta gamma delta epsilon")
        sig = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=0.0, x=0.0, ell=1.0, r=0.0)
        pot = sv.compute_potentials(schema, ep, sig)
        feasible = sv.feasible_actions(pot)
        assert A.SUPERSEDE not in feasible        # p=0.2 < 0.35
        assert A.SPLIT in feasible                # κ ≥ 0.12
        assert A.NEW not in feasible              # K ≥ 0.43
        decision = sv.decide(schema, ep, sig, _stats())
        # 迟滞带内（见 AC3c），保持 REINFORCE 且候选必须是 SPLIT（唯一可行替代）
        assert decision.action == A.REINFORCE
        assert decision.candidate == A.SPLIT
        assert A.SUPERSEDE not in decision.feasible
        assert A.NEW not in decision.feasible

    def test_hard_new_topic_guard_forces_new(self):
        """K < θ_new 且 m_t=0 且 d < θ_d → 强制 NEW。"""
        schema = _schema(n_support=2)
        ep = _episode(tokens=("x", "y", "z"), slot_value="engineer")
        sig = Signals(m_c=0.0, m_t=0.0, o_e=0.0, o_r=0.0, x=0.0, ell=0.0, r=0.0)
        decision = sv.decide(schema, ep, sig, _stats())
        assert decision.action == A.NEW
        assert decision.reason == "hard-new-topic"


# ════════════════════════════════════════════════════════════════
# AC3 — 迟滞（核心验收点）
# ════════════════════════════════════════════════════════════════

class TestHysteresis:
    def test_isolated_strong_contradiction_reinforces(self):
        """(a) p≥0.60 但 r=0、x=0 → 归 REINFORCE（不切换）。"""
        schema = _schema(n_support=3, slot_value="engineer")
        ep = _episode(tokens=("alice", "engineer", "acme"), slot_value="astronaut")
        sig = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=1.0, x=0.0, ell=1.0, r=0.0)
        pot = sv.compute_potentials(schema, ep, sig)
        assert pot.p >= 0.60          # 强矛盾前置成立
        decision = sv.decide(schema, ep, sig, _stats(episodes=4))
        assert decision.action == A.REINFORCE
        assert decision.reason == "isolated-contradiction"
        assert decision.as_exception is True   # p ≥ 0.45 → exception

    def test_sustained_contradiction_switches(self):
        """(b) r ≥ m_r(=2) 且有足够 Δ → 切换到替代动作。"""
        schema = _schema(n_support=3, slot_value="engineer")
        ep = _episode(tokens=("alice", "engineer", "acme"), slot_value="astronaut")
        sig = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=1.0, x=0.0, ell=1.0, r=2.0)
        pot = sv.compute_potentials(schema, ep, sig)
        assert pot.r_norm == pytest.approx(1.0)   # r=2 → 归一化=1
        decision = sv.decide(schema, ep, sig, _stats(episodes=4))
        assert decision.action != A.REINFORCE
        assert decision.action == A.SUPERSEDE
        assert decision.reason == "switch"
        assert decision.delta > decision.m_eff

    def test_intermediate_band_stays_with_candidate(self):
        """(c) m_stay < Δ ≤ m_eff → 保持 REINFORCE 并标记候选。"""
        schema = _schema(n_support=2)
        schema.slot_values = {"role": (sv.tokenize("alpha beta gamma delta"),)}
        ep = _episode(tokens=("alice", "engineer", "acme"),
                      slot_value="alpha beta gamma delta epsilon")
        sig = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=0.0, x=0.0, ell=1.0, r=0.0)
        decision = sv.decide(schema, ep, sig, _stats(episodes=3))
        # 前置成立：确实落在迟滞带内
        assert decision.delta is not None and decision.m_eff is not None
        assert M_STAY < decision.delta <= decision.m_eff
        assert decision.action == A.REINFORCE
        assert decision.reason == "intermediate"
        assert decision.candidate == A.SPLIT

    def test_r_from_zero_to_two_flips_decision(self):
        """(d) r 0→2：决策从 REINFORCE 变为切换（证明迟滞非恒真）。"""
        schema = _schema(n_support=3, slot_value="engineer")
        ep = _episode(tokens=("alice", "engineer", "acme"), slot_value="astronaut")
        stats = _stats(episodes=4)
        sig_r0 = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=1.0, x=0.0, ell=1.0, r=0.0)
        sig_r2 = Signals(m_c=1.0, m_t=1.0, o_e=1.0, o_r=1.0, x=0.0, ell=1.0, r=2.0)
        d0 = sv.decide(schema, ep, sig_r0, stats)
        d2 = sv.decide(schema, ep, sig_r2, stats)
        assert d0.action == A.REINFORCE
        assert d2.action != A.REINFORCE


# ════════════════════════════════════════════════════════════════
# AC4 / AC5 — 版本链 + 回滚
# ════════════════════════════════════════════════════════════════

class TestVersionChain:
    def test_supersede_links_and_states(self):
        reg = SchemaRegistry()
        s0 = reg.add(SchemaState(id="s0", version=1, active=True))
        s1 = reg.supersede("s0")
        assert s1.predecessor == "s0"
        assert s0.successor == s1.id
        assert s0.active is False        # historical
        assert s1.active is True         # active
        assert s1.version == s0.version + 1

    def test_split_preserves_old_state(self):
        reg = SchemaRegistry()
        s0 = reg.add(SchemaState(id="s0", version=1, active=True))
        branch = reg.split("s0")
        assert branch.predecessor == "s0"
        assert s0.active is True         # 旧状态保留，不被关闭
        assert branch.active is True

    def test_rollback_restores_and_chain_intact(self):
        reg = SchemaRegistry()
        s0 = reg.add(SchemaState(id="s0", version=1))
        s1 = reg.supersede("s0")
        s2 = reg.supersede(s1.id)
        assert [s.version for s in reg.lineage(s2.id)] == [1, 2, 3]

        target = reg.rollback(s2.id, to_version=1)
        assert target.id == "s0"
        assert s0.active is True
        assert s1.active is False
        assert s2.active is False
        # 链完整不断裂：successor/predecessor 双向一致，head 仍为最新版本
        assert s0.successor == s1.id and s1.predecessor == "s0"
        assert s1.successor == s2.id and s2.predecessor == s1.id
        assert s2.successor is None
        assert reg.head(s2.id).id == s2.id
        assert [s.id for s in reg.lineage(s2.id)] == [s0.id, s1.id, s2.id]

    def test_rollback_unknown_version_raises(self):
        reg = SchemaRegistry()
        reg.add(SchemaState(id="s0", version=1))
        with pytest.raises(ValueError):
            reg.rollback("s0", to_version=99)


# ════════════════════════════════════════════════════════════════
# AC6 — 原始 episode 不可变（provenance）
# ════════════════════════════════════════════════════════════════

class TestEpisodeImmutability:
    def test_episode_bytes_unchanged_across_actions(self):
        ep = _episode(slot_value="astronaut")
        before = sv.episode_fingerprint(ep)

        reg = SchemaRegistry()
        reg.add(SchemaState(id="s0", version=1))
        s1 = reg.supersede("s0")
        reg.split("s0")
        reg.rollback(s1.id, to_version=1)

        after = sv.episode_fingerprint(ep)
        assert before == after
        assert ep.raw_text == "the alice engineer acme episode"
        assert ep.slots == (("role", "astronaut"),)


# ════════════════════════════════════════════════════════════════
# AC7 — 默认关闭 + 零回归（属性级 decide() 行为不变）
# ════════════════════════════════════════════════════════════════

class TestDefaultOffZeroRegression:
    def test_config_default_off(self):
        assert sv.SchemaVersioningConfig().enabled is False
        from config.settings import Settings
        assert Settings().schema_versioning.enabled is False

    def test_attribute_decide_unchanged(self):
        """core.schema_evolver.decide() 行为不得改变（回归守卫）。"""
        from core.schema_evolver import decide, Action, AttrStat
        assert decide(AttrStat("t", "CEO", "x", {"a": 1}, [], 0.38), None) == Action.EMERGE
        assert decide(AttrStat("t", "CEO", "x", {"a": 5, "b": 5}, [], 1.0), None) == Action.SOLIDIFY
        solid = {"value": "CEO", "value_blake3": "x", "conf": 0.7, "active": True}
        assert decide(AttrStat("t", "CEO", "x", {"a": 5, "b": 5}, [], 1.0), solid) == Action.STRENGTHEN
        assert decide(AttrStat("t", "CTO", "y", {"a": 5, "b": 5, "c": 5}, [], 0.90), solid) == Action.CORRECT
        assert decide(AttrStat("t", "CTO", "y", {"a": 2, "b": 2}, [], 0.74), solid) == Action.IGNORE
