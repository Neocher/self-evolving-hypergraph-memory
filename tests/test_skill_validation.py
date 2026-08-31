"""测试 Phase 3 技能验证门：失败查询 A/B 配对检验 → 技能候选 ACCEPT/REJECT。

覆盖：
  - 技能注入提升 → ACCEPT 且 SKILL.md 写入（sync_from_dream 集成）
  - 技能注入无提升 → REJECT 且目标文件不存在
  - 失败查询集 <12 item → 跳过验证直接写入（兼容）
  - 无 query_router（默认）→ 跳过验证直接写入（兼容）
  - extract_failure_queries_from_file 解析（dict-of-dicts / 缺失 / 损坏 / 空）
  - 与真实 held_out_paired_gate 集成（不 mock gate，探针确认配对分数喂给真实门）

全部走公共入口（validate_skill_candidate / sync_from_dream / extract_...），
不直调被 mock 的内部方法。
"""
import json
import os

import pytest

from core.skill_bridge import sync_from_dream
from core.skill_validation import (
    extract_failure_queries_from_file,
    validate_skill_candidate,
)
from core.validation_gate import Verdict
from retrieval.failure_eval import query_id


def _cfg(**over):
    """构造带全字段的 QueryRouterConfig 替身（可覆盖个别字段）。"""
    class Config:
        weight_fusion_vector = 0.35
        weight_fusion_bm25 = 0.40
        weight_fusion_entity = 0.25
        tau_weight = 0.4
        vector_weight = 0.6
        top_k_l1 = 5
        top_k_fusion = 30
        top_k_keyword = 20
        top_k_vector = 20
        bm25_k1 = 1.5
        bm25_b = 0.75
        mesa_boost = 0.4

    c = Config()
    for k, v in over.items():
        setattr(c, k, v)
    return c


class SkillSensitiveRouter:
    """查询含技能知识 marker（"mermaid"）→ 高质量；否则低质量。

    模仿 ParamSensitiveRouter 手法：10 条 score 0.8（quality 0.92）
    vs 2 条 score 0.3（quality 0.28）。技能正文必含 marker，失败查询必不含。
    """

    def __init__(self, config=None):
        self.config = config or _cfg()
        self.marker = "mermaid"

    def retrieve(self, q, include_archived=False, session_ts=None, level=None, rerank=None):
        if self.marker in q:
            return [{"content": f"c{i}", "score": 0.8} for i in range(10)]
        return [{"content": f"c{i}", "score": 0.3} for i in range(2)]


class FixedRouter:
    """结果与查询无关 → base == cand（无提升 → REJECT）。"""

    def __init__(self, config=None, results=None):
        self.config = config or _cfg()
        self._results = results or [{"content": "r0"}, {"content": "r1"}]

    def retrieve(self, q, include_archived=False, session_ts=None, level=None, rerank=None):
        return self._results


def _failure_queries(n: int, prefix: str = "failed query") -> list[dict]:
    """n 个互不相同的失败查询 meta（Phase 2 持久化格式的 value 形态）。"""
    return [{
        "query": f"{prefix} {i}",
        "num_results": 2,
        "avg_score": 0.2,
        "quality": 0.24,
        "source": "retrieve",
        "first_failed_at": 1.0,
    } for i in range(n)]


def _write_failure_queries(path: str, items: list[dict]) -> str:
    """按 Phase 2 persist_failed_queries 格式（{qid: meta}）写失败查询集。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {query_id(it["query"]): it for it in items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _summary(report: str, patterns=None) -> dict:
    """过质量门的社区摘要：patterns 非空 + report 非 raw JSON + 非元话术。"""
    return {
        "report": report,
        "patterns": patterns or ["synthesize the diagram", "verify each step"],
        "member_count": 5,
    }


def _find_skills(skills_dir: str) -> list[str]:
    """递归找 skills_dir 下所有 SKILL.md（验证 REJECT 场景目标文件不存在）。"""
    out = []
    for dp, _dirs, fns in os.walk(skills_dir):
        for f in fns:
            if f == "SKILL.md":
                out.append(os.path.join(dp, f))
    return sorted(out)


# ═══════════════ 1. extract_failure_queries_from_file 解析 ═══════════════

class TestExtractFailureQueriesFromFile:
    def test_dict_of_dicts_format(self, tmp_path):
        path = _write_failure_queries(
            str(tmp_path / "fq.json"), _failure_queries(3))
        got = extract_failure_queries_from_file(path)
        assert len(got) == 3
        assert {m["query"] for m in got} == {
            "failed query 0", "failed query 1", "failed query 2"}
        assert all("quality" in m and "source" in m for m in got)

    def test_list_format(self, tmp_path):
        path = str(tmp_path / "fq.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"query": "q1"}, {"query": "q2"}], f)
        got = extract_failure_queries_from_file(path)
        assert [m["query"] for m in got] == ["q1", "q2"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert extract_failure_queries_from_file(
            str(tmp_path / "nope.json")) == []

    def test_corrupt_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        assert extract_failure_queries_from_file(path) == []

    def test_empty_payload_returns_empty(self, tmp_path):
        path = str(tmp_path / "empty.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        assert extract_failure_queries_from_file(path) == []


# ═══════════════ 2. validate_skill_candidate 与真实 gate 集成 ═══════════════

class TestValidateSkillCandidate:
    def test_below_12_items_returns_none(self):
        v = validate_skill_candidate(
            "---\nname: x\n---\nbody", _failure_queries(5), SkillSensitiveRouter())
        assert v is None

    def test_real_gate_accept_on_improvement(self):
        """技能注入提升（0.28→0.92）→ 真实 gate ACCEPT（不 mock gate）。"""
        skill_md = ("---\nname: mermaid-diagram-generation-workflow\n"
                    "---\n\nmermaid diagram generation workflow for daily reports\n")
        verdict = validate_skill_candidate(
            skill_md, _failure_queries(15), SkillSensitiveRouter())
        assert isinstance(verdict, Verdict)
        assert verdict.accept is True
        assert verdict.net > 0.5          # 每 item 0.28 → 0.92
        assert verdict.n_improved == 15
        assert verdict.n_regressed == 0   # reg_cap=0：零回归
        assert verdict.ci[0] > 0          # CI 在改善侧排除 0
        assert verdict.reason == "net improvement, CI excludes 0"

    def test_real_gate_reject_on_no_improvement(self):
        """无差异（base == cand）→ 真实 gate REJECT（CI 含 0）。"""
        verdict = validate_skill_candidate(
            "---\nname: x\n---\nbody", _failure_queries(15), FixedRouter())
        assert isinstance(verdict, Verdict)
        assert verdict.accept is False
        assert verdict.net == 0.0
        assert "CI includes 0" in verdict.reason

    def test_gate_receives_paired_scores_from_same_items(self, monkeypatch):
        """探针确认：validate 把同一批 item 的 base/cand 配对分数喂给真实 gate。"""
        captured = {}

        from core.validation_gate import held_out_paired_gate
        real = held_out_paired_gate

        def spy(base, cand, **kw):
            captured["base"] = base
            captured["cand"] = cand
            return real(base, cand, **kw)

        monkeypatch.setattr("core.validation_gate.held_out_paired_gate", spy)
        skill_md = ("---\nname: mermaid-diagram-generation-workflow\n"
                    "---\n\nmermaid diagram generation workflow for daily reports\n")
        verdict = validate_skill_candidate(
            skill_md, _failure_queries(15), SkillSensitiveRouter())
        assert verdict.accept is True           # 真实门判定（spy 内调用 real）
        assert set(captured["base"]) == set(captured["cand"])
        assert len(captured["base"]) == 15      # 同一批 held-out item 配对
        assert all(len(s) == 3 for s in captured["base"].values())  # n_seeds=3
        assert captured["base"] != captured["cand"]

    def test_pluggable_injector(self):
        """可插拔注入策略：自定义 injector 生效（不依赖默认查询增强）。"""
        seen = {}

        def my_injector(router, skill_md):
            seen["skill_md"] = skill_md

            class Wrapped:
                config = router.config

                def retrieve(self, q, *a, **kw):
                    return router.retrieve(q, *a, **kw)

            return Wrapped()

        skill_md = "custom injection body"
        v = validate_skill_candidate(
            skill_md, _failure_queries(15), FixedRouter(),
            injector=my_injector)
        assert seen["skill_md"] == skill_md
        assert v is not None and v.accept is False  # 无提升 → REJECT


# ═══════════════ 3. sync_from_dream 验证门集成 ═══════════════

class TestSyncFromDreamGate:
    def test_accept_writes_skill(self, tmp_path):
        skills_dir = str(tmp_path / "skills")
        fq_path = _write_failure_queries(
            str(tmp_path / "failure_queries.json"), _failure_queries(15))
        router = SkillSensitiveRouter()
        summaries = [_summary(
            "mermaid diagram generation workflow for daily reports")]
        created = sync_from_dream(summaries, skills_dir=skills_dir,
                                  query_router=router,
                                  failure_queries_path=fq_path)
        assert created == ["mermaid-diagram-generation-workflow-daily-reports"]
        target = os.path.join(skills_dir, created[0], "SKILL.md")
        assert os.path.exists(target)
        with open(target, encoding="utf-8") as f:
            text = f.read()
        assert "mermaid" in text and "## 步骤" in text

    def test_reject_does_not_write(self, tmp_path):
        skills_dir = str(tmp_path / "skills")
        fq_path = _write_failure_queries(
            str(tmp_path / "failure_queries.json"), _failure_queries(15))
        router = FixedRouter()
        summaries = [_summary("network anomaly triage checklist for alerts")]
        created = sync_from_dream(summaries, skills_dir=skills_dir,
                                  query_router=router,
                                  failure_queries_path=fq_path)
        assert created == []
        assert _find_skills(skills_dir) == []  # REJECT：目标文件不存在

    def test_below_12_items_skips_validation_writes(self, tmp_path, monkeypatch):
        """失败查询集 <12 item → 跳过验证直接写入（兼容）。"""
        skills_dir = str(tmp_path / "skills")
        fq_path = _write_failure_queries(
            str(tmp_path / "failure_queries.json"), _failure_queries(5))
        router = SkillSensitiveRouter()
        # 统计门若被调用必然 import core.validation_gate.held_out_paired_gate ——
        # 打补丁使其抛错即可证明未走统计门（<12 时 validate 提前返回 None）
        def _boom(*a, **k):
            raise AssertionError("held_out_paired_gate 不应被调用（item < 12）")
        monkeypatch.setattr("core.validation_gate.held_out_paired_gate", _boom)
        summaries = [_summary(
            "mermaid diagram generation workflow for daily reports")]
        created = sync_from_dream(summaries, skills_dir=skills_dir,
                                  query_router=router,
                                  failure_queries_path=fq_path)
        assert created == ["mermaid-diagram-generation-workflow-daily-reports"]
        assert os.path.exists(os.path.join(skills_dir, created[0], "SKILL.md"))

    def test_no_query_router_default_skips_validation_writes(self, tmp_path, monkeypatch):
        """无 query_router（默认）→ 跳过验证直接写入（兼容，行为与改动前一致）。"""
        skills_dir = str(tmp_path / "skills")
        summaries = [_summary(
            "mermaid diagram generation workflow for daily reports")]

        # 若验证被调用必然 import core.skill_validation —— 打补丁抛错证明未走验证
        def _boom(*a, **k):
            raise AssertionError("validate_skill_candidate 不应被调用（无 router）")
        monkeypatch.setattr("core.skill_validation.validate_skill_candidate", _boom)

        created = sync_from_dream(summaries, skills_dir=skills_dir)
        assert created == ["mermaid-diagram-generation-workflow-daily-reports"]
        assert os.path.exists(os.path.join(skills_dir, created[0], "SKILL.md"))

    def test_default_failure_path_missing_skips_validation(self, tmp_path, monkeypatch):
        """有 router 但默认失败查询文件缺失（空集）→ 跳过验证直接写入（兼容）。"""
        skills_dir = str(tmp_path / "skills")
        router = SkillSensitiveRouter()
        # 指向不存在的文件路径 → extract 返回 [] → <12 → 跳过
        summaries = [_summary(
            "mermaid diagram generation workflow for daily reports")]
        created = sync_from_dream(summaries, skills_dir=skills_dir,
                                  query_router=router,
                                  failure_queries_path=str(
                                      tmp_path / "no_such_failure.json"))
        assert created == ["mermaid-diagram-generation-workflow-daily-reports"]
        assert os.path.exists(os.path.join(skills_dir, created[0], "SKILL.md"))
