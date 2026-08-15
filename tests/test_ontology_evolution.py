"""
OntologyEvolution (v5.38.0 Schema 自演化) 测试
================================================
覆盖：
· merge 3 例 — new_type 注册 / 已存在合并去重 / LLM 失败 skip
· 加载 2 例 — JSON 存在合并且不覆盖原生 / 缺失损坏 → 空 dict
· 集成 1 例 — mock LLM new_type → 注册 → 新 validator _classify_ontology_type 命中
· CC 修正 3 例 — 跨类型 key 冲突 → skip / ≥2 key 全泛词 → skip / max 1 新类型/轮
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from core.ontology_evolution import (
    classify_with_extended,
    evolve_once,
    load_extended,
    merged_types,
)
from core.ontology_validator import ONTOLOGY_TYPES, OntologyValidator


def run(coro):
    return asyncio.run(coro)


def _llm(response: str) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.chat = AsyncMock(return_value=response)
    return client


def _summaries() -> list[dict]:
    return [
        {
            "topics": ["quantum entanglement", "nonlocality"],
            "report": "Community about quantum entanglement experiments and nonlocality.",
        },
    ]


# ─── 加载 2 例 ───────────────────────────────────────────────


def test_load_extended_missing_returns_empty(tmp_path):
    assert load_extended(str(tmp_path / "nope.json")) == {}


def test_load_extended_corrupt_returns_empty(tmp_path):
    p = tmp_path / "ontology_extended.json"
    p.write_text("{not valid json!!", encoding="utf-8")
    assert load_extended(str(p)) == {}


def test_merged_types_native_priority(tmp_path):
    """JSON 存在合并且不覆盖原生：extended 同名键被原生定义覆盖。"""
    p = tmp_path / "ontology_extended.json"
    p.write_text(json.dumps({
        "event_date": {"description": "HACKED", "conflict_keys": ["x"]},
        "quantum_result": {"description": "量子实验", "conflict_keys": ["量子"]},
    }, ensure_ascii=False), encoding="utf-8")
    merged = merged_types(load_extended(str(p)))
    assert merged["event_date"]["description"] == ONTOLOGY_TYPES["event_date"]["description"]
    assert merged["quantum_result"]["description"] == "量子实验"


def test_validator_polluted_extended_not_override_native():
    """污染 extended（含原生同名键）→ 原生优先不被覆盖，全局 ONTOLOGY_TYPES 不被污染。"""
    native_desc = ONTOLOGY_TYPES["event_date"]["description"]
    native_keys = list(ONTOLOGY_TYPES["event_date"]["conflict_keys"])
    validator = OntologyValidator(extended_types={
        "event_date": {"description": "HACKED", "conflict_keys": ["hacked"]},
        "quantum_result": {"description": "量子实验", "conflict_keys": ["量子"]},
    })
    merged = validator._merged_ontology_types()
    # 原生优先：污染 extended 无法覆盖 event_date
    assert merged["event_date"]["description"] == native_desc
    assert merged["event_date"]["conflict_keys"] == native_keys
    # 分类走原生定义（含原生 conflict_keys 才命中）
    assert validator._classify_ontology_type("会议于2024年5月1日举行", []) == "event_date"
    # 全局未被污染
    assert ONTOLOGY_TYPES["event_date"]["description"] == native_desc
    assert ONTOLOGY_TYPES["event_date"]["conflict_keys"] == native_keys


# ─── merge 3 例 ─────────────────────────────────────────────


def test_evolve_new_type_registers(tmp_path):
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子实验发现",
        "conflict_keys": ["quantum", "entanglement", "nonlocality"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"
    assert result["type"] == "quantum_result"
    extended = load_extended(str(p))
    assert "quantum_result" in extended
    assert extended["quantum_result"]["conflict_keys"] == ["quantum", "entanglement", "nonlocality"]


def test_evolve_merge_existing_dedup(tmp_path):
    p = tmp_path / "ontology_extended.json"
    p.write_text(json.dumps({
        "quantum_result": {
            "description": "量子实验发现",
            "conflict_keys": ["quantum", "entanglement"],
        },
    }, ensure_ascii=False), encoding="utf-8")
    client = _llm(json.dumps({
        "action": "merge_existing",
        "type": "quantum_result",
        "conflict_keys": ["quantum", "纠缠"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "merge_existing"
    extended = load_extended(str(p))
    entry = extended["quantum_result"]
    # 去重合并，description 不被覆盖
    assert entry["conflict_keys"] == ["quantum", "entanglement", "纠缠"]
    assert entry["description"] == "量子实验发现"


def test_evolve_merge_native_type_skipped(tmp_path):
    """merge_existing 指向原生类型 → skip（merge_target_native），原生不被改、不落盘。"""
    p = tmp_path / "ontology_extended.json"
    native_keys = list(ONTOLOGY_TYPES["event_date"]["conflict_keys"])
    client = _llm(json.dumps({
        "action": "merge_existing",
        "type": "event_date",
        "conflict_keys": ["hacked_key"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert result["reason"] == "merge_target_native"
    assert not p.exists()  # 未落盘
    # 全局 ONTOLOGY_TYPES 未被原地污染
    assert ONTOLOGY_TYPES["event_date"]["conflict_keys"] == native_keys


def test_evolve_llm_failure_skip(tmp_path):
    p = tmp_path / "ontology_extended.json"
    client = _llm(None)  # chat 返回 None → 视为失败
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert result["reason"] == "llm_failed"
    assert not p.exists()


def test_evolve_no_llm_client_skip(tmp_path):
    result = run(evolve_once(_summaries(), None, str(tmp_path / "x.json")))
    assert result["action"] == "skip"
    assert result["reason"] == "no_llm_client"


def test_evolve_persist_failure_returns_skip(tmp_path, monkeypatch):
    """_atomic_write 失败 → 不声称成功，返回 skip（persist_failed）且不落盘。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子实验发现",
        "conflict_keys": ["quantum", "entanglement"],
    }))
    monkeypatch.setattr("core.ontology_evolution._atomic_write", lambda *a, **k: False)
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert result["reason"] == "persist_failed"
    assert not p.exists()


# ─── CC 修正 3 例 ────────────────────────────────────────────


def test_evolve_cross_type_key_clash_skip(tmp_path):
    """新类型 keys 与已有类型（person_birth）重叠 → skip（first-match-wins）。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "person_record",
        "description": "人的记录",
        "conflict_keys": ["person", "birth", "量子"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert "person_birth" in result["reason"]
    assert not p.exists()


def test_evolve_all_generic_keys_skip(tmp_path):
    """≥2 keys 但全泛词 → skip（not_enough_specific_keys）。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "generic_thing",
        "description": "泛化类型",
        "conflict_keys": ["data", "信息", "内容"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "skip"
    assert result["reason"] == "not_enough_specific_keys"
    assert not p.exists()


def test_evolve_max_one_new_type_per_round(tmp_path):
    """LLM 返回 2 个 new_types → 只注册第一个通过守卫的（max 1/轮）。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "new_types": [
            {"name": "type_a", "description": "A", "conflict_keys": ["alpha", "beta"]},
            {"name": "type_b", "description": "B", "conflict_keys": ["gamma", "delta"]},
        ],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"
    assert result["type"] == "type_a"
    extended = load_extended(str(p))
    assert "type_a" in extended
    assert "type_b" not in extended


def test_evolve_try_next_proposal_when_first_guard_fails(tmp_path):
    """首个提案守卫失败 → 继续尝试下一个通过守卫的（P2 修正）。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "new_types": [
            # 全泛词 → not_enough_specific_keys 守卫拒绝
            {"name": "bad_type", "description": "bad", "conflict_keys": ["data", "信息"]},
            {"name": "good_type", "description": "good", "conflict_keys": ["alpha", "beta"]},
        ],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"
    assert result["type"] == "good_type"
    extended = load_extended(str(p))
    assert "good_type" in extended
    assert "bad_type" not in extended


# ─── 集成 1 例 ───────────────────────────────────────────────


def test_integration_register_then_classify_hits(tmp_path):
    """mock LLM new_type → 注册 → 新 validator._classify_ontology_type 命中。"""
    p = tmp_path / "ontology_extended.json"
    client = _llm(json.dumps({
        "action": "new_type",
        "type": "quantum_result",
        "description": "量子纠缠实验发现",
        "conflict_keys": ["量子", "纠缠"],
    }))
    result = run(evolve_once(_summaries(), client, str(p)))
    assert result["action"] == "new_type"

    validator = OntologyValidator(extended_types=load_extended(str(p)))
    assert validator._classify_ontology_type("量子纠缠态非局域性", []) == "quantum_result"
    # 原生类型不受影响（原生优先）
    assert validator._classify_ontology_type("他出生于1985年", []) == "person_birth"
    # 独立函数入口一致
    assert classify_with_extended("量子纠缠态", [], load_extended(str(p))) == "quantum_result"
