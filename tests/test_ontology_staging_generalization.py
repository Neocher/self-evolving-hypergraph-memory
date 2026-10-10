"""本体暂存台账 —— 候选聚合泛化验收测试（中文场景实用覆盖）

上一轮（cf89332）实现了「候选键语义模板聚合」，但实例槽位识别只覆盖
「英文大写名 + 中文报告动词 + 第N」三种形态。本轮把槽位识别泛化到
中文生产场景（SHM 语料以中文为主），同时绝不放松负样本保护。

覆盖 AC1–AC6 的可判定断言：
- AC1 中文姓名聚合（2/3 字名 + 复姓）
- AC2 英文报告动词（said/reported/... 等常见词）
- AC3 其他编号式（Report N / No. N）
- AC4 句首小写拉丁名
- AC5 负样本保护（主语/宾语即主题、同句式不同主题、谓词不同、不抹空）
- AC6 上一轮成果不回归（原 12→1 模板 / 基础去重语义）
"""

from __future__ import annotations

import pytest

from core.ontology_staging import (
    _mask_instance_slots,
    candidate_key,
    get_staging,
    reset_staging,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """每个用例独立台账文件 + 清洁单例 + 开关默认开。"""
    monkeypatch.delenv("SHM_ONTOLOGY_STAGING", raising=False)
    reset_staging(str(tmp_path / "staging.json"))
    yield
    reset_staging(str(tmp_path / "staging.json"))


def _same(a: str, b: str) -> None:
    assert candidate_key(a) == candidate_key(b), f"应聚合却分键:\n  {a!r}\n  {b!r}"


def _diff(a: str, b: str) -> None:
    assert candidate_key(a) != candidate_key(b), f"应分键却聚合:\n  {a!r}\n  {b!r}"


# ─────────────────────── AC1 中文姓名聚合 ───────────────────────

def test_ac1_chinese_name_with_ordinal_aggregates():
    _same("张三认为X是第0条独特观测", "李四认为X是第1条独特观测")


def test_ac1_chinese_name_identical_content_aggregates():
    _same("张三认为X需要维护", "李四认为X需要维护")


def test_ac1_chinese_three_char_name_aggregates():
    _same("王小明指出X需要维护", "陈晓华指出X需要维护")


def test_ac1_compound_surname_aggregates():
    _same("欧阳修表示X需要维护", "司马光表示X需要维护")


# ─────────────────────── AC2 英文报告动词 ───────────────────────

def test_ac2_english_reporting_verb_aggregates():
    _same("Alice said X needs maintenance", "Bob said X needs maintenance")


def test_ac2_english_reporting_verbs_cover_common_list():
    for verb in ("said", "reported", "stated", "claimed", "noted", "mentioned",
                 "argued", "proposed", "wrote"):
        _same(f"Alice {verb} X needs check", f"Bob {verb} X needs check")


# ─────────────────────── AC3 其他编号式 ───────────────────────

def test_ac3_report_labeled_number_aggregates():
    _same("Report 12: X needs quarterly check", "Report 45: X needs quarterly check")


def test_ac3_no_labeled_number_aggregates():
    _same("No. 3 X needs check", "No. 7 X needs check")


# ─────────────────────── AC4 句首小写拉丁名 ───────────────────────

def test_ac4_lowercase_leading_name_aggregates():
    _same("alice 认为X需要维护", "bob 认为X需要维护")


# ─────────────────────── AC5 负样本保护 ───────────────────────

def test_ac5_negative_subject_is_topic_not_masked():
    """(a) 主语即主题（Python/Java 是主语言）不得当姓名抹掉。"""
    _diff("Python 是数据处理的主语言", "Java 是数据处理的主语言")


def test_ac5_negative_object_language_not_masked_as_name():
    """(b) 宾语语言名不得抹掉；不同语义（是主语言 vs 使用）不得合并。"""
    assert _mask_instance_slots("我们使用 Python 和 Java 两种语言") == \
        "我们使用 Python 和 Java 两种语言"
    _diff("Python 是数据处理的主语言", "我们使用 Python 和 Java 两种语言")


def test_ac5_negative_same_syntax_different_topic():
    """(c) 同句式不同主题（柴发机组/风冷机组）→ 不同键。"""
    _diff("张三认为柴发机组需要每季度保养", "张三认为风冷机组需要每季度保养")


def test_ac5_negative_same_speaker_different_predicate():
    """(d) 谓词不同（冷却液/机油）→ 不同键。"""
    _diff("张三指出X的冷却液需要更换", "张三指出X的机油需要更换")


def test_ac5_negative_no_empty_residual():
    """(e) 空键与真实文本不同；任意非空输入抹除后残余不得为空。"""
    assert candidate_key("") != candidate_key("张三认为X需要维护")
    for text in ("张三认为", "Alice said", "第3条", "Report 12", "No. 3",
                 "张三", "任意真实文本", "X"):
        assert _mask_instance_slots(text).strip() != "", f"残余为空: {text!r}"


# ─────────────────────── AC6 上一轮成果不回归 ───────────────────────

_ORIGINAL_TEMPLATE = (
    "{name} 认为量子浮点校准仪需要每季度做一次非线性漂移补偿，这是第{n}条独特观测"
)
_ORIGINAL_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
]


def test_ac6_original_template_still_aggregates_12_to_1():
    """原 AC1 样本（英文名 + 第N条）仍 12→1。"""
    s = get_staging()
    for i, name in enumerate(_ORIGINAL_NAMES):
        s.record(_ORIGINAL_TEMPLATE.format(name=name, n=i))
    items = s.list_candidates()
    assert len(items) == 1
    assert items[0]["count"] == 12


def test_ac6_identical_text_same_key():
    """基础去重语义不坏：完全相同的句子仍返回相同键。"""
    assert candidate_key("任意固定文本_同一语义_abc123") == \
        candidate_key("任意固定文本_同一语义_abc123")
