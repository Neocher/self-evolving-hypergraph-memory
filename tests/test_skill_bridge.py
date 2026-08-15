"""
Skill-Bridge 记忆→Skill 一体化测试
=================================
质量门三合一 + 动作短语命名 + 判重 + frontmatter + 集成（7 社区 4 坏 3 好 → 3 skill）。
"""
from __future__ import annotations

import os

import yaml

from core.skill_bridge import (
    _scan_existing_skills,
    extract_reusable_patterns,
    generate_skill_md,
    generate_skill_name,
    should_create_skill,
    sync_from_dream,
)


def _comm(report: str, patterns: list, member_count: int = 100) -> dict:
    return {"report": report, "patterns": patterns, "member_count": member_count}


def test_quality_gate():
    """质量门：patterns 空 / raw JSON / 元话术 跳过，全通过保留。"""
    assert extract_reusable_patterns([_comm("valid report", [])]) == []
    assert extract_reusable_patterns([_comm('{"a": 1}', ["p"])]) == []
    assert extract_reusable_patterns([_comm('["a","b"]', ["p"])]) == []  # raw JSON 数组
    assert extract_reusable_patterns([_comm("```json\n{\"a\": 1}\n```", ["p"])]) == []  # ```json 围栏
    assert extract_reusable_patterns([_comm("We need to analyze data", ["p"])]) == []
    assert extract_reusable_patterns([_comm("The task is to merge", ["p"])]) == []
    assert extract_reusable_patterns([_comm("我们需要分析 feishu 报表", ["p"])]) == []  # 中文元话术
    ok = extract_reusable_patterns([_comm("Valid reusable pattern text", ["p1", "p2"])])
    assert len(ok) == 1 and ok[0]["patterns"] == ["p1", "p2"]


def test_generate_skill_name():
    assert generate_skill_name("Feishu report delivery", ["p"]) == "feishu-report-delivery"
    assert generate_skill_name("1. Feishu report delivery", ["p"]) == "feishu-report-delivery"  # 序号剥离
    assert generate_skill_name("- Feishu report delivery", ["p"]) == "feishu-report-delivery"  # 列表符号剥离
    assert generate_skill_name("1. this feishu of cluster delivery", ["p"]) == "feishu-delivery"  # 纯数字/停用词过滤
    assert generate_skill_name("飞书日报自动化", ["send feishu daily report"]) == "send-feishu-daily-report"  # 纯中文回退 patterns
    assert generate_skill_name("飞书日报自动化", ["分步下发"]) == ""  # 纯中文无拉丁 → 空串
    assert generate_skill_name("", []) == ""  # 提取失败 → 跳过
    assert generate_skill_name("   ", ["p"]) == ""  # 单词短语不足 → 跳过


def test_should_create_skill():
    existing = [{"name": "feishu-report-delivery", "description": "deliver feishu reports daily"}]
    assert not should_create_skill("feishu-report-delivery", existing)  # 同名
    assert should_create_skill("feishu-delivery", existing)  # 单个词重叠（feishu）不阻断
    assert not should_create_skill("feishu-reports-daily", existing)  # ≥2 非泛词重叠 → 判重
    assert should_create_skill("meeting-notes-summary", [])  # 全新


def test_should_create_skill_no_false_kill():
    """真实 skills 目录误杀回归：单个 token 重叠不再阻断（旧逻辑五个全部误拒）。"""
    existing = [
        {"name": "feishu-report-delivery", "description": "deliver feishu reports daily"},
        {"name": "weekly-report-automation", "description": "automate weekly reports with python"},
        {"name": "python-notes", "description": "write and run python scripts"},
        {"name": "system-monitoring", "description": "monitor system uptime"},
        {"name": "meeting-summary", "description": "summary of daily team meetings"},
    ]
    for name in ("feishu-auth-debugging", "report-automation", "python-pitfalls-v2",
                 "system-health-check", "daily-meeting-notes"):
        assert should_create_skill(name, existing)


def test_scan_existing_skills_yaml_block(tmp_path):
    """description: | YAML 块应被解析为实际文本（而非 '|'）；.archive 目录被排除。"""
    (tmp_path / "feishu-report-delivery").mkdir()
    (tmp_path / "feishu-report-delivery" / "SKILL.md").write_text(
        "---\nname: feishu-report-delivery\ndescription: |\n  deliver feishu reports daily\n"
        "trigger: x\n---\n", encoding="utf-8"
    )
    arch_dir = tmp_path / ".archive" / "python-pitfalls"
    arch_dir.mkdir(parents=True)
    (arch_dir / "SKILL.md").write_text(
        "---\nname: python-pitfalls\ndescription: python pitfalls notes\n---\n", encoding="utf-8"
    )
    existing = _scan_existing_skills(str(tmp_path))
    # 块解析生效 → description 非 '|' → ≥2 非泛词重叠判重
    assert not should_create_skill("feishu-reports-daily", existing)
    # .archive 被排除 → python/pitfalls 不参与判重 → 单个 token 重叠不阻断
    assert should_create_skill("python-pitfalls-v2", existing)


def test_generate_skill_md_frontmatter():
    md = generate_skill_md({
        "name": "feishu-report-delivery",
        "report": "Feishu report delivery",
        "patterns": ["step one", "step two"],
        "member_count": 93,
    })
    data = yaml.safe_load(md.split("---", 2)[1])
    assert set(data) >= {"name", "description", "trigger"}
    assert data["name"] == "feishu-report-delivery"
    assert "## 步骤" in md and "step one" in md and "## 来源" in md


def test_sync_from_dream_integration(tmp_path):
    """7 社区 4 坏 3 好 → 只产出 3 个 skill；同原料再同步不覆盖不重复。"""
    comms = [
        _comm('{"raw": "json"}', ["p"]),               # bad: raw JSON
        _comm("We need to analyze trends", ["p"]),     # bad: 元话术
        _comm("report", []),                           # bad: patterns 空
        _comm("The task is to merge", ["p"]),          # bad: 元话术
        _comm("Feishu report delivery process", ["分步下发", "回执确认"], 93),
        _comm("Daily meeting summary workflow", ["要点提取", "待办生成"], 120),
        _comm("Onboarding checklist creation", ["环境准备", "权限申请"], 88),
    ]
    created = sync_from_dream(comms, str(tmp_path))
    assert len(created) == 3
    for name in created:
        assert os.path.exists(os.path.join(str(tmp_path), name, "SKILL.md"))
    assert sync_from_dream(comms, str(tmp_path)) == []  # 已有 skill 不覆盖不重复
