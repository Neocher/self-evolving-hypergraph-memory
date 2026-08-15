"""
Skill-Bridge 记忆→Skill 一体化测试
=================================
质量门三合一 + 动作短语命名 + 判重 + frontmatter + 集成（7 社区 4 坏 3 好 → 3 skill）。
"""
from __future__ import annotations

import os

import yaml

from core.skill_bridge import (
    _MAX_NAME_WORDS,
    _scan_existing_skills,
    _stem_word,
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


def test_generate_skill_name_long_report_truncates():
    """长英文 report（> _MAX_NAME_WORDS）→ 截断到前 N 个有效词，不再整体作废。"""
    report = ("This cluster captures a research and coding workflow including frequent "
              "generation of comprehensive multi-section reports with tables and headers")
    name = generate_skill_name(report, [])
    assert name == "research-coding-workflow-frequent-generation-comprehensive"
    assert len(name.split("-")) == _MAX_NAME_WORDS


def test_generate_skill_name_patterns_fallback_iterates():
    """patterns 回退遍历全部：第一个失败（无拉丁/单 token）→ 用第一个成功者。"""
    assert generate_skill_name("飞书日报自动化", ["分步下发", "send feishu daily report"]) == "send-feishu-daily-report"
    assert generate_skill_name("", ["single", "deliver feishu reports"]) == "deliver-feishu-reports"
    assert generate_skill_name("飞书日报自动化", ["分步下发", "要点提取"]) == ""


def test_should_create_skill():
    existing = [{"name": "feishu-report-delivery", "description": "deliver feishu reports daily"}]
    assert not should_create_skill("feishu-report-delivery", existing)  # 同名
    assert should_create_skill("feishu-delivery", existing)  # 归一后 2 词重叠（feishu+deliver）不阻断
    assert not should_create_skill("feishu-reports-daily", existing)  # ≥3 非泛词归一 token 重叠 → 判重
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


def test_should_create_skill_overlap_threshold():
    """重叠阈值：2 个 token 重叠（弱信号）不再阻断；≥3（强信号）阻断。"""
    existing = [{"name": "research-workflow-analysis", "description": "research workflow analysis"}]
    assert should_create_skill("research-workflow-tools", existing)  # {research, workflow} = 2 不阻断
    assert not should_create_skill("research-workflow-analysis-tools", existing)  # 3 token 阻断
    assert not should_create_skill("research-workflow-analysis", existing)  # 同名阻断


def test_should_create_skill_stemming():
    """词干归一：report/reports、delivery/deliver 归一后重叠 ≥3 → 判重（旧逻辑漏判）。"""
    existing = [{"name": "feishu-report-delivery", "description": "deliver feishu reports daily"}]
    # 归一后 {daily, report, deliver} 三重重叠 → 判重（未归一时重叠仅 daily=1，会漏判）
    assert not should_create_skill("daily-report-delivery", existing)
    # 归一后仅 {daily, report} = 2 重叠 → 不阻断
    assert should_create_skill("daily-report-check", existing)


def test_should_create_skill_generic_words_no_false_kill():
    """判重泛词（collection/operational/technical/set/logs）不参与计数：仅泛词重叠不误杀。"""
    existing = [
        {"name": "generating-threat-intelligence-reports",
         "description": "collection of operational intelligence and technical analysis reports"},
        {"name": "implementing-web-application-logging",
         "description": "set up operational logging for web applications"},
    ]
    # 真实候选：与已有 description 仅经泛词（collection/operational/technical）重叠 → 不阻断
    assert should_create_skill("collection-research-summaries-technical-guides-operational", existing)
    # 真实候选：仅经泛词（set/operational/logs）+ 信号词 research/report（各 1 个）重叠 → 不阻断
    assert should_create_skill("diverse-set-research-reports-operational-logs", existing)


def test_should_create_skill_analyses_stem():
    """词干归一补全回归：analyses→analysis 归一后重叠 ≥3 → 判重（旧逻辑漏放行近重复）。"""
    existing = [{"name": "research-workflow-analysis",
                 "description": "research workflow analysis"}]
    # 归一后 {research, workflow, analysis} 三重重叠 → 判重（旧逻辑 analyses 未归一仅 2 重叠漏判）
    assert not should_create_skill("research-workflow-analyses", existing)
    # 归一后仅 2 重叠（research+workflow）→ 不阻断
    assert should_create_skill("research-workflow-documentation", existing)


def test_should_create_skill_framework_words_no_false_kill():
    """框架词回归：不同主题长报告共享 consists/various/includes/report 不误判重复（旧逻辑阈值 3 误杀）。"""
    existing = [
        {"name": "data-pipeline-monitoring",
         "description": "This report consists of various monitoring metrics for data pipelines "
                        "including throughput and latency statistics"},
        {"name": "customer-onboarding-automation",
         "description": "This report consists of various onboarding steps and includes email "
                        "templates welcome sequences"},
    ]
    # 候选名若由旧逻辑从 "This report consists of various steps for customer support email
    # drafting" 提取，会含 report/consists/various → 与两条 description 各共享 ≥3 token 误杀
    assert should_create_skill("report-consists-various-steps-customer", existing)
    # 实质主题无重叠（marketing/campaign/analysis vs monitoring）→ 不阻断
    assert should_create_skill("marketing-campaign-analysis", existing)


def test_stem_word_whitelist():
    """_stem_word 白名单归一：复数/变形 → 基础形（判重同形，cases→case 等）。"""
    assert _stem_word("cases") == "case"
    assert _stem_word("houses") == "house"
    assert _stem_word("summaries") == "summary"
    assert _stem_word("statuses") == "status"
    assert _stem_word("reports") == "report"
    assert _stem_word("delivery") == "deliver"
    assert _stem_word("deliveries") == "deliver"
    assert _stem_word("logs") == "log"
    assert _stem_word("sets") == "set"
    # R9 补齐的高频变形（data/dream_candidates 实测频率）
    assert _stem_word("analyses") == "analysis"
    assert _stem_word("processes") == "process"
    assert _stem_word("systems") == "system"
    assert _stem_word("documents") == "document"
    assert _stem_word("tools") == "tool"
    assert _stem_word("files") == "file"
    assert _stem_word("notes") == "note"
    assert _stem_word("tables") == "table"
    assert _stem_word("findings") == "finding"
    assert _stem_word("errors") == "error"
    assert _stem_word("comparisons") == "comparison"
    assert _stem_word("evaluations") == "evaluation"


def test_stem_word_no_blind_strip():
    """反例：白名单外单词原样返回——speed/sing/thing/red/feed 与 -s/-es 固有词绝不剥。"""
    for w in ("speed", "sing", "thing", "red", "feed",
              "news", "analysis", "process", "status", "summary",
              "case", "house", "report", "deliver", "log", "set",
              "system", "document", "tool", "file", "note", "table",
              "finding", "error", "comparison", "evaluation"):
        assert _stem_word(w) == w


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
