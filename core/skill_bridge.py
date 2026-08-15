"""
记忆→Skill 桥接（Memory-Skill-Bridge）
====================================
将梦境巩固（dream consolidation）产出的社区摘要自动固化为 Hermes skill。

原理：
- 梦境阶段的 LLM 已产出社区 report + patterns（可复用模式），此处不做 LLM 调用
  （_dream_poll_loop 60s 间隔不能卡）
- 质量门三合一过滤 → 动作短语命名（kebab-case）→ 判重 → 生成 SKILL.md 写入
- Hermes skills_dir.rglob("SKILL.md") 递归扫描，直接写文件即被加载

约束：
- 不调用 LLM；命名失败跳过（宁缺毋滥）；每轮最多 3 个；不覆盖已有 skill
"""

from __future__ import annotations

import json
import logging
import os
import re

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "skills")
_MAX_SKILLS_PER_ROUND = 3
_MAX_NAME_WORDS = 6

# 扫描时排除的目录（.archive 归档技能、VCS/缓存等）
_SKIP_DIRS = {".archive", ".git", ".svn", "__pycache__", "node_modules", ".venv"}

# 元话术：LLM 未真正分析内容时的模板废话（中英文）
_META_PHRASES = (
    "we need to analyze",
    "the task is to",
    "我们需要分析",
    "我们需要对",
    "我们需要深入",
    "本报告将",
    "本报告旨在",
    "该任务是",
    "这个任务是",
    "我们将分析",
)

# 判重与命名共用的泛词（不承载语义，不参与 token 重叠/命名）
# 含社区 report 框架语：captures/contains/contain/including（"This cluster captures...,
# including..." 每个社区摘要必出现）；consists/various/diverse/includes/include
# （"This report consists of various..." 框架句必现，不同主题报告共享会误判重复）。
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "at", "by", "from", "this", "that", "these", "those", "it", "its",
    "is", "are", "be", "was", "were", "as", "we", "you", "they", "he",
    "she", "i", "my", "our", "your", "their", "about", "into", "including",
    "over", "under", "between", "among", "will", "can", "should", "would",
    "could", "may", "might", "must", "not", "no", "very", "just",
    "also", "then", "than", "cluster", "clusters", "captures", "contain",
    "contains", "consists", "various", "diverse", "includes", "include",
})

# 判重专用泛词（仅从语义重叠计数中排除；命名/description 中仍保留原文）
# 高频但不承载判别力：collection/operational/technical/set/logs（logs 词干归一后为 log）。
# 注意：不入 _STOPWORDS —— 否则 skill 名/description 会缺这些词；
#       report(s)/research 保留为判重信号词（report vs reports 归一判重依赖它）。
_DEDUP_GENERIC = frozenset({
    "collection", "operational", "technical", "set", "log",
})

# 开头序号/列表符号：1. / 1) / 1、 / - / * / • 等
_LEADING_LIST_RE = re.compile(r"^\s*(?:\d+(?:[\.\)、．:]\s*|\s)|[-*•·–—]\s+)+")


def _kebab(text: str) -> str:
    """任意文本 → kebab-case（仅保留 [a-z0-9]，连字符分隔）。"""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)


def _meaningful_tokens(text: str) -> set[str]:
    """提取有意义 token：词边界分词，排除停用词与纯数字。"""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}


# 判重场景高频词的显式词干归一白名单（复数/变形 → 基础形）。
# 不做通用尾缀剥离：盲剥会把 cases→cas、speed→spe、summary→summar、
# status→statu、houses→hous、news→new 剥坏（与 docstring 声称的守卫矛盾）。
# 判重场景只有几十个高频词，白名单比完整 Porter 词干更可靠、零误剥风险；
# 白名单外单词原样返回（news/analysis/process/status 天然不动）。
# 按 data/dream_candidates 实测频率补齐 12 对：analyses/processes/systems/
# documents/tools/files/notes/tables/findings/errors/comparisons/evaluations
# （systems×39/documents×34/tools×32/analyses×6 等高频变形未归一 → 近重复漏判）。
_STEM_WHITELIST = {
    "reports": "report",
    "delivery": "deliver",
    "deliveries": "deliver",
    "summaries": "summary",
    "cases": "case",
    "statuses": "status",
    "houses": "house",
    "logs": "log",
    "sets": "set",
    "analyses": "analysis",
    "processes": "process",
    "systems": "system",
    "documents": "document",
    "tools": "tool",
    "files": "file",
    "notes": "note",
    "tables": "table",
    "findings": "finding",
    "errors": "error",
    "comparisons": "comparison",
    "evaluations": "evaluation",
}


def _stem_word(word: str) -> str:
    """极简词干归一（判重专用）：report/reports、deliver/delivery 归一到同形。

    显式白名单映射（只收录判重场景高频词的复数/变形，analyses→analysis、
    processes→process 等），不做通用尾缀剥离；白名单外单词原样返回
    （cases→case、houses→house、summaries→summary、statuses→status 归一到
    同形，news/analysis/process/status 不动）。
    """
    return _STEM_WHITELIST.get(word, word)


def _dedup_tokens(text: str) -> set[str]:
    """判重专用 token 集：停用词过滤 → 词干归一 → 排除判重泛词。"""
    return {_stem_word(w) for w in _meaningful_tokens(text)} - _DEDUP_GENERIC


def _strip_code_fence(text: str) -> str:
    """剥离开头 ```lang 围栏（含行内 ```lang{...} 与尾部闭合 ```）。"""
    t = text.lstrip()
    m = re.match(r"^```[a-zA-Z0-9_+\-]*", t)
    if not m:
        return t
    t = t[m.end():].lstrip()
    if t.startswith("\n"):
        t = t[1:]
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.lstrip()


def _is_raw_json(report: str) -> bool:
    """report 是否为 raw JSON（裸对象/数组，或 ```json 围栏包裹）。"""
    return _strip_code_fence(report).startswith(("{", "["))


def _name_from_text(text: str, max_words: int = _MAX_NAME_WORDS) -> str:
    """文本 → kebab 名：剥序号/列表符号，过滤纯数字/停用词，截断到 max_words。

    【v5.37】长句不再整体作废——只取前 max_words 个有效词（原逻辑要求整行词数
    恰在 [2, _MAX_NAME_WORDS] 内，长 report/pattern 句全部 "cannot derive"）。
    """
    text = text.strip(" \t#*.")
    text = _LEADING_LIST_RE.sub("", text).strip(" \t#*.")
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
             if w not in _STOPWORDS and not w.isdigit()]
    return "-".join(words[:max_words])


def extract_reusable_patterns(community_summaries: list) -> list[dict]:
    """质量门三合一：patterns 非空 + report 非 raw JSON + report 非元话术。"""
    out = []
    for comm in community_summaries or []:
        if not isinstance(comm, dict):
            continue
        if not (comm.get("patterns") or []):
            continue
        report = (comm.get("report") or "").strip()
        if not report or _is_raw_json(report):
            continue
        if any(p in report.lower() for p in _META_PHRASES):
            continue
        out.append({
            "report": report,
            "patterns": comm["patterns"],
            "member_count": comm.get("member_count", 0),
        })
    return out


def generate_skill_name(report: str, patterns: list) -> str:
    """从 report/patterns 提取动作短语 → kebab-case 名。提取失败返回空串（跳过）。

    【v5.37】剥离序号/列表符号、过滤纯数字与停用词；长句截断到 _MAX_NAME_WORDS
    而非整体作废（保留 ≥2 词下限）；纯中文/无拉丁 token 时遍历全部 patterns 回退，
    仍失败则记 warning（不再静默跳过）。
    """
    for line in str(report or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        name = _name_from_text(line)
        if len(name.split("-")) >= 2:
            return name
    for p in patterns or []:
        cand = str(p.get("pattern", p)) if isinstance(p, dict) else str(p)
        name = _name_from_text(cand)
        if len(name.split("-")) >= 2:
            return name
    if str(report or "").strip() or patterns:
        logger.warning(
            "Skill-Bridge: cannot derive kebab name (report=%r, patterns=%r)",
            str(report or "")[:80], patterns,
        )
    return ""


def _scan_existing_skills(skills_dir: str) -> list[dict]:
    """扫描 skills 目录已有 SKILL.md → [{name, description}]（YAML frontmatter 解析）。

    【v5.37】改用 yaml.safe_load 解析 frontmatter（description: | 块不再解析成 '|'）；
    排除 .archive/.git 等目录。
    """
    out = []
    if not os.path.isdir(skills_dir):
        return out
    for dp, dirs, fns in os.walk(skills_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if os.path.basename(dp) in _SKIP_DIRS:
            continue
        for f in fns:
            if f != "SKILL.md":
                continue
            name = os.path.basename(dp)
            description = ""
            try:
                with open(os.path.join(dp, f), encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                parts = text.split("---", 2)
                if len(parts) >= 2:
                    data = yaml.safe_load(parts[1])
                    if isinstance(data, dict):
                        description = str(data.get("description") or "").strip()
            except (OSError, yaml.YAMLError, ValueError):
                continue
            out.append({"name": name, "description": description})
    return out


def should_create_skill(name: str, existing_skills: list) -> bool:
    """同名判重 + description 语义重叠检测。

    【v5.37】判重增强：停用词过滤后先词干归一（report/reports、deliver/delivery
    同形），再排除判重泛词（collection/operational/technical/set/logs）；≥3 个
    非泛词归一 token 词边界重叠才判语义重复。单/双词重叠（如 feishu、
    research+workflow）不阻断；仅泛词重叠（如 collection/operational/technical）
    不再误杀。
    """
    name_tokens = _dedup_tokens(name)
    if not name_tokens:
        return False
    for ex in existing_skills or []:
        ex_name = ex.get("name", "") if isinstance(ex, dict) else str(ex)
        if _kebab(ex_name) == name:
            return False
        ex_desc = (ex.get("description", "") if isinstance(ex, dict) else "") or ""
        if len(name_tokens & _dedup_tokens(ex_desc)) >= 3:
            return False
    return True


def generate_skill_md(entry: dict) -> str:
    """生成 SKILL.md：frontmatter(name/description/trigger) + 正文(场景/步骤/来源)。"""
    name = entry["name"]
    report = (entry.get("report") or "").strip()
    patterns = entry.get("patterns") or []

    desc = report.split("\n", 1)[0].strip(" -#*.")
    if len(desc) > 100:
        desc = desc[:100].rstrip() + "…"

    steps = []
    for i, p in enumerate(patterns, 1):
        step = p.get("pattern", p) if isinstance(p, dict) else str(p)
        steps.append(f"{i}. {step}")

    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(desc, ensure_ascii=False)}\n"
        f"trigger: When the user needs to {name.replace('-', ' ')}\n"
        "---\n\n"
        "## 场景\n\n"
        f"{report}\n\n"
        "## 步骤\n\n"
        + ("\n".join(steps) if steps else "- 无明确步骤，参考 report 中的模式\n") + "\n\n"
        "## 来源\n\n"
        f"- 来源: SHM 梦境巩固 (dream consolidation)\n"
        f"- 社区规模: {entry.get('member_count', 0)} 节点\n"
    )


def sync_from_dream(community_summaries: list, skills_dir: str | None = None) -> list[str]:
    """主入口：质量门 → 命名 → 判重 → 生成 → 写入。每轮最多 3 个。"""
    if skills_dir is None:
        skills_dir = _DEFAULT_SKILLS_DIR
    existing = _scan_existing_skills(skills_dir)
    created = []
    for entry in extract_reusable_patterns(community_summaries):
        if len(created) >= _MAX_SKILLS_PER_ROUND:
            break
        name = generate_skill_name(entry["report"], entry["patterns"])
        if not name:
            continue
        if not should_create_skill(name, existing):
            continue
        entry = {**entry, "name": name}
        target = os.path.join(skills_dir, name, "SKILL.md")
        if os.path.exists(target):  # 不覆盖已有 skill
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(generate_skill_md(entry))
        except OSError as e:
            logger.warning("Skill-Bridge write failed for %s: %s", name, e)
            continue
        existing.append({"name": name, "description": entry["report"].split("\n", 1)[0]})
        created.append(name)
        logger.info("Skill-Bridge created skill: %s", name)
    return created
