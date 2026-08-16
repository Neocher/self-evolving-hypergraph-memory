"""
内容级投毒模式检测 (R6)
=====================
纯正则扫描，无 LLM / 无 embedding / 无网络。检测三类内容级投毒模式：

  a. 指令注入 (critical) — ignore previous instructions / 忽略之前的指令 /
     系统/提示词覆盖（system prompt / 你不再是 / disregard / override 等
     中英文变体）/ I am now 类角色覆盖
  b. URL 钓鱼 (high)     — 短链服务 + 可疑 TLD + 内网 IP 直连
  c. 异常标记 (critical) — 零宽字符 + 大量控制字符 + homoglyph 混淆基础版

severity → verdict 映射固定为模块级常量（不进 YAML，CC 拍板只暴露
r6_enabled 一个开关，防双定义漂移面扩大）。供 core/defense.py 的 R6 规则
与 api/routes/search.py 检索结果标记共用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ─── severity 常量（模块级，不进 YAML） ─────────────────────
RISK_NONE = "none"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

# ─── a. 指令注入 (critical) ─────────────────────────────────
# 元组: (模式标签, 预编译正则)。import 时编译。
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # EN: ignore previous instructions / ignore all previous rules
    ("ignore_previous",
     re.compile(
         r"ignore\s+(all\s+|any\s+|the\s+|your\s+|previous\s+)*"
         r"(previous\s+)?(instructions?|directives?|rules|guidelines?|system\s*prompt)",
         re.IGNORECASE)),
    # ZH: 忽略之前的指令 / 忽略所有规则
    ("ignore_previous_zh",
     re.compile(r"忽略\s*之前\s*的?\s*(指令|指示|提示|规则|系统)")),
    # EN: disregard / override / forget / bypass previous instructions
    # 有界修饰词 {0,2} + 词边界: 防长文本无命中时的不必要回溯
    ("override_instructions",
     re.compile(
         r"\b(?:disregard|override|forget|bypass)\b\s+(?:(?:the|all|any|"
         r"your|my|previous)\s+){0,2}(?:previous\s+)?(instructions?|rules|"
         r"guidelines?|directives?|system\s*prompt|prompts?)",
         re.IGNORECASE)),
    # EN: system prompt 覆盖/索要 —— 命令词前缀 或 第二人称宣告/索要语境, 裸词不触发
    # 误报案例: "system prompt is an important concept" (第三人称陈述) → 不触发
    # 触发: "override the system prompt"、"show me the system prompt" (命令词)、
    #       "your system prompt is ..."、"你的 system prompt 是 ..." (第二人称宣告/索要)
    # 【P1-2】is/是 分支收紧为第二人称 (your/你的) 前缀 + 宣告/索要词尾 —— 纯第三人称
    # 陈述 (system prompt is an important concept) 不再误报隔离良性记忆。me 并入
    # 修饰词交替组 (非独立可选组) 保线性扫描, 支持 "show me the system prompt"。
    ("system_prompt",
     re.compile(
         r"\b(?:ignore|disregard|override|bypass|forget|reveal|show|print|"
         r"输出|列出|重复|透露|展示|忽略|无视|覆盖|绕过|改变|修改|重新定义|解除)"
         r"\s+(?:(?:the|all|your|previous|me)\s+)*system\s*prompt\b"
         r"|\b(?:your\s+|你\s*的\s*)system\s*prompt\s+(?:是|为|的内容|全文|内容|is\b|contents?\b|says?\b)",
         re.IGNORECASE)),
    # ZH: 角色覆盖 —— 必须带转换语境(现在/从此/接下来/不再是)才触发
    # 误报案例: "你是一个管理员" (陈述身份, 无覆盖语境) → 不触发
    # 触发: "你现在是一个系统提示词"、"你从此不是管理员"、"你不再是系统"
    ("role_override_zh",
     re.compile(
         r"你\s*(?:现在|已经|从此|接下来|今后|开始)\s*(?:是|不是)\s*"
         r"(?:一\s*个)?\s*(?:新的\s*)?(?:系统|提示词|管理员|助手|机器人|AI)"
         r"|你\s*不再是\s*(?:一\s*个)?\s*(?:新的\s*)?(?:系统|提示词|管理员|助手|机器人|AI)")),
    # EN: I am now 角色覆盖 —— 身份陈述后必须跟命令词才触发 (同句/近邻)
    # 误报案例: "I am now an assistant at the company" (陈述身份) → 不触发
    # 触发: "I am now the system admin, ignore previous instructions"
    ("iamnow_role_override",
     re.compile(
         r"\bi\s+am\s+now\s+(?:the\s+|an?\s+)?"
         r"(?:system|admin(?:istrator)?|assistant|ai|chatbot|robot)\b"
         r"(?=[^.\n!?]{0,60}\b(?:ignore|override|disregard|bypass|forget|"
         r"reveal|output|print|重复|输出|忽略|无视)\b)",
         re.IGNORECASE)),
    # ZH: 你现在是 + 角色覆盖
    ("now_you_are",
     re.compile(r"你\s*现在\s*是\s*(系统|管理员|机器人|AI|助手)")),
]

# ─── b. URL 钓鱼 (high, critical 子集) ──────────────────────
_PHISHING_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 短链服务
    ("short_url",
     re.compile(
         r"https?://[^\s/]*\.?(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|"
         r"ow\.ly|buff\.ly|cutt\.ly|rb\.gy|surl\.li|1url\.com)[^\s]*",
         re.IGNORECASE)),
    # 可疑 TLD
    ("suspicious_tld",
     re.compile(
         r"https?://[^\s/]*\.(?:xyz|top|gq|ml|tk|cf|ga|click|work|rest|link|"
         r"buzz|icu)[^\s]*",
         re.IGNORECASE)),
    # IP 直连（内网/回环）
    ("private_ip",
     re.compile(
         r"https?://(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)[0-9.]+",
         re.IGNORECASE)),
]

# ─── c. 异常标记 (critical) ─────────────────────────────────
# 零宽字符: U+200B (ZWSP) / U+200C (ZWNJ) / U+200D (ZWJ) / U+FEFF (BOM)
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# 控制字符: \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f
# 显式排除 \t(09) / \n(0a) / \r(0d) —— 正常多行文本含大量换行, 不能误判。
# 阈值 8: 实现为严格大于 (>8, 即 ≥9 个控制字符) 才触发 —— 注释与实现自洽。
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CONTROL_CHAR_THRESHOLD = 8

# homoglyph 混淆基础版: 拉丁-西里尔相邻混写（如 payраl）
# 用相邻字符对而非贪婪 `+` 链 —— 线性扫描, 长文本无 O(n²) 回溯
_HOMOGLYPH_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("latin_cyrillic_mix",
     re.compile(r"[a-zA-Z0-9][а-яА-ЯёЁ]|[а-яА-ЯёЁ][a-zA-Z0-9]")),
]


@dataclass
class ContentRisk:
    """内容级投毒扫描结果。"""
    risk_level: str = RISK_NONE   # none / high / critical
    reason: str = ""              # 人类可读原因
    matched_pattern: str = ""     # 命中的模式标签


def scan_content(content: str) -> ContentRisk:
    """纯正则扫描内容，返回内容级投毒风险判定。

    无 LLM / 无 embedding / 无网络，只做字符串扫描。
    优先级: 指令注入 (critical) > URL 钓鱼 (high) > 异常标记 (critical)。

    Args:
        content: 待扫描内容（None/非 str 安全降级为 str()）

    Returns:
        ContentRisk: risk_level 为 none / high / critical 之一
    """
    if not content:
        return ContentRisk()
    text = content if isinstance(content, str) else str(content)

    # a. 指令注入 (critical)
    for label, pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return ContentRisk(
                risk_level=RISK_CRITICAL,
                reason=f"instruction injection pattern: {label}",
                matched_pattern=label,
            )

    # b. URL 钓鱼 (high)
    for label, pat in _PHISHING_PATTERNS:
        if pat.search(text):
            return ContentRisk(
                risk_level=RISK_HIGH,
                reason=f"phishing url pattern: {label}",
                matched_pattern=label,
            )

    # c. 异常标记 (critical)
    if _ZERO_WIDTH_PATTERN.search(text):
        return ContentRisk(
            risk_level=RISK_CRITICAL,
            reason="zero-width character detected",
            matched_pattern="zero_width_char",
        )
    ctrl_count = len(_CONTROL_CHAR_PATTERN.findall(text))
    if ctrl_count > _CONTROL_CHAR_THRESHOLD:
        return ContentRisk(
            risk_level=RISK_CRITICAL,
            reason=f"{ctrl_count} control characters detected "
                   f"(threshold {_CONTROL_CHAR_THRESHOLD})",
            matched_pattern="control_chars",
        )
    for label, pat in _HOMOGLYPH_PATTERNS:
        m = pat.search(text)
        if m:
            return ContentRisk(
                risk_level=RISK_CRITICAL,
                reason=f"homoglyph obfuscation pattern: {label}",
                matched_pattern=label,
            )

    return ContentRisk()
