"""Attribute Extractor — 纯规则属性提取（无 LLM）。从 Episode 内容按实体锚点提取属性。"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

@dataclass(slots=True)
class PatternSpec:
    pattern: str
    partition: str
    entity_role: int = 2
    attr_name: str = ""
    local_conf: float = 0.5

@dataclass(slots=True)
class ExtractedAttribute:
    entity_id: str
    attr_name: str
    attr_value: str
    partition: str
    evidence_episode_id: str
    evidence_span: str
    local_conf: float

# entity_type -> attr_name -> [PatternSpec]
ATTRIBUTE_PATTERNS: dict[str, dict[str, list[PatternSpec]]] = {
    "Person": {
        "title": [
            PatternSpec(r"(?:现任|担任|出任)([^，。]+?)(?:[，。]|$)", "pattern_title_cn", 0, "title", 0.5),
            PatternSpec(r"(?:is|as) the (CEO|CTO|COO|CFO)", "pattern_title_en", 0, "title", 0.5),
        ],
        "company": [
            PatternSpec(r"(?:就职于|加入|任职于)(.+?公司)", "pattern_company_cn", 1, "company", 0.5),
            PatternSpec(r"(?:joined|works at) (.+?(?:Inc|Corp|Ltd|公司))", "pattern_company_en", 1, "company", 0.5),
        ],
    },
    "Organization": {
        "industry": [
            PatternSpec(r"(?:专注|深耕)(.+?领域)", "pattern_industry_cn", 0, "industry", 0.5),
            PatternSpec(r"(?:focuses on|specializes in) (.+?)", "pattern_industry_en", 0, "industry", 0.5),
        ],
        "founded": [
            PatternSpec(r"成立于(\d{4})年", "pattern_founded_cn", 0, "founded", 0.8),
            PatternSpec(r"founded in (\d{4})", "pattern_founded_en", 0, "founded", 0.8),
        ],
        "location": [
            PatternSpec(r"总部位于(.+?)[，。]", "pattern_location_cn", 0, "location", 0.6),
            PatternSpec(r"headquartered in (.+?)[,.]", "pattern_location_en", 0, "location", 0.6),
        ],
    },
}

_WINDOW = 80

def extract_attributes(episode_id: str, episode_content: str, entities: Sequence[Any]) -> list[ExtractedAttribute]:
    """在实体锚点 ±80 字符窗口内匹配属性模式。返回去重后的 ExtractedAttribute 列表。"""
    out: list[ExtractedAttribute] = []
    seen: set[tuple[str, str, str]] = set()
    for ent in entities:
        eid = getattr(ent, "entity_id", None) or getattr(ent, "id", None) or getattr(ent, "elementKey", None)
        if not eid:
            continue
        etype = getattr(ent, "entity_type", None) or getattr(ent, "type", None) or "Person"
        name = getattr(ent, "name", None) or ""
        names = [name] if name else []
        aliases = getattr(ent, "aliases", None) or []
        names.extend(aliases)
        patterns = ATTRIBUTE_PATTERNS.get(etype, {})
        if not patterns or not names:
            continue
        for nm in names:
            if not nm or len(str(nm)) < 2:
                continue
            for m in re.finditer(re.escape(str(nm)), episode_content):
                win = episode_content[max(0, m.start()-_WINDOW): m.end()+_WINDOW]
                for attr_name, specs in patterns.items():
                    for spec in specs:
                        for pm in re.finditer(spec.pattern, win):
                            val = (pm.group(1) if pm.groups() else pm.group(0)).strip()
                            if not val or len(val) > 40:
                                continue
                            key = (eid, attr_name, val)
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append(ExtractedAttribute(
                                entity_id=eid, attr_name=attr_name, attr_value=val,
                                partition=spec.partition, evidence_episode_id=episode_id,
                                evidence_span=win.strip()[:200], local_conf=spec.local_conf,
                            ))
    return out
