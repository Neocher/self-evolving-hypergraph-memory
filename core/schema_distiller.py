"""Schema 模式蒸馏器（阶段4-1，v6.0.0 纯规则）。

对标 Dream Pipeline 的 llm_patterns（LLM-gated，无 LLM 时恒空）：本模块以
SSM 回放语义的规则蒸馏替代——扫描 EpisodeNode 内容 → 提取频繁模式（文档频率
≥ min_support 的共享词/双字 gram）→ 属性化 Schema 节点（:Conceptual 标签）
落库 → 检索通道 _schema_recall 消费（cat1 聚合线索）。

纯规则、确定性、零外部依赖；run_once 为「评测前跑一轮蒸馏」入口（只读 +
幂等 upsert，不阻塞在线检索）。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

logger = logging.getLogger("shm.schema_distiller")

# 术语提取：拉丁词 + CJK 重叠双字 gram
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")
_CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "is", "are", "was", "were", "be", "been", "this", "that",
    "it", "i", "you", "he", "she", "we", "they", "my", "your", "our",
    "as", "by", "from", "into", "about", "than", "then", "there",
})

_LABEL = "Conceptual"  # Schema 节点 label（设计 4-1：:Conceptual 标签）


def extract_terms(content: str, max_terms: int = 8) -> list[str]:
    """内容 → 术语集：拉丁词（小写、停用词过滤）+ CJK 连续段（≥2 字符）。

    CJK 保留完整段（非双字 gram）：schema 落库 pattern_keywords 为空格连接串，
    检索 CONTAINS 子串匹配（查询 "机器学习" 命中段 "机器学习"）；拆 bigram 会
    碎词导致 CONTAINS 失配（实测）。
    """
    text = str(content or "")
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        if t and t not in seen and len(t) >= 2:
            seen.add(t)
            terms.append(t)

    for w in _WORD_RE.findall(text.lower()):
        if w not in _STOPWORDS:
            _add(w)
    for seq in _CJK_SEQ_RE.findall(text):
        _add(seq)
    return terms[:max_terms]


def distill(episodes: list[dict], min_support: int = 2,
            max_terms: int = 5) -> list[dict]:
    """SSM 回放式频繁模式蒸馏 → 属性化 Schema 节点 dict 列表。

    1. 逐 episode 提取术语 → 文档频率（df）计数
    2. df ≥ min_support 的术语为频繁项；每个 episode 取其频繁项集
    3. 频繁项集相同的 episode 归为一组（共享模式）→ 组支持数 ≥ min_support
       则产出 Schema 节点 {id, schema_name, pattern_keywords, support,
       source_ids, description, created_at}
    """
    if not episodes:
        return []
    docs: list[tuple[str, list[str]]] = []
    df: dict[str, int] = {}
    for ep in episodes:
        eid = str(ep.get("id") or "")
        content = ep.get("content") or ""
        terms = extract_terms(content)
        if not eid or not terms:
            continue
        docs.append((eid, terms))
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    frequent = {t for t, c in df.items() if c >= min_support}
    if not frequent:
        return []

    groups: dict[tuple[str, ...], list[str]] = {}
    for eid, terms in docs:
        shared = tuple(sorted(set(terms) & frequent))[:max_terms]
        if not shared:
            continue
        groups.setdefault(shared, []).append(eid)

    schemas: list[dict] = []
    for pattern in sorted(groups, key=lambda p: (-len(groups[p]), p)):
        src_ids = groups[pattern]
        if len(src_ids) < min_support:
            continue
        now = time.time()
        schemas.append({
            "id": str(uuid.uuid4()),
            "schema_name": "_".join(pattern[:3]),
            "pattern_keywords": list(pattern),
            "support": len(src_ids),
            "source_ids": json.dumps(src_ids, ensure_ascii=False),
            # 【实证】GraphLite GQL lexer 保留 description/desc（ORDER BY DESC 关键字）
    # → 属性命名 summary（与社区 report/summary 语义一致）
    "summary": (
        f"Schema 模式: {'、'.join(pattern)} 出现在 {len(src_ids)} 条记录中"
    ),
            "created_at": now,
        })
    return schemas


def run_once(store, limit: int = 200, min_support: int = 2) -> list[str]:
    """评测前蒸馏入口：读 EpisodeNode → 蒸馏 → 落库 Schema 节点（幂等）。

    只读 + 写（非在线检索热路径，评测 harness 显式调用）；任一步失败 → 返回
    []（不抛，不阻塞）。
    """
    try:
        if store is None or not hasattr(store, "query_cypher") \
                or not hasattr(store, "create_schema_node"):
            return []
        rows = store.query_cypher(
            "MATCH (e:EpisodeNode) "
            "WHERE (e.archived IS NULL OR e.archived = false) "
            "RETURN e.id AS id, e.content AS content LIMIT $limit",
            {"limit": int(limit)},
        )
        if not isinstance(rows, (list, tuple)) or not rows:
            return []
        episodes = []
        for row in rows:
            if isinstance(row, dict):
                episodes.append({"id": row.get("id", ""), "content": row.get("content", "")})
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                episodes.append({"id": row[0], "content": row[1]})
        schemas = distill(episodes, min_support=min_support)
        created: list[str] = []
        for schema in schemas:
            try:
                created.append(str(store.create_schema_node(schema)))
            except Exception:
                logger.debug("create_schema_node failed, skipping", exc_info=True)
        logger.info("Schema distill done: %d schemas, %d created", len(schemas), len(created))
        return created
    except Exception:
        logger.debug("Schema distill degraded", exc_info=True)
        return []
