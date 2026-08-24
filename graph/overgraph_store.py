"""OverGraphStore — 基于 OverGraph (Rust/PyO3, v0.17.0) 的图存储适配器。

SHM v6.0.0 OverGraph 引擎接入（阶段1）。与 GraphLiteStore 同接口契约：
`config/graph.backend: overgraph` 时经 api/app.py make_store(cfg) 装配，
上层业务（write/search/dashboard/query_router）零改动（duck-typing，
svc.graphlite_store 属性名保留）。FAISS 主通道同期替换为 OverGraph HNSW
（retrieval/vector_index.py VectorIndexAdapter）。

═══════════════════════════════════════════════════════════════════
R1 PoC 定标结论（2026-08-19, overgraph 0.17.0）——score 度量决策：
- vector_search(mode="dense", k, dense_query, label_filter={"labels": [...]})
  恒返回 **cosine 相似度** s∈[-1,1]（实证：相同向量→1.0，正交→0.0，
  相反→-1.0；dense_vector_metric="l2"/"cosine" 选项均接受但 score 输出不变）。
- 实证补强（R1 P2#9）：l2/cosine 双开库同向量对拍 score 逐位一致 —— 引擎
  完全忽略 dense_vector_metric 选项（open 时必传参数，纯占位）→ **d=1/s-1
  映射对两选项均成立**（非仅 cosine）；选项保留仅为引擎 open() 参数兼容。
- OverGraph 不暴露 L2 距离 → 采纳 design_overgraph_vector.md D5 cosine 分支：
  **distance d = 1/s - 1 (s>0)**，下游 1/(1+d) = s ∈ (0,1] 保持单调 + [0,1] 契约。
- s ≤ 0（正交/相反）视为非近邻剔除（FAISS 语义：非 top-k 不进结果），不足 k 补 -1。
- 归一化向量下 OverGraph 排名 == numpy cosine 排名 == FAISS FlatL2 排名（同序），
  → R3 HNSW 近似召回风险可控（测试以 Jaccard≥0.98 把关）。

═══════════════════════════════════════════════════════════════════
OverGraph GQL 语法契约（PoC 8 轮实证，design_overgraph_vector.md D 节）：
- elementKey 不参与 GQL（`e.key` 属性返回 None）→ GQL 一律匹配 props `id`
  （elementKey 策略 A2：node_id 直映 elementKey + id 同时落 props 一份）。
- `INSERT (n:Label {...})` 语句不支持（parse error）→ 翻译层：节点创建走
  typed upsert_node；边创建转 CREATE。
- `CREATE (n:Label {props})` 节点带属性 map 不支持（InvalidReturnExpression）
  → 节点创建只能 typed upsert_node。
- 逗号分隔 MATCH（`MATCH (a),(b)`）不支持 → 重复 MATCH（`MATCH (a) MATCH (b)`）。
- MERGE 不支持；NOT EXISTS 不支持 → OPTIONAL MATCH + count=0 等价改写
  （_rewrite_not_exists，实测语义；EXISTS 子查询/模式谓词均 parse error）。
- `RETURN e.*` 不支持 → `RETURN e`（_flatten_row 展开 props）。
- 裸 `RETURN 1 AS test`（无 MATCH 前缀）不支持 → 合成行。
- LIKE 不支持（parse error）→ CONTAINS（前缀语义等价，尾部 % 剥离）。
- 分号多语句不支持 → 每条 execute_gql 单语句（翻译层不产生分号）。
- 未标注 label 的节点 MATCH → 需 allow_full_scan=True（health 的 `MATCH (n)`）。
- 边 map 内 `weight` 是一等字段（GQL `r.weight` 读不到、返回 None）→ 边创建
  翻译为 CREATE + `SET __rN.weight = w`（写 props，r.weight 可读，与 GraphLite
  的 weight 属性语义一致）。
- dense_vector 一等字段 GQL 不可读写（`SET e.dense_vector` 写的是同名 props）
  → 向量写入只能 typed upsert_node(dense_vector=)（见 batch_upsert_embeddings）。
- 空串 $param：CONTAINS '' 恒真（同 GraphLite）→ 保留 '__SHM_NO_VALUE__' 哨兵。
- 中文原生直写直查，零 b64；CONTAINS 大小写敏感（P3c 双变体适用）。
- execute_gql 返回 dict（取 `r["rows"]`）；mutation_stats.nodes_updated 判 CAS。
- 事务：begin_write_txn + upsert_node_as/upsert_edge_as（ref dict
  `{"local": alias}` / `{"labels": [...], "key": ...}` / `{"id": N}`）+ commit/rollback。
"""

import logging
import os
import re
import threading
import time
import uuid
import hashlib
import json

import numpy as np

try:
    import overgraph
    from overgraph import OverGraph, OverGraphError
except ImportError as _e:  # pragma: no cover — 依赖缺失时给出明确指引
    raise ImportError(
        "OverGraph SDK 未安装。SHM v6.0.0 overgraph 后端需要: pip install overgraph>=0.17.0"
    ) from _e

# 复用 graph.common 的熔断器 / 缓存 / GQL 字面量 helpers（跨后端共享符号，
# v6.0.0 起不再依赖 graphlite_store；OverGraphCircuitBreaker 覆盖 infra 异常集）
from graph.common import (
    CircuitBreaker,
    CircuitBreakerOpen,
    _backup_corrupt_db,
    _dict_to_gql_set_values,
)
from core.retry import with_retry

logger = logging.getLogger("shm.overgraph_store")

# 【P0】熔断器计数的「基础设施异常」集合（设计 A7）：
# overgraph.OverGraphError 为唯一 SDK 成员（SDK 未区分连接失败/坏 GQL —— 与
# graphlite_sdk.QueryError 同权衡：两者都计数，比永不跳闸的死代码好）。
# 内置 ConnectionError/TimeoutError 保留以兼容测试 mock。
_INFRA_EXCEPTIONS = (
    OverGraphError,
    ConnectionError,
    TimeoutError,
)

# SHM 节点 label（与 GraphLite 后端共用同一套图模型）
LABEL_EPISODE = "EpisodeNode"
LABEL_HYPEREDGE = "HyperedgeNode"
LABEL_SESSION = "SessionNode"
LABEL_VISUAL = "VisualNode"
LABEL_PROPERTY_VER = "PropertyVerNode"
LABEL_COMMUNITY = "CommunityNode"
LABEL_SYSTEM = "SystemNode"
LABEL_CONFLICT = "ConflictNode"
LABEL_ENTITY = "EntityNode"
LABEL_MENTIONS = "MENTIONS"
LABEL_FACT = "AtomicFactNode"
LABEL_FACT_MENTIONS = "FACT_MENTIONS"
LABEL_ONTOLOGY_TYPE = "OntologyType"
LABEL_ONTOLOGY_ENTITY = "OntologyEntity"

# 每 (entity_id, attr_name) 属性版本上限（与 GraphLiteStore PROPERTY_MAX_VERSIONS 对齐）
PROPERTY_MAX_VERSIONS = 8


def _now() -> float:
    return time.time()


def _as_float32(vec) -> np.ndarray:
    """向量统一转 float32 ndarray（typed API 输入契约）。"""
    if isinstance(vec, np.ndarray):
        return np.asarray(vec, dtype=np.float32)
    return np.asarray(list(vec), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# GQL 白名单翻译层（SHM 101 处裸 GQL 的收敛点）
# ═══════════════════════════════════════════════════════════════

_CLAUSE_KEYWORDS = (
    "RETURN", "SET", "DELETE", "DETACH", "INSERT", "CREATE",
    "WHERE", "WITH", "UNWIND", "ORDER", "LIMIT", "SKIP", "CALL",
)
_CLAUSE_KEYWORD_SET = frozenset(_CLAUSE_KEYWORDS)
_EDGE_REL_RE = re.compile(r"-\[(?:(\w+):)?:?(\w+)\s*(\{([^}]*)\})?\]->")
_BARE_RETURN_RE = re.compile(r"^\s*RETURN\s+(.+?)(?:\s+AS\s+(\w+))?\s*$", re.S)


def _split_top_level_commas(s: str) -> list[str]:
    """按顶层逗号切分（跳过引号/括号/花括号内部）。"""
    out: list[str] = []
    depth = 0
    in_str = False
    esc = False
    cur = ""
    for ch in s:
        if in_str:
            cur += ch
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            cur += ch
        elif ch in "({[":
            depth += 1
            cur += ch
        elif ch in ")}]":
            depth = max(0, depth - 1)
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _split_comma_matches(query: str) -> str:
    """`MATCH (a), (b), ...` → `MATCH (a) MATCH (b) ...`。

    OverGraph 不支持逗号分隔的 read-prefix MATCH 列表（PoC 实证），
    只在括号深度 0 处按逗号切分 MATCH 前缀区（遇 RETURN/SET/... 结束）。
    """
    idx = query.find("MATCH")
    if idx < 0:
        return query
    i = idx + len("MATCH")
    depth = 0
    seg_start = idx
    end = len(query)
    parts: list[str] = []
    while i < len(query):
        ch = query[i]
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth = max(0, depth - 1)
        elif depth == 0 and ch == ",":
            parts.append(query[seg_start:i])
            seg_start = i + 1
        elif depth == 0 and ch.isspace():
            m = re.match(r"\s*([A-Za-z_]+)", query[i:])
            if m and m.group(1).upper() in _CLAUSE_KEYWORD_SET:
                end = i
                break
        i += 1
    parts.append(query[seg_start:end])
    return " MATCH ".join(p.strip() for p in parts if p.strip()) + query[end:]


def _strip_edge_weight(map_str: str) -> tuple[str, str | None]:
    """从边属性 map 提取 `weight: W`（一等字段，GQL 不可见）→ (剩余 map, W)。"""
    if not map_str.strip():
        return "", None
    parts: list[str] = []
    weight_val: str | None = None
    for part in _split_top_level_commas(map_str):
        m = re.match(r"^\s*weight\s*:\s*(.+?)\s*$", part)
        if m:
            weight_val = m.group(1).strip()
            continue
        parts.append(part)
    return ", ".join(parts), weight_val


def _rewrite_edges(query: str) -> str:
    """INSERT 边 → CREATE 边 + weight 移入 SET。

    `MATCH (a),(b) INSERT (a)-[:R {weight: w, x: 1}]->(b), (b)-[:R]->(a)`
    → `MATCH (a) MATCH (b) CREATE (a)-[__w0:R {x: 1}]->(b), (b)-[:R]->(a)
       SET __w0.weight = w`
    """
    if "INSERT" in query:
        query = query.replace("INSERT", "CREATE", 1)
    create_idx = query.find("CREATE")
    if create_idx < 0:
        return query
    head = query[:create_idx]
    tail = query[create_idx + len("CREATE"):]

    sets: list[str] = []
    weight_seq = [0]

    def _edge_repl(m: re.Match) -> str:
        rel_var, rel_label, map_str = m.group(1), m.group(2), m.group(4) or ""
        rest_map, weight_val = _strip_edge_weight(map_str)
        props_str = f" {{{rest_map.strip()}}}" if rest_map.strip() else ""
        if weight_val is None:
            var = f"{rel_var}:" if rel_var else ":"
            return f"-[{var}{rel_label}{props_str}]->"
        var = f"__w{weight_seq[0]}"
        weight_seq[0] += 1
        sets.append(f" SET {var}.weight = {weight_val}")
        return f"-[{var}:{rel_label}{props_str}]->"

    tail = _EDGE_REL_RE.sub(_edge_repl, tail)
    return head + "CREATE" + tail + "".join(sets)


def _rewrite_like(query: str) -> str:
    """`LIKE 'apple%'` → `CONTAINS 'apple'`（OverGraph 无 LIKE；前缀语义等价）。"""

    def _repl(m: re.Match) -> str:
        literal = m.group(1)
        if literal.endswith("%"):
            literal = literal[:-1]
        return f"CONTAINS '{literal}'"

    return re.sub(r"LIKE\s+'([^']*)'", _repl, query)


def _rewrite_star(query: str) -> str:
    """`RETURN e.*` → `RETURN e`（OverGraph 不支持 `.*` 展开）。"""
    return re.sub(r"\bRETURN\s+(\w+)\.\*", r"RETURN \1", query)


def _rewrite_offset(query: str) -> str:
    """`LIMIT N OFFSET M` → `SKIP M LIMIT N`。

    OverGraph 无 OFFSET（静默返回空）；且 SKIP 必须在 LIMIT 之前（PoC 实证
    `LIMIT 2 SKIP 2` 返回空）。dashboard 分页依赖此重写。
    """
    m_limit = re.search(r"\bLIMIT\s+(\d+)", query)
    m_offset = re.search(r"\bOFFSET\s+(\d+)", query)
    if not m_offset:
        return query
    q = re.sub(r"\bLIMIT\s+(\d+)", "", query)
    q = re.sub(r"\bOFFSET\s+(\d+)", "", q)
    parts: list[str] = [p for p in (q, f"SKIP {m_offset.group(1)}") if p]
    if m_limit:
        parts.append(f"LIMIT {m_limit.group(1)}")
    return " ".join(parts)


_NOT_EXISTS_RE = re.compile(
    r"\bWHERE\s+NOT\s+EXISTS\s*\{\s*"
    r"(\(\s*\w+\s*\))"                         # 源节点 (h)
    r"(\s*(?:<\[[^\]]*\]-|-\[[^\]]*\]->)\s*)"  # 边 -[:LABEL]->
    r"(\([^)]*\))"                             # 目标节点 (...)
    r"\s*\}",
    re.S,
)


def _rewrite_not_exists(query: str) -> str:
    """`WHERE NOT EXISTS { (h)-[:L]->() }` → OPTIONAL MATCH + count=0 过滤。

    OverGraph 不支持 EXISTS 子查询 / 模式谓词（PoC 实证：parse error；
    `NOT (h)-[:L]->()` 同样报 expected expression）——超边孤儿清理
    （graph/hyperedge.py purge_orphaned_hyperedges）的 NOT EXISTS 若直接透传
    会 parse error → query_cypher 永不抛契约吞掉 → orphan_count 恒 0 静默 no-op。
    等价改写（实测语义）：OPTIONAL MATCH 目标 + `WITH count(目标)=0` 过滤，
    与 NOT EXISTS（无出边）严格等价 —— 有出边 count>0 剔除，无出边 count=0 保留。
    `RETURN count(h)` 与 `DETACH DELETE h` 后缀均透传（聚合/删除作用域不变）。
    """
    m = _NOT_EXISTS_RE.search(query)
    if not m:
        return query
    src, edge, target = m.group(1), m.group(2), m.group(3)
    t = target.strip()[1:-1].strip()
    tm = re.match(r"^(\w+)?\s*(?::\s*(\w+))?\s*$", t)
    count_var = tm.group(1) if (tm and tm.group(1)) else "__m"
    label = tm.group(2) if tm else None
    new_target = f"({count_var}" + (f":{label}" if label else "") + ")"
    prefix = query[: m.start()].rstrip()
    suffix = query[m.end():]
    return (
        f"{prefix} OPTIONAL MATCH {src}{edge}{new_target} "
        f"WITH {src.strip('()')}, count({count_var}) AS __n WHERE __n = 0{suffix}"
    )


def _eval_bare_return(query: str) -> list[dict]:
    """裸 RETURN 合成（健康检查 `RETURN 1 AS test`；OverGraph 无无前缀 RETURN）。"""
    m = _BARE_RETURN_RE.match(query)
    if not m:
        return []
    expr, alias = m.group(1).strip(), m.group(2) or "result"
    low = expr.lower()
    if low == "true":
        val: object = True
    elif low == "false":
        val = False
    elif low == "null":
        val = None
    else:
        try:
            val = int(expr)
        except ValueError:
            try:
                val = float(expr)
            except ValueError:
                val = expr.strip("'\"")
    return [{alias: val}]


def _eval_value(v_raw: str, params: dict) -> object:
    """属性值求值：$param / 'str' / number / true/false/null / 原样。"""
    v = v_raw.strip()
    if v.startswith("$"):
        return params.get(v[1:], v)
    if v.startswith("'") and v.endswith("'") and len(v) >= 2:
        return v[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
        return v[1:-1].replace('\\"', '"')
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_props_map(map_str: str, params: dict | None) -> dict:
    """`{k: v, k2: $p, ...}` → props dict（字面量求值 + $param 查参）。"""
    props: dict = {}
    params = params or {}
    for part in _split_top_level_commas(map_str):
        seg = part.strip()
        if not seg:
            continue
        colon = seg.find(":")
        if colon < 0:
            continue
        k = seg[:colon].strip()
        v_raw = seg[colon + 1:].strip()
        if not k or not v_raw:
            continue
        props[k] = _eval_value(v_raw, params)
    return props


def _translate_gql(query: str, params: dict | None = None) -> list:
    """GraphLite 方言 → OverGraph 执行单元列表。

    返回 [(kind, payload)]：
      ("gql", (overgraph_gql, allow_full_scan)) —— 直接 execute_gql
      ("upsert_node", (label, props))            —— typed upsert_node
      ("synth", rows)                            —— 裸 RETURN 合成行
    """
    q = query.strip()
    if not q:
        return [("synth", [])]
    # 1. 裸 RETURN（无 MATCH 前缀）
    if re.match(r"^RETURN\b", q):
        return [("synth", _eval_bare_return(q))]
    # 2. INSERT 节点创建 → typed upsert_node（变量可选: `(:Label` 或 `(n:Label`）
    m = re.match(r"^\s*INSERT\s*\(\s*(?:\w+:)?:?(\w+)\s*\{([^{}]*)\}\s*\)\s*$", q, re.S)
    if m:
        props = _parse_props_map(m.group(2), params)
        return [("upsert_node", (m.group(1), props))]
    # 3. 独立 CREATE 节点（无 MATCH 前缀，dream_pipeline 超边/社区创建）
    if re.match(r"^CREATE\b", q):
        mc = re.match(r"^\s*CREATE\s*\(\s*(?:\w+:)?:?(\w+)\s*\{([^{}]*)\}\s*\)\s*$", q, re.S)
        if mc:
            props = _parse_props_map(mc.group(2), params)
            return [("upsert_node", (mc.group(1), props))]
        q2 = _rewrite_like(q)
        q2 = _rewrite_edges(q2)
        return [("gql", (q2, True))]
    # 4. MATCH 系：NOT EXISTS 改写（须先于逗号拆分，OPTIONAL MATCH/WITH 结构
    #    不能被按逗号切分）+ 逗号拆分 + 边创建重写 + e.* / LIKE / OFFSET 重写
    q2 = _rewrite_not_exists(q)
    q2 = _split_comma_matches(q2)
    q2 = _rewrite_edges(q2)
    q2 = _rewrite_like(q2)
    q2 = _rewrite_star(q2)
    q2 = _rewrite_offset(q2)
    return [("gql", (q2, True))]


def _prepare_params(params: dict | None) -> dict | None:
    """OverGraph 原生参数适配（与 GraphLite _interpolate 语义对齐）：

    【P0-2 哨兵收缩】空串 → '__SHM_NO_VALUE__' 哨兵仅作用于 read_validate 的
    $new_value 比较点（CONTAINS '' 恒真 → 空值矛盾漏检防护，与 GraphLite 一致）。
    不再替换通用 mutation 路径的全部字符串参数——合法空串属性（用户写入的空字段）
    不会被字面存储为哨兵（update_with_version 与通用路径同语义，一套行为）。
    - numpy 标量 → Python 标量；numpy 数组 → list（execute_gql 不支持 ndarray）
    """
    if not params:
        return None
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, str):
            out[k] = "__SHM_NO_VALUE__" if (not v and k == "new_value") else v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _node_key(props: dict) -> str:
    """elementKey 生成（A2：node_id 直映；无 id/name 时 uuid4 兜底）。"""
    for k in ("id", "name"):
        if props.get(k) is not None:
            return str(props[k])
    return str(uuid.uuid4())


class OverGraphCircuitBreaker(CircuitBreaker):
    """OverGraph 版熔断器：record_failure 门控用 overgraph 异常集。

    基类 CircuitBreaker.record_failure 的 isinstance 门控引用的是
    graphlite_store 模块级 _INFRA_EXCEPTIONS（graphlite_sdk 异常）——
    OverGraphError 不在其中 → 直接复用基类会让熔断器成为死代码
    （AGENTS.md「SDK 异常类型不匹配 → 熔断器死代码」坑）。
    设计 A7：overgraph.OverGraphError 为 _INFRA_EXCEPTIONS 唯一成员。
    """

    def record_failure(self, exc=None):
        if exc is not None and not isinstance(exc, _INFRA_EXCEPTIONS):
            return  # 应用错误不计数（同基类语义）
        # 已通过 overgraph 异常门控 → 显式失败信号（None）计数，不触发基类
        # 的 graphlite 门控（否则 OverGraphError 又被基类误判为非 infra 跳过）
        super().record_failure(None)


class OverGraphStore:
    """OverGraph-backed graph store（与 GraphLiteStore 同接口契约）。

    39 个公开方法对照 graph/graphlite_store.py：Episode CRUD / 属性版本 /
    超边 / 社区 / Hebbian / Session / Visual / 裸 GQL（翻译层）/ 生命周期，
    外加 4 个向量方法（vector_search_dense / batch_upsert_embeddings /
    get_episode_keys / get_node_internal_id）。
    """

    def __init__(self, config=None, cb_config=None):
        self._db: OverGraph | None = None
        self._db_path: str = ""
        self.config = config or type(
            "cfg", (), {"database_path": "", "dense_vector_dimension": 512,
                        "dense_vector_metric": "cosine"})()
        self.circuit_breaker = OverGraphCircuitBreaker(cb_config)
        # session 访问锁：OverGraph 引擎非线程安全防护 + 与 GraphLiteStore 相同的
        # 串行化契约（RLock 可重入，写线程内嵌套调用不死锁）
        self._session_lock = threading.RLock()
        # thread-local 降级信号（query_cypher 永不抛契约，P3a R7）
        self._local = threading.local()

    # ─── 连接 ─────────────────────────────────────────

    @property
    def conn(self):
        """兼容 GraphLiteStore.conn（暴露底层引擎对象）。"""
        if self._db is None:
            raise RuntimeError("OverGraphStore not connected")
        return self._db

    def _locked_execute_gql(self, gql: str, params: dict | None = None) -> dict:
        """串行化执行 GQL（mode=auto：SHM query_cypher 历史上可携带 mutation，
        GraphLite query()/execute() 均执行任意语句 —— 行为对齐）。
        allow_full_scan=True：OverGraph 未标注 label 的 MATCH 需要（health 的
        `MATCH (n)` / hebbian 的无 label 端点），对已标注查询无性能影响。
        """
        with self._session_lock:
            assert self._db is not None
            return self._db.execute_gql(
                gql, _prepare_params(params), mode="auto", allow_full_scan=True
            )

    def _locked_upsert_node(self, label: str, key: str, props: dict,
                            dense_vector=None) -> None:
        """串行化 typed upsert_node（props 剔 None —— 对齐 GraphLite 缺省字段
        IS NULL 语义）。"""
        with self._session_lock:
            assert self._db is not None
            clean = {k: v for k, v in props.items() if v is not None}
            self._db.upsert_node(label, str(key), props=clean, dense_vector=dense_vector)

    def connect(self) -> None:
        """Open/create OverGraph DB（dense_vector_dimension 必配，否则向量写入报错）。

        edge_uniqueness=False（默认）：GQL CREATE 重复建边不抛错 —— 与 GraphLite
        INSERT 的静默重复语义一致（SHM 侧由 MATCH-exists 守卫防重复积累）。
        """
        db_path = getattr(self.config, "database_path", "") or os.path.join(
            os.path.dirname(__file__), "..", "data", "shm_overgraph_db"
        )
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        dim = int(getattr(self.config, "dense_vector_dimension", 512))
        metric = str(getattr(self.config, "dense_vector_metric", "cosine"))
        try:
            self._db = OverGraph.open(
                db_path,
                dense_vector_dimension=dim,
                dense_vector_metric=metric,
                edge_uniqueness=False,
            )
        except Exception:
            # open 失败（WAL/段损坏）→ 备份损坏库再 re-raise（同 GraphLiteStore）
            _backup_corrupt_db(db_path)
            raise
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """尽力创建 EpisodeNode 属性索引（P1-2 超边窗口查询等价），失败仅日志。

        OverGraph 索引 spec 为单字段 dict（`{field: "property", key: field}` +
        kind="equality"）；created_at 为保留元数据字段不可索引 → 仅 source。
        """
        try:
            self._db.ensure_node_property_index(
                LABEL_EPISODE,
                {"fields": [{"source": "property", "key": "source"}],
                 "kind": "equality"},
            )
        except Exception as e:
            logger.warning("EpisodeNode source index skipped (non-fatal): %s", e)
        logger.debug("EpisodeNode source index ensured")

    # ─── Episode CRUD ─────────────────────────────────

    def create_episode(self, episode: dict) -> str:
        """创建 EpisodeNode（typed upsert_node）。Returns id.

        【Archive-Supersedes】写时基线：新节点默认 archived=false（一处覆盖 4 个
        写入点）；【Source-Trust】默认 direct；【Dual-Track】默认 active ——
        与 GraphLiteStore.create_episode 完全一致。
        """
        episode.setdefault("archived", False)
        episode.setdefault("source_type", "direct")
        episode.setdefault("fact_track", "active")
        eid = str(episode.get("id", str(uuid.uuid4())))
        props = dict(episode)
        props["id"] = eid  # id 同时落 props 一份（A2 读侧零改动）
        props.setdefault("version", 1)  # 乐观锁基线（OverGraph upsert_node 可直带）
        self._locked_upsert_node(LABEL_EPISODE, eid, props)
        return eid

    def get_episode(self, node_id: str) -> dict | None:
        """get_node_by_key 取 EpisodeNode（elementKey 一级索引 O(1)）。"""
        try:
            view = self._db.get_node_by_key(LABEL_EPISODE, str(node_id))
        except Exception:
            return None
        if view is None:
            return None
        return self._flatten_view(view)

    @with_retry(
        max_attempts=2, base_delay=0.2, backoff=2.0,
        retryable_exceptions=_INFRA_EXCEPTIONS,
    )
    def _get_episodes_batch_retryable(self, node_ids: list[str]) -> list[dict]:
        """底层批量查询（熔断门控 + 重试）：成功返回 episodes。

        - open 状态 raise CircuitBreakerOpen（query_router L1 传播链入口）
        - 基础设施错误交给 with_retry 重试；失败计数由 get_episodes_batch 统一记录
        - 应用错误不计数、不重试，返回 []
        """
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpen("circuit breaker open, batch lookup rejected")
        try:
            views = self._db.get_nodes_by_keys(
                [{"labels": [LABEL_EPISODE], "key": str(i)} for i in node_ids]
            )
        except _INFRA_EXCEPTIONS:
            raise
        except Exception:
            return []
        self.circuit_breaker.record_success()
        return [self._flatten_view(v) for v in views if v is not None]

    def get_episodes_batch(self, node_ids: list[str]) -> list[dict]:
        """Batch GET by ids（熔断门控 + 重试，契约同 GraphLiteStore）。"""
        if not node_ids:
            return []
        try:
            return self._get_episodes_batch_retryable(node_ids)
        except _INFRA_EXCEPTIONS as e:
            try:
                self.circuit_breaker.record_failure(e)
            except CircuitBreakerOpen:
                raise CircuitBreakerOpen(
                    "circuit breaker open, batch lookup rejected"
                ) from e
            return []

    def get_active_episodes(self, time_window_seconds: float = 1800) -> list[dict]:
        """Get recently created episodes（created_at 业务 float 秒属性保真）。"""
        cutoff = _now() - time_window_seconds
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_EPISODE}) WHERE e.created_at >= {cutoff} RETURN e"
            )
            return [self._flatten_row(r, "e") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    # ── EntityNode（Schema 自演化持久化，v6.1.0 P0-①）──
    def create_entity(self, entity_name: str, entity_type: str = "Person",
                      props: dict | None = None) -> str:
        """创建/复用 EntityNode（规范化 name 确定性 key 幂等）。Returns entity id.

        Schema 自演化持久化闭环（P0-①）：梦境实体链接产出的实体落库，
        key = ent_<sha1(norm_name)> 保证同名复用不重复建节点。
        """
        name = (entity_name or "").strip()
        if not name:
            raise OverGraphError("entity name required")
        norm = name.lower().strip()
        eid = f"ent_{hashlib.sha1(norm.encode('utf-8')).hexdigest()[:16]}"
        p = dict(props or {})
        p["id"] = eid
        p["name"] = name
        p["norm_name"] = norm
        p["entity_type"] = entity_type
        p.setdefault("archived", False)
        p.setdefault("created_at", _now())
        self._locked_upsert_node(LABEL_ENTITY, eid, p)
        return eid

    def create_atomic_fact(self, subject: str, predicate: str, object_: str,
                           valid_time: str = "", source_episode: str = "",
                           confidence: float = 1.0,
                           props: dict | None = None) -> str:
        """创建/复用 AtomicFactNode（sha1 确定性 key 幂等）。Returns fact id.

        P0-③ AtomicFact 事实级中间层：Episode 拆 subject-predicate-object/time
        原子事实单元，key = fact_<sha1(subject|predicate|object|valid_time)> 保证
        同事实同版本不重复建节点；source_episode 记证据链 + FACT_MENTIONS 边。
        """
        subj = (subject or "").strip()
        pred = (predicate or "").strip()
        obj = (object_ or "").strip()
        if not subj or not pred or not obj:
            raise OverGraphError("atomic fact requires subject/predicate/object")
        vt = (valid_time or "").strip()
        raw = "|".join([subj.lower(), pred.lower(), obj.lower(), vt.lower()])
        fid = f"fact_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"
        p = dict(props or {})
        p["id"] = fid
        p["subject"] = subj
        p["predicate"] = pred
        p["object"] = obj
        p["valid_time"] = vt
        p["source_episode"] = str(source_episode or "")
        p["confidence"] = float(confidence)
        p.setdefault("archived", False)
        p.setdefault("created_at", _now())
        self._locked_upsert_node(LABEL_FACT, fid, p)
        if source_episode:
            try:
                frm = self._require_internal_id(fid, LABEL_FACT)
                to = self._require_internal_id(str(source_episode), LABEL_EPISODE)
                self._ensure_edge(frm, to, LABEL_FACT_MENTIONS)
            except Exception:
                pass  # 边失败不阻塞（节点已落库）
        return fid

    def get_atomic_facts_by_episode(self, episode_id: str, limit: int = 50) -> list[dict]:
        """EpisodeNode → FACT_MENTIONS → AtomicFactNode（梦境反查/审计）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (f:{LABEL_FACT})-[r:{LABEL_FACT_MENTIONS}]->(ep:{LABEL_EPISODE}) "
                f"WHERE ep.id = '{episode_id}' RETURN f.id AS id, f.subject AS subject, "
                f"f.predicate AS predicate, f.object AS object, f.valid_time AS valid_time "
                f"LIMIT {int(limit)}"
            )
            rows = (result or {}).get("rows", [])
            out = []
            for r in rows:
                if isinstance(r, dict):
                    out.append({
                        "id": str(r.get("id", "")),
                        "subject": str(r.get("subject", "")),
                        "predicate": str(r.get("predicate", "")),
                        "object": str(r.get("object", "")),
                        "valid_time": str(r.get("valid_time", "")),
                    })
            return out
        except Exception:
            return []

    def get_atomic_facts_by_subject(self, subject: str, limit: int = 50) -> list[dict]:
        """按 subject 查 AtomicFactNode（检索候选定位）。"""
        subj = (subject or "").strip()
        if not subj:
            return []
        try:
            result = self._locked_execute_gql(
                f"MATCH (f:{LABEL_FACT}) WHERE f.subject CONTAINS '{subj}' "
                f"OR f.subject CONTAINS '{subj.lower()}' "
                f"AND (f.archived IS NULL OR f.archived = false) "
                f"RETURN f.id AS id, f.subject AS subject, f.predicate AS predicate, "
                f"f.object AS object, f.valid_time AS valid_time "
                f"LIMIT {int(limit)}"
            )
            rows = (result or {}).get("rows", [])
            out = []
            for r in rows:
                if isinstance(r, dict):
                    out.append({
                        "id": str(r.get("id", "")),
                        "subject": str(r.get("subject", "")),
                        "predicate": str(r.get("predicate", "")),
                        "object": str(r.get("object", "")),
                        "valid_time": str(r.get("valid_time", "")),
                    })
            return out
        except Exception:
            return []

    def get_entity(self, entity_name: str) -> dict | None:
        """按规范化 name 查 EntityNode（确定性 key O(1)）。"""
        norm = (entity_name or "").strip().lower()
        if not norm:
            return None
        eid = f"ent_{hashlib.sha1(norm.encode('utf-8')).hexdigest()[:16]}"
        try:
            view = self._db.get_node_by_key(LABEL_ENTITY, eid)
        except Exception:
            return None
        return self._flatten_view(view) if view is not None else None

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        try:
            view = self._db.get_node_by_key(LABEL_ENTITY, str(entity_id))
        except Exception:
            return None
        return self._flatten_view(view) if view is not None else None

    def link_entity_to_episode(self, entity_name: str, episode_id: str,
                               entity_type: str = "Person") -> str:
        """实体 → EpisodeNode MENTIONS 边（幂等）。Returns entity id."""
        eid = self.create_entity(entity_name, entity_type=entity_type)
        try:
            frm = self._require_internal_id(eid, LABEL_ENTITY)
            to = self._require_internal_id(str(episode_id), LABEL_EPISODE)
            self._ensure_edge(frm, to, LABEL_MENTIONS)
        except Exception:
            pass  # 边失败不阻塞（节点已落库）
        return eid

    def get_entity_episodes(self, entity_name: str, limit: int = 50) -> list[str]:
        """EntityNode → MENTIONS → EpisodeNode ids（检索候选定位）。"""
        eid = self.get_entity(entity_name)
        if not eid:
            return []
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_ENTITY})-[r:{LABEL_MENTIONS}]->(ep:{LABEL_EPISODE}) "
                f"WHERE e.id = '{eid.get('id')}' RETURN ep.id AS ep_id LIMIT {int(limit)}"
            )
            rows = (result or {}).get("rows", [])
            out = []
            for r in rows:
                ep = r.get("ep_id") if isinstance(r, dict) else None
                if ep:
                    out.append(str(ep))
            return out
        except Exception:
            return []

    def get_entity_episodes_by_episode(self, episode_id: str, limit: int = 100) -> list[str]:
        """反查：EpisodeNode → MENTIONS → EntityNode names（Schema 演化全量重放用）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_ENTITY})-[r:{LABEL_MENTIONS}]->(ep:{LABEL_EPISODE}) "
                f"WHERE ep.id = '{episode_id}' RETURN e.name AS ent_name LIMIT {int(limit)}"
            )
            out = []
            for row in (result or {}).get("rows", []):
                nm = row.get("ent_name") if isinstance(row, dict) else None
                if nm:
                    out.append(str(nm))
            return out
        except Exception:
            return []

    def get_entities(self, limit: int = 200) -> list[dict]:
        """列出实体（调试/评测/梦境消费）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_ENTITY}) RETURN e LIMIT {int(limit)}"
            )
            return [self._flatten_row(r, "e") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    # ── Schema 自进化 P0-②（实体属性/关系演化，v6.2.0）───────────────

    def locked_update_entity_props(self, entity_id: str, mutator) -> dict:
        """锁内读-改-写 EntityNode.props（sidecar 演化安全）。

        _locked_upsert_node 是整包替换（非字段合并）→ 属性/关系侧车累积必须
        在 session_lock 内 read-modify-write，避免并发写丢失。Returns 新 props。
        """
        with self._session_lock:
            assert self._db is not None
            try:
                view = self._db.get_node_by_key(LABEL_ENTITY, str(entity_id))
            except Exception:
                view = None
            props = self._flatten_view(view) if view is not None else {"id": str(entity_id)}
            new_props = mutator(props)
            if new_props is None:
                new_props = props
            clean = {k: v for k, v in new_props.items() if v is not None}
            self._db.upsert_node(LABEL_ENTITY, str(entity_id), props=clean)
            return dict(new_props)

    def create_rel_edge(self, src_entity_id: str, dst_entity_id: str,
                        predicate: str, confidence: float = 0.6,
                        evidence_episode_ids: list[str] | None = None) -> str:
        """EntityNode → EntityNode 谓词边（REL_<PRED>，三元组幂等）。

        边创建后不原地更新（_ensure_edge 语义）：confidence 是固化时快照，
        后续演化只写 rels_json 侧车。
        """
        label = f"REL_{predicate}"
        if not re.match(r"^[A-Z][A-Z0-9_]{0,63}$", str(predicate)):
            raise OverGraphError(f"invalid predicate label: {str(predicate)[:40]}")
        rel_key = f"rel:{predicate}:{src_entity_id}:{dst_entity_id}"
        key_hash = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()[:16]
        try:
            frm = self._require_internal_id(str(src_entity_id), LABEL_ENTITY)
            to = self._require_internal_id(str(dst_entity_id), LABEL_ENTITY)
        except Exception:
            raise OverGraphError(
                f"rel edge endpoints missing: {str(src_entity_id)[:12]} -> {str(dst_entity_id)[:12]}"
            )
        self._ensure_edge(
            frm, to, label,
            props={"predicate": predicate, "confidence": float(confidence),
                   "evidence_episode_ids": list(evidence_episode_ids or []),
                   "rel_key": key_hash},
            weight=float(confidence),
        )
        return key_hash

    def get_rel_neighbors(self, entity_id: str,
                          predicates: list[str] | None = None) -> list[dict]:
        """1 跳谓词邻居（P2 检索通道用）。返回 [{dst_id, predicate, confidence}]。

        边 props 含 predicate → 不需要枚举 label，GQL 全出边后按 props 过滤。
        """
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_ENTITY})-[r]->(n:{LABEL_ENTITY}) "
                f"WHERE e.id = '{entity_id}' "
                f"RETURN n.id AS dst_id, r.predicate AS predicate, "
                f"r.confidence AS confidence LIMIT 200"
            )
            out = []
            for row in (result or {}).get("rows", []):
                pred = row.get("predicate") if isinstance(row, dict) else None
                if not pred:
                    continue
                if predicates and pred not in predicates:
                    continue
                out.append({
                    "dst_id": str(row.get("dst_id") or ""),
                    "predicate": str(pred),
                    "confidence": float(row.get("confidence") or 0.0),
                })
            return out
        except Exception:
            return []

    @staticmethod
    def _decode_sidecar(props: dict, key: str) -> dict:
        """读取 attrs_json / rels_json 侧车（兼容 dict 或 JSON 字符串存储）。"""
        raw = props.get(key)
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return {}

    def get_entity_attributes(self, entity_id: str) -> dict:
        """读 EntityNode attrs_json 侧车（候选 + 固化 + 证据）。"""
        ent = self.get_entity_by_id(str(entity_id))
        if not ent:
            return {}
        return self._decode_sidecar(ent, "attrs_json")

    def get_entity_relations(self, entity_id: str) -> dict:
        """读 EntityNode rels_json 侧车 + 已固化 REL 边（出边 + 入边）。

        出边存 src 视角（sidecar 内），入边以 direction=in 标记补全（消费方
        dst 视角可读）。已固化边只更新 confidence/solidified，不覆盖侧车候选
        字段（target_name/votes/evidence）。
        """
        ent = self.get_entity_by_id(str(entity_id))
        sidecar: dict = {}
        if ent:
            sidecar = self._decode_sidecar(ent, "rels_json")
        # 合并已固化 REL 边（GQL 侧，出边 + 入边）
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_ENTITY})-[r]->(n:{LABEL_ENTITY}) "
                f"WHERE e.id = '{entity_id}' AND r.predicate IS NOT NULL "
                f"RETURN n.id AS dst_id, r.predicate AS predicate, "
                f"r.confidence AS confidence, 'out' AS direction LIMIT 200"
            )
            result_in = self._locked_execute_gql(
                f"MATCH (n:{LABEL_ENTITY})-[r]->(e:{LABEL_ENTITY}) "
                f"WHERE e.id = '{entity_id}' AND r.predicate IS NOT NULL "
                f"RETURN n.id AS dst_id, r.predicate AS predicate, "
                f"r.confidence AS confidence, 'in' AS direction LIMIT 200"
            )
            rows = ((result or {}).get("rows", []) + (result_in or {}).get("rows", []))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pred = row.get("predicate")
                if not pred:
                    continue
                dst = str(row.get("dst_id") or "")
                slot = sidecar.setdefault(str(pred), {}).setdefault(dst, {})
                # 只更新固化信息，保留候选证据字段
                slot["confidence"] = float(row.get("confidence") or 0.0)
                slot["solidified"] = True
                if row.get("direction") == "in":
                    slot["direction"] = "in"
        except Exception:
            pass
        return sidecar

    def get_episodes_by_tau_range(self, min_tau: float, max_tau: float,
                                  limit: int = 100) -> list[dict]:
        """Filter by tau range。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_EPISODE}) WHERE e.tau_initial >= {min_tau} "
                f"AND e.tau_initial <= {max_tau} RETURN e LIMIT {int(limit)}"
            )
            return [self._flatten_row(r, "e") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def update_with_version(self, node_id: str, updates: dict,
                            expected_version: int | None) -> bool:
        """乐观锁更新（CAS 单条 GQL：WHERE version SET，读 mutation_stats.nodes_updated）。

        expected_version=None → force 写入（跳过版本检查、不递增 version —— 保持
        v5.31.0 语义）。节点不存在 / version 不匹配 / 旧数据无 version → False。
        PoC 实证：重复 CAS（版本已变）nodes_updated=0；成功=1。
        """
        set_clause = _dict_to_gql_set_values(updates, skip_keys={"id", "version"})
        if not set_clause:
            return True
        with self._session_lock:
            assert self._db is not None
            try:
                if expected_version is None:
                    self._db.execute_gql(
                        f"MATCH (e:{LABEL_EPISODE} {{id: $id}}) SET {set_clause}",
                        {"id": str(node_id)}, mode="auto", allow_full_scan=True,
                    )
                    return True
                nxt = int(expected_version) + 1
                result = self._db.execute_gql(
                    f"MATCH (e:{LABEL_EPISODE} {{id: $id}}) "
                    f"WHERE e.version = $v "
                    f"SET e.version = {nxt}, {set_clause}",
                    {"id": str(node_id), "v": int(expected_version)},
                    mode="auto", allow_full_scan=True,
                )
                ms = (result or {}).get("mutation_stats") or {}
                return int(ms.get("nodes_updated", 0)) > 0
            except Exception:
                return False

    def archive_node(self, node_id: str, replacement_id: str | None = None) -> bool:
        """标记 EpisodeNode 归档（archived=true）；replacement_id 非空时建
        SUPERSEDES 血统边（幂等守卫，防重复边）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_EPISODE} {{id: $id}}) SET e.archived = true",
                {"id": str(node_id)},
            )
        except Exception:
            logger.warning("archive_node failed for %s", str(node_id)[:12], exc_info=True)
            return False
        ms = (result or {}).get("mutation_stats") or {}
        if int(ms.get("nodes_updated", 0)) <= 0:
            return False
        if replacement_id:
            try:
                self._ensure_edge(
                    self._require_internal_id(node_id),
                    self._require_internal_id(replacement_id),
                    "SUPERSEDES",
                )
            except Exception:
                logger.warning(
                    "archive_node: SUPERSEDES edge insert failed for %s -> %s",
                    str(node_id)[:12], str(replacement_id)[:12], exc_info=True)
        return True

    def unarchive(self, node_id: str) -> bool:
        """撤销归档：SET archived=false（幂等；节点不存在 → False）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (e:{LABEL_EPISODE} {{id: $id}}) SET e.archived = false",
                {"id": str(node_id)},
            )
        except Exception:
            logger.warning("unarchive failed for %s", str(node_id)[:12], exc_info=True)
            return False
        ms = (result or {}).get("mutation_stats") or {}
        return int(ms.get("nodes_updated", 0)) > 0

    # ─── Property Version CRUD（P0-1 实体-属性-时间三维建模）────────

    def create_property_version(
        self,
        entity_id: str,
        attr_name: str,
        value: str,
        valid_from: float | None = None,
        supersedes_id: str | None = None,
        superseded_by: str | None = None,
        formula: str | None = None,
        filter_: str | None = None,
        owner: str | None = None,
    ) -> str:
        """INSERT PropertyVerNode + SUPERSEDES 血统边（begin_write_txn 多语句原子）。

        设计 A4：create_property_version 补偿链用 WriteTxn —— 全部编排 stage 进
        单个事务，commit 原子落库；任一步失败 rollback 整体回滚（补偿由事务原子性
        免费提供，替代 GraphLiteStore 的手工补偿链）。语义与 GraphLiteStore 对齐：
        - 新版本不写 expired_at（IS NULL 即当前有效）
        - supersedes_id 非空 → 旧版本打 expired_at + (old)-[:SUPERSEDES]->(new)
        - superseded_by 非空（乱序中段插入 R4/R6）→ 新版本打 expired_at=后继
          valid_from + (new)-[:SUPERSEDES]->(succ)；中段时先删旧 P→S 边防分支图
        - 【2026-08-23 口径治理】formula/filter_/owner 可选口径字段（借鉴
          《本体论增强问数》指标构件：公式/过滤/owner 为治理资产）；None 不写
          字段（向后兼容，旧节点零迁移）
        """
        pid = str(uuid.uuid4())
        now = valid_from if valid_from is not None else time.time()
        # 前置读（事务外）：后继 valid_from（R6-P1 时机修复；读不到 → 一致性错误）
        succ_ts: float = now
        succ_int: int | None = None
        if superseded_by:
            succ = self._db.get_node_by_key(LABEL_PROPERTY_VER, str(superseded_by))
            if succ is None or succ.props.get("valid_from") is None:
                raise OverGraphError(
                    f"successor property version not found: {str(superseded_by)[:12]}"
                )
            succ_ts = float(succ.props["valid_from"])
            succ_int = int(succ.id)
        # 前置读：前驱节点（合并 expired_at 用；upsert_node 整体替换 props）
        old_props: dict | None = None
        if supersedes_id:
            old = self._db.get_node_by_key(LABEL_PROPERTY_VER, str(supersedes_id))
            if old is not None:
                old_props = dict(old.props)

        new_props: dict = {
            "id": pid, "entity_id": str(entity_id), "attr_name": attr_name,
            "value": str(value), "valid_from": now,
        }
        # 【2026-08-23 口径治理】可选口径字段（None 不写，向后兼容）
        if formula is not None:
            new_props["formula"] = str(formula)
        if filter_ is not None:
            new_props["filter"] = str(filter_)
        if owner is not None:
            new_props["owner"] = str(owner)
        if superseded_by:
            new_props["expired_at"] = succ_ts
        try:
            txn = self._db.begin_write_txn()
        except Exception:
            logger.exception("create_property_version: begin_write_txn failed")
            raise
        try:
            txn.upsert_node_as("new", LABEL_PROPERTY_VER, pid, props=new_props)
            if old_props is not None:
                # 中段插入：先删旧 P→S 边（防分支图，R4-P1）
                if superseded_by and succ_int is not None:
                    try:
                        old_succ = self._db.get_edge_by_triple(
                            self._require_internal_id(str(supersedes_id),
                                                      LABEL_PROPERTY_VER),
                            succ_int, "SUPERSEDES")
                        if old_succ is not None:
                            txn.delete_edge({"id": int(old_succ.id)})
                    except Exception:
                        logger.warning(
                            "create_property_version: DELETE old P->S edge failed "
                            "(non-fatal) old=%s succ=%s",
                            str(supersedes_id)[:12], str(superseded_by)[:12],
                            exc_info=True)
                old_merged = dict(old_props)
                old_merged["expired_at"] = now
                txn.upsert_node_as(
                    "old", LABEL_PROPERTY_VER, str(supersedes_id), props=old_merged
                )
                txn.upsert_edge_as(
                    "e_old_new", {"local": "old"}, {"local": "new"}, "SUPERSEDES",
                )
            if superseded_by and succ_int is not None:
                succ_node = self._db.get_node_by_key(
                    LABEL_PROPERTY_VER, str(superseded_by))
                if succ_node is not None:
                    txn.upsert_edge_as(
                        "e_new_succ", {"local": "new"},
                        {"labels": [LABEL_PROPERTY_VER], "key": str(superseded_by)},
                        "SUPERSEDES",
                    )
            txn.commit()
        except Exception:
            try:
                txn.rollback()
            except Exception:
                logger.warning("create_property_version: rollback failed", exc_info=True)
            raise
        return pid

    def get_latest_property_version(self, entity_id: str, attr_name: str) -> dict | None:
        """当前最新（未过期）版本：expired_at IS NULL + valid_from DESC 取第一。"""
        rows = self.execute_cypher(
            f"MATCH (p:{LABEL_PROPERTY_VER}) "
            "WHERE p.entity_id = $eid AND p.attr_name = $name "
            "AND (p.expired_at IS NULL) "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from DESC LIMIT 1",
            {"eid": str(entity_id), "name": attr_name},
        )
        return rows[0] if rows else None

    def get_property_versions(self, entity_id: str, attr_name: str) -> list[dict]:
        """(entity_id, attr_name) 全部版本（valid_from ASC 旧→新）。"""
        return self.execute_cypher(
            f"MATCH (p:{LABEL_PROPERTY_VER}) "
            "WHERE p.entity_id = $eid AND p.attr_name = $name "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from ASC",
            {"eid": str(entity_id), "name": attr_name},
        )

    def prune_property_versions(
        self, entity_id: str, attr_name: str,
        max_versions: int = PROPERTY_MAX_VERSIONS,
    ) -> int:
        """写时惰性裁剪（决策 5）：每 (entity_id, attr_name) 保留最近 N=8 版。"""
        versions = self.get_property_versions(entity_id, attr_name)
        if len(versions) <= max_versions:
            return 0
        removed = 0
        for v in versions[: len(versions) - max_versions]:
            vid = v.get("id", "")
            if not vid:
                continue
            try:
                self._locked_execute_gql(
                    f"MATCH (p:{LABEL_PROPERTY_VER} {{id: $id}}) DETACH DELETE p",
                    {"id": str(vid)},
                )
                removed += 1
            except Exception:
                logger.warning("prune_property_versions: delete failed for %s (non-fatal)",
                               str(vid)[:12])
        return removed

    def get_property_versions_for_entities(self, entity_ids: list[str]) -> list[dict]:
        """批量取多个实体的全部属性版本（P1-2 归一化前缀匹配，LIKE→CONTAINS
        前缀语义等价；复用 query_cypher 永不抛契约：失败 → []）。"""
        if not entity_ids:
            return []
        from core.entity_resolver import normalize_entity_name
        patterns: list[str] = []
        seen: set[str] = set()
        for eid in entity_ids:
            norm = normalize_entity_name(eid)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            # N6 词边界后置过滤：'apple' 命中 'Apple Inc' 但不命中 "Applebee's"
            patterns.append(
                f"(LOWER(p.entity_id) = '{norm}' "
                f"OR LOWER(p.entity_id) CONTAINS '{norm} ')"
            )
        if not patterns:
            return []
        where = " OR ".join(patterns)
        return self.query_cypher(
            f"MATCH (p:{LABEL_PROPERTY_VER}) "
            f"WHERE {where} "
            "RETURN p.id AS id, p.entity_id AS entity_id, p.attr_name AS attr_name, "
            "p.value AS value, p.valid_from AS valid_from, p.expired_at AS expired_at "
            "ORDER BY p.valid_from DESC"
        )

    def get_distinct_attr_names(self) -> list[str]:
        """全部属性名清单（PropertyVerNode.attr_name distinct，v5.50.0 P2）。"""
        rows = self.query_cypher(
            f"MATCH (p:{LABEL_PROPERTY_VER}) RETURN DISTINCT p.attr_name AS attr_name"
        )
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("attr_name", "") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    # ─── Hyperedge CRUD ─────────────────────────────────

    def create_hyperedge_node(self, hyperedge: dict) -> str:
        """创建 HyperedgeNode。"""
        hid = str(hyperedge.get("id", str(uuid.uuid4())))
        props = dict(hyperedge)
        props["id"] = hid
        self._locked_upsert_node(LABEL_HYPEREDGE, hid, props)
        return hid

    def link_hyperedge_member(self, hyperedge_id: str, episode_id: str) -> None:
        """HyperedgeNode -[:HYPEREDGE_MEMBER]-> EpisodeNode（幂等守卫防重复边）。"""
        self._ensure_edge(
            self._require_internal_id(hyperedge_id, LABEL_HYPEREDGE),
            self._require_internal_id(episode_id, LABEL_EPISODE),
            "HYPEREDGE_MEMBER",
        )

    def get_hyperedge_members(self, hyperedge_id: str) -> list[dict]:
        try:
            result = self._locked_execute_gql(
                f"MATCH (h:{LABEL_HYPEREDGE} {{id: $id}})-[:HYPEREDGE_MEMBER]->(e) "
                "RETURN e",
                {"id": str(hyperedge_id)},
            )
            return [self._flatten_row(r, "e") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def create_schema_node(self, schema: dict) -> str:
        """INSERT Schema 节点（:Conceptual 标签，阶段4-1 模式蒸馏产物）。

        pattern_keywords list → 空格连接串（CONTAINS 友好）；source_ids JSON 串。
        """
        sid = str(schema.get("id", str(uuid.uuid4())))
        props = dict(schema)
        props["id"] = sid
        props["pattern_keywords"] = " ".join(schema.get("pattern_keywords") or [])
        self._locked_upsert_node("Conceptual", sid, props)
        return sid

    def query_schema_nodes(self, terms: list[str], limit: int = 5) -> list[dict]:
        """按术语 OR CONTAINS 检索 Schema 节点（pattern_keywords/schema_name）。"""
        terms = [str(t) for t in (terms or []) if t]
        if not terms:
            return []
        try:
            conditions = " OR ".join(
                f"(s.pattern_keywords CONTAINS $t{i} OR s.schema_name CONTAINS $t{i})"
                for i in range(len(terms))
            )
            params = {f"t{i}": t for i, t in enumerate(terms)}
            result = self._locked_execute_gql(
                f"MATCH (s:Conceptual) WHERE {conditions} "
                f"RETURN s ORDER BY s.support DESC LIMIT {int(limit)}",
                params,
            )
            return [self._flatten_row(r, "s") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def get_hyperedges_by_node(self, node_id: str) -> list[dict]:
        try:
            result = self._locked_execute_gql(
                f"MATCH (h:{LABEL_HYPEREDGE})-[:HYPEREDGE_MEMBER]->"
                f"(e:{LABEL_EPISODE} {{id: $id}}) RETURN h",
                {"id": str(node_id)},
            )
            return [self._flatten_row(r, "h") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def get_hypergraph_neighbors(
        self, seed_ids: list[str], limit: int = 20
    ) -> dict[str, list[dict]]:
        """向量种子 → 共享超边成员扩散（1 跳）。返回 {seed_id: [{id, content,
        co_occurrence}]}。OverGraph 中文原生直读（零 b64）。空输入/异常 → {}。"""
        if not seed_ids:
            return {}
        result: dict[str, list[dict]] = {}
        for sid in seed_ids:
            rows = self.query_cypher(
                "MATCH (e:EpisodeNode {id: $sid})-[:HYPEREDGE_MEMBER]"
                "-(h:HyperedgeNode)-[:HYPEREDGE_MEMBER]-(e2:EpisodeNode) "
                "WHERE e2.id <> $sid "
                "RETURN DISTINCT e2.id AS id, e2.content AS content, "
                "count(h) AS co_occurrence "
                "ORDER BY co_occurrence DESC LIMIT $limit",
                {"sid": sid, "limit": int(limit)},
            )
            if not rows:
                continue
            neighbors: list[dict] = []
            for row in rows:
                nid = row.get("id", "")
                if not nid:
                    continue
                try:
                    cooc = int(row.get("co_occurrence", 0))
                except (TypeError, ValueError):
                    cooc = 0
                neighbors.append({
                    "id": str(nid),
                    "content": str(row.get("content", "") or ""),
                    "co_occurrence": cooc,
                })
            if neighbors:
                result[sid] = neighbors
        return result

    def get_communities_by_seeds(self, seed_ids: list[str]) -> list[dict]:
        """种子节点 → 所属社区反查（v5.41 社区扩召回）。"""
        if not seed_ids:
            return []
        rows = self.query_cypher(
            f"MATCH (c:{LABEL_COMMUNITY})-[:COMMUNITY_MEMBER]->(e:{LABEL_EPISODE}) "
            "WHERE e.id IN $ids "
            "RETURN c.id AS community_id, c.summary AS summary, e.id AS member_id",
            {"ids": list(seed_ids)},
        )
        if not rows:
            return []
        communities: dict[str, dict] = {}
        for row in rows:
            cid = row.get("community_id", "") or ""
            if not cid:
                continue
            entry = communities.setdefault(cid, {
                "community_id": cid,
                "summary": row.get("summary", "") or "",
                "member_ids": [],
            })
            mid = row.get("member_id", "") or ""
            if mid and mid not in entry["member_ids"]:
                entry["member_ids"].append(mid)
        return list(communities.values())

    def get_community_members(self, community_id: str, limit: int = 10) -> list[dict]:
        """按社区批量取成员（content/归档/fact_track/tau 一次取回）。"""
        if not community_id:
            return []
        rows = self.query_cypher(
            f"MATCH (c:{LABEL_COMMUNITY} {{id: $cid}})-[:COMMUNITY_MEMBER]->"
            f"(e:{LABEL_EPISODE}) "
            "RETURN e.id AS member_id, e.content AS content, "
            "e.archived AS archived, e.fact_track AS fact_track, "
            "e.tau_initial AS tau_value "
            "ORDER BY e.tau_initial DESC LIMIT $limit",
            {"cid": community_id, "limit": int(limit)},
        )
        members: list[dict] = []
        for row in rows:
            mid = row.get("member_id", "") or ""
            if not mid:
                continue
            members.append({
                "member_id": mid,
                "content": row.get("content", "") or "",
                "archived": row.get("archived", False),
                "fact_track": row.get("fact_track", "active") or "active",
                "tau_value": row.get("tau_value", 0.0) or 0.0,
            })
        return members

    # ─── Hebbian ───────────────────────────────────────

    def get_all_hebbian_connections(self) -> list[dict]:
        """全部 HEBBIAN_CONNECTION 边（weight 为 GQL 可见 props —— 边创建经
        SET r.weight 落 props，见翻译层 _rewrite_edges）。"""
        try:
            result = self._locked_execute_gql(
                "MATCH (a)-[r:HEBBIAN_CONNECTION]->(b) "
                "RETURN a.id AS src, b.id AS dst, r.weight AS weight"
            )
            return list((result or {}).get("rows", []))
        except Exception:
            return []

    def get_all_connections(self) -> dict[str, dict[str, float]]:
        """全部 Hebbian 连接 {src_id: {dst_id: weight}}（供 Hebbian 更新器）。"""
        conns: dict[str, dict[str, float]] = {}
        try:
            for row in self.get_all_hebbian_connections():
                src = row.get("src") or row.get("a.id")
                dst = row.get("dst") or row.get("b.id")
                if not src or not dst:
                    continue
                try:
                    w = float(row.get("weight") or row.get("r.weight") or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                conns.setdefault(str(src), {})[str(dst)] = w
        except Exception:
            pass
        return conns

    # ─── Session / Visual CRUD ─────────────────────────

    def ensure_session(self, session_id: str) -> None:
        """确保 SessionNode 存在（查询-插入两段式幂等，同 GraphLite）。"""
        try:
            result = self._locked_execute_gql(
                f"MATCH (s:{LABEL_SESSION} {{id: $id}}) RETURN s.id",
                {"id": str(session_id)},
            )
            if (result or {}).get("rows"):
                return
        except Exception:
            pass
        ts = int(time.time())
        self._locked_upsert_node(LABEL_SESSION, str(session_id),
                                 {"id": str(session_id), "created_at": ts,
                                  "last_seen": ts})

    def link_to_session(self, session_id: str, episode_id: str) -> None:
        """Link episode to session node（幂等守卫）。"""
        self._ensure_edge(
            self._require_internal_id(session_id, LABEL_SESSION),
            self._require_internal_id(episode_id, LABEL_EPISODE),
            "SESSION_MEMBER",
        )

    def get_session_memories(self, session_id: str, limit: int = 100) -> list[dict]:
        try:
            result = self._locked_execute_gql(
                f"MATCH (s:{LABEL_SESSION} {{id: $id}})-[:SESSION_MEMBER]->(e) "
                "RETURN e LIMIT $lim",
                {"id": str(session_id), "lim": int(limit)},
            )
            return [self._flatten_row(r, "e") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def get_or_create_session(self, session_id: str, metadata: str | None = None) -> str:
        """获取或创建 SessionNode，返回 session_id。"""
        self.ensure_session(session_id)
        if metadata:
            try:
                self._locked_execute_gql(
                    f"MATCH (s:{LABEL_SESSION} {{id: $id}}) SET s.metadata = $m",
                    {"id": str(session_id), "m": str(metadata)},
                )
            except Exception:
                logger.warning("get_or_create_session: metadata set failed (non-fatal)",
                               exc_info=True)
        return str(session_id)

    def link_session_member(self, session_node_id: str, episode_id: str) -> None:
        """Link episode to session node（幂等守卫）。"""
        self._ensure_edge(
            self._require_internal_id(session_node_id, LABEL_SESSION),
            self._require_internal_id(episode_id, LABEL_EPISODE),
            "SESSION_MEMBER",
        )

    def create_visual_node(self, node: dict) -> str:
        """INSERT VisualNode（embedding list 原生属性直存，零 b64）。"""
        vid = str(node.get("id", str(uuid.uuid4())))
        props = dict(node)
        props["id"] = vid
        self._locked_upsert_node(LABEL_VISUAL, vid, props)
        return vid

    def get_visual_node(self, visual_id: str) -> dict | None:
        try:
            view = self._db.get_node_by_key(LABEL_VISUAL, str(visual_id))
        except Exception:
            return None
        if view is None:
            return None
        return self._flatten_view(view)

    def get_visual_nodes(self, limit: int = 50) -> list[dict]:
        try:
            result = self._locked_execute_gql(
                f"MATCH (v:{LABEL_VISUAL}) RETURN v LIMIT {int(limit)}"
            )
            return [self._flatten_row(r, "v") for r in (result or {}).get("rows", [])]
        except Exception:
            return []

    def delete_namespace(self, namespace: str) -> int:
        """按命名空间删除：删除 SessionNode 及其 SESSION_MEMBER 关联的 EpisodeNode。

        返回删除的 EpisodeNode 数（DETACH DELETE 处理关联边）。
        """
        try:
            result = self._locked_execute_gql(
                f"MATCH (s:{LABEL_SESSION} {{id: $ns}})-[:SESSION_MEMBER]->"
                f"(e:{LABEL_EPISODE}) RETURN e",
                {"ns": str(namespace)},
            )
            ep_ids = [self._flatten_row(r, "e").get("id", "") for r in
                      (result or {}).get("rows", [])]
            ep_ids = [i for i in ep_ids if i]
        except Exception:
            return 0
        deleted = 0
        for eid in ep_ids:
            try:
                self._locked_execute_gql(
                    f"MATCH (e:{LABEL_EPISODE} {{id: $id}}) DETACH DELETE e",
                    {"id": eid},
                )
                deleted += 1
            except Exception:
                pass
        try:
            self._locked_execute_gql(
                f"MATCH (s:{LABEL_SESSION} {{id: $ns}}) DETACH DELETE s",
                {"ns": str(namespace)},
            )
        except Exception:
            pass
        return deleted

    # ─── Direct GQL（白名单翻译层）────────────────────────

    def execute_cypher(self, query: str, params: dict | None = None) -> list:
        """Execute GQL directly, return list of row dicts.

        熔断门控 + 不吞异常（P2-2 写路径熔断中立：只参与 allow_request 门控，
        不 record_success/failure）；不加 @with_retry —— 写操作不自动重试。
        无 RETURN 的 mutation 成功时返回 [{"status": "ok"}]（对齐 GraphLite
        INSERT/SET 的状态行 truthy 契约，供 _flush_hebbian_batch 等判定成功）。
        """
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpen("circuit breaker open, query rejected")
        return self._run_translated(query, params)

    @with_retry(
        max_attempts=2, base_delay=0.2, backoff=2.0,
        retryable_exceptions=_INFRA_EXCEPTIONS,
    )
    def _query_retryable(self, query: str, params: dict | None = None) -> list:
        """底层查询（熔断门控 + 重试）：成功返回 rows；open 状态返回 []。"""
        if not self.circuit_breaker.allow_request():
            self._local.last_infra_degraded = True
            return []
        try:
            rows = self._run_translated(query, params)
            self.circuit_breaker.record_success()
            self._local.last_infra_degraded = False
            return rows
        except _INFRA_EXCEPTIONS:
            raise
        except Exception:
            self._local.last_infra_degraded = False
            return []

    def query_cypher(self, query: str, params: dict | None = None) -> list:
        """Query GQL, return list of dicts. 永不抛异常契约（P0-2 关键设计决策）。

        - open 状态返回 []（静默降级；query_router 显式 is_open() 检查级联）
        - 基础设施错误（OverGraphError + 内置类）由 _query_retryable 重试，耗尽后
          统一计 1 次失败 → 返回 []；跳闸静默返回 []（保持永不抛契约）
        - 应用错误（翻译失败/语法错误）不计数、不重试，返回 []
        """
        try:
            return self._query_retryable(query, params)
        except _INFRA_EXCEPTIONS as e:
            try:
                self.circuit_breaker.record_failure(e)
            except CircuitBreakerOpen:
                pass
            self._local.last_infra_degraded = True
            return []
        except CircuitBreakerOpen:
            self._local.last_infra_degraded = True
            return []
        except Exception:
            self._local.last_infra_degraded = False
            return []

    def last_query_infra_degraded(self) -> bool:
        """最近一次 query_cypher 是否基础设施降级（thread-local）。"""
        return getattr(self._local, "last_infra_degraded", False)

    def _run_translated(self, query: str, params: dict | None = None) -> list:
        """翻译并执行：返回 rows（mutation 无 RETURN → [{"status": "ok"}]）。

        【读侧兼容】整节点行（`RETURN e` 的 NodeView/EdgeView dict）转 GraphLite
        兼容混合形态 —— 同时含 `props`（OverGraphStore._flatten_row）与
        `Node`/`Relationship` 包裹（GraphLiteStore._flatten_row，user_profile/
        dream_scheduler 的零改动消费方）→ 两套 flatten 均返回 props。
        【P0 契约】`RETURN x.*`（已重写为 `RETURN x`）→ 整节点 props 提升到行
        顶层 —— communities/hyperedges/system/gateway 消费方直接 row.get("id")
        零改动（R1 P0#3；GraphLite 原始 `RETURN x.*` 为嵌套 Node 形态，消费方
        按扁平 props 编写，OverGraph 侧补平该契约）。
        """
        ops = _translate_gql(query, params)
        rows: list = []
        star_return = re.search(r"\bRETURN\s+\w+\.\*", query) is not None
        for kind, payload in ops:
            if kind == "synth":
                rows.extend(payload)
            elif kind == "upsert_node":
                label, props = payload
                key = _node_key(props)
                self._locked_upsert_node(label, key, props)
                rows.append({"status": "ok"})
            else:
                gql, allow_full_scan = payload
                result = self._locked_execute_gql(gql, params)
                r = list((result or {}).get("rows") or [])
                rows.extend(
                    self._flatten_star_rows(r) if star_return else self._compat_rows(r)
                )
        if not rows and "RETURN" not in query:
            # 无 RETURN 的 mutation：状态行 truthy 契约（同 GraphLite INSERT/SET）
            return [{"status": "ok"}]
        return rows

    @classmethod
    def _compat_rows(cls, rows: list) -> list:
        """NodeView/EdgeView dict → GraphLite 兼容混合形态（读侧零改动）。"""

        def _wrap(v):
            if not isinstance(v, dict) or "props" not in v:
                return v
            props = v.get("props") or {}
            if "from_id" in v:  # EdgeView dict
                return {"Relationship": {"properties": props}, "props": props}
            if "labels" in v or "key" in v:  # NodeView dict
                return {"Node": {"properties": props}, "props": props,
                        "id": v.get("id"), "labels": v.get("labels"),
                        "key": v.get("key")}
            return v

        return [{k: _wrap(val) for k, val in row.items()} if isinstance(row, dict)
                else row for row in rows]

    @classmethod
    def _flatten_star_rows(cls, rows: list) -> list:
        """`RETURN x.*`（已重写为 `RETURN x`）→ 整节点 props 提升到行顶层。

        消费方契约（communities/hyperedges/system/gateway/hyperedge 直接
        row.get("id")/row["id"]，按扁平 props 编写）：每个属性为顶层列。
        多列形态 `RETURN h.*, member_ids` → h 的 props 与 member_ids 等其余
        列合并进同一行。（注意：GraphLite 原始 `RETURN x.*` 为嵌套 Node
        形态 —— 本方法补的是消费方契约，非 GraphLite SDK 原始形态。）
        """
        out: list = []
        for row in rows:
            if not isinstance(row, dict):
                out.append(row)
                continue
            flat: dict = {}
            merged = False
            for k, v in row.items():
                if isinstance(v, dict) and "props" in v:
                    merged = True
                    for pk, pv in (v.get("props") or {}).items():
                        flat.setdefault(pk, pv)
                else:
                    flat[k] = v
            out.append(flat if merged else row)
        return out

    # ─── 向量方法（v6.0.0 HNSW 主通道）────────────────────

    def vector_search_dense(
        self,
        k: int,
        query_vec,
        label_filter: list[str] | None = None,
        scope_start_node_id: int | None = None,
        scope_max_depth: int | None = None,
        scope_direction: str | None = None,
        scope_at_epoch: int | None = None,
    ) -> list[tuple[str, float]]:
        """OverGraph HNSW dense 检索 → [(ep_id, cosine_score)]。

        R1 定标：score 为 cosine s∈[-1,1]（无 L2 可用）——由调用方（adapter）按
        D5 映射 d=1/s-1。scope 强化（D9）Phase 1 透传：scope_start_node_id 需
        引擎内部 ID(int)，经 get_node_internal_id 转换。阶段3（v6.0.0 图作用域
        检索）补 scope_direction/scope_at_epoch 透传（D7/D8/D9；direction 语义
        见 vector_search_scoped PoC 定标注释）。
        """
        labels = list(label_filter) if label_filter else [LABEL_EPISODE]
        kwargs: dict = {}
        ef_search = getattr(self.config, "ef_search", None)
        if ef_search is not None:
            kwargs["ef_search"] = int(ef_search)
        if scope_start_node_id is not None:
            kwargs["scope_start_node_id"] = int(scope_start_node_id)
            kwargs["scope_max_depth"] = int(scope_max_depth or 1)
            if scope_direction is not None:
                kwargs["scope_direction"] = scope_direction
            if scope_at_epoch is not None:
                kwargs["scope_at_epoch"] = int(scope_at_epoch)
        with self._session_lock:
            assert self._db is not None
            hits = self._db.vector_search(
                "dense", int(k),
                dense_query=_as_float32(query_vec),
                label_filter={"labels": labels},
                **kwargs,
            )
            if not hits:
                return []
            views = self._db.get_nodes([int(h.node_id) for h in hits])
        keys = {int(v.id): str(v.key) for v in views if v is not None}
        return [(keys.get(int(h.node_id), str(h.node_id)), float(h.score))
                for h in hits]

    # ─── 图作用域检索（阶段3 D5-D10）──────────────────────

    def vector_search_scoped(
        self,
        seed_episode_id: str,
        k: int,
        query_vec,
        max_depth: int = 2,
        at_ts: float | None = None,
    ) -> list[tuple[str, float]]:
        """以种子 Episode 为作用域起点的 HNSW 检索 → [(ep_id, cosine_score)]。

        【R1 PoC 定标（2026-08-19, overgraph 0.17.0）】——scope 语义实证：
        - direction 每跳独立：outgoing=沿离开节点的边，incoming=沿进入节点的边，
          both=每跳可入可出（非法值引擎抛 ValueError）。
        - HYPEREDGE_MEMBER 为 hyperedge→episode 单向边：co-member（ep2 经共享
          超边 h：ep1←h 入跳 + h→ep2 出跳）**必须 scope_direction="both"**，
          "outgoing" 恒不命中（D7 实证成立）→ 本方法硬编码 both。
        - scope_max_depth 计跳数：depth=1(both) 仅直达邻居，depth=2 才含
          共享超边 co-member（实测 ep2 在 depth=2 命中、depth=1 不命中）。
        - scope_at_epoch=int(at_ts*1000)（D8/D9：SHM 秒 → 引擎毫秒）接受且无
          副作用（SHM 边无时序）→ 仅作时间锚透传，created_at<=at_ts 节点过滤
          由上层负责，不替代。
        - 降级安全：种子不存在 → 返回 []（不抛）；孤立种子 → 仅自身。
        """
        iid = self.get_node_internal_id(str(seed_episode_id))
        if iid is None:
            return []
        kwargs: dict = {}
        ef_search = getattr(self.config, "ef_search", None)
        if ef_search is not None:
            kwargs["ef_search"] = int(ef_search)
        if at_ts is not None:
            kwargs["scope_at_epoch"] = int(float(at_ts) * 1000)
        with self._session_lock:
            assert self._db is not None
            hits = self._db.vector_search(
                "dense", int(k),
                dense_query=_as_float32(query_vec),
                label_filter={"labels": [LABEL_EPISODE]},
                scope_start_node_id=iid,
                scope_max_depth=int(max_depth),
                scope_direction="both",
                **kwargs,
            )
            if not hits:
                return []
            views = self._db.get_nodes([int(h.node_id) for h in hits])
        keys = {int(v.id): str(v.key) for v in views if v is not None}
        return [(keys.get(int(h.node_id), str(h.node_id)), float(h.score))
                for h in hits]

    def batch_upsert_embeddings(self, nodes: list[dict]) -> int:
        """批量写 EpisodeNode.dense_vector（读-合并-批量 upsert，不破坏 props）。

        OverGraph 无「仅更新向量」API（GQL SET e.dense_vector 写的是同名 props
        非一等字段）→ 只能 typed upsert_node(dense_vector=)，而它整体替换 props，
        故先读回现有 props 合并。nodes: [{"node_id": str, "embedding": vec}]。
        """
        if not nodes:
            return 0
        items: list[dict] = []
        with self._session_lock:
            assert self._db is not None
            for n in nodes:
                nid = str(n.get("node_id") or n.get("id") or "")
                if not nid:
                    continue
                view = self._db.get_node_by_key(LABEL_EPISODE, nid)
                props = dict(view.props) if view is not None else {"id": nid}
                props["id"] = nid
                items.append({
                    "labels": [LABEL_EPISODE],
                    "key": nid,
                    "props": props,
                    "dense_vector": _as_float32(n.get("embedding")),
                })
            if not items:
                return 0
            self._db.batch_upsert_nodes(items)
            return len(items)

    def get_episode_keys(self, internal_ids: list[int] | None = None) -> list[str]:
        """内部 ID → elementKey 转换（vector_search 返回内部 ID，adapter 反查）。

        internal_ids=None 时返回全部 EpisodeNode elementKey（rebuild/统计用）。
        """
        with self._session_lock:
            assert self._db is not None
            if internal_ids is None:
                views = self._db.get_nodes_by_labels(LABEL_EPISODE)
                return [str(v.key) for v in views]
            views = self._db.get_nodes([int(i) for i in internal_ids])
            return [str(v.key) for v in views if v is not None]

    def get_node_internal_id(self, node_id: str, label: str = LABEL_EPISODE) -> int | None:
        """elementKey → 引擎内部 ID(int)（scope_start_node_id 需 int，D9）。"""
        try:
            view = self._db.get_node_by_key(label, str(node_id))
        except Exception:
            return None
        return int(view.id) if view is not None else None

    # ─── Helpers ──────────────────────────────────────

    def _require_internal_id(self, node_id: str, label: str = LABEL_EPISODE) -> int:
        """取内部 ID；节点不存在抛 OverGraphError（调用方守卫兜底）。"""
        iid = self.get_node_internal_id(str(node_id), label)
        if iid is None:
            raise OverGraphError(f"node not found: {label} {str(node_id)[:12]}")
        return iid

    def _ensure_edge(self, from_id: int, to_id: int, label: str,
                     props: dict | None = None, weight: float | None = None) -> None:
        """边幂等创建：MATCH-exists 守卫 + upsert（GraphLite INSERT 语义 + 防重复）。"""
        with self._session_lock:
            assert self._db is not None
            try:
                if self._db.get_edge_by_triple(from_id, to_id, label) is not None:
                    return
            except Exception:
                pass  # 查询失败 → 走创建路径
            self._db.upsert_edge(
                from_id, to_id, label,
                props=props or {},
                weight=weight if weight is not None else 1.0,
            )

    @staticmethod
    def _flatten_view(view) -> dict:
        """NodeView（或 txn dict）→ props dict（读侧零改动：id 已落 props）。"""
        if view is None:
            return {}
        if isinstance(view, dict):
            props = view.get("props") or {}
            return dict(props)
        props = getattr(view, "props", None) or {}
        return dict(props)

    @staticmethod
    def _flatten_row(row: dict, label: str = "") -> dict:
        """GQL row → dict（兼容 NodeView dict / EdgeView dict / 别名列）。

        - `RETURN e` → row['e'] = NodeView dict（props 为整包）→ 展开 props
        - `RETURN e.id AS id` → 别名列原样透传
        - label 指定时（如 flatten(row, "e")）仅返回该节点的 props（对齐
          GraphLiteStore._flatten_row(row, label) 语义）
        """
        if not isinstance(row, dict):
            return row
        result: dict = {}
        for k, v in row.items():
            if isinstance(v, dict) and "props" in v:
                flat = dict(v.get("props") or {})
                if label and k == label:
                    return flat
                result[k] = flat
            else:
                result[k] = v
        return result

    # ─── Lifecycle ────────────────────────────────────

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
