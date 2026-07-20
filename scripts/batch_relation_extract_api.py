"""
SHM v5.8.1 — 旧数据批量关系抽取（通过 API 调用）
=================================================
1. 通过 SHM 的 /query 接口读取所有 EpisodeNode
2. 本地运行关系抽取
3. 通过 SHM 的 episodes 写入接口（带关系抽取）重新处理
   只处理能提取出关系的数据
"""

import json, time, logging, sys, re
from typing import List, Dict, Any, Set, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(levelname)s|%(message)s")
logger = logging.getLogger("batch_relation_api")

API_BASE = "http://127.0.0.1:8000"

def api_post(path: str, data: dict = None) -> dict:
    """调用 SHM API"""
    url = f"{API_BASE}{path}"
    body = json.dumps(data or {}).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"},
                  method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        logger.error(f"API 请求失败 {path}: {e}")
        return {}
    except json.JSONDecodeError:
        return {}

def api_get(path: str) -> dict:
    """GET 请求"""
    url = f"{API_BASE}{path}"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return {}

# ──────────────────────────────────────────────────────────────
# 1. 读取全部 EpisodeNode
# ──────────────────────────────────────────────────────────────
def get_all_episodes() -> List[Dict[str, Any]]:
    """通过查询接口获取所有 episode 内容"""
    result = api_post("/query", {
        "query": "MATCH (e:EpisodeNode) RETURN e.id, e.content, e.title ORDER BY e.created_at"
    })
    rows = result.get("rows", result.get("results", []))
    episodes = []
    for row in rows:
        if isinstance(row, list) and len(row) >= 2:
            episodes.append({
                "id": str(row[0]) if row[0] else "",
                "content": str(row[1]) if row[1] else "",
                "title": str(row[2]) if len(row) > 2 and row[2] else "",
            })
        elif isinstance(row, dict):
            episodes.append(row)
    logger.info(f"读取到 {len(episodes)} 条 EpisodeNode")
    return episodes

# ──────────────────────────────────────────────────────────────
# 2. 关系抽取（直接本地跑，不依赖 SHM 进程内的 module）
# ──────────────────────────────────────────────────────────────
# 直接复制 RELATION_PATTERNS 和抽取逻辑
# （避免 import kuzu 冲突）

RELATION_PATTERNS = [
    # X founded Y
    ("FOUNDED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+founded\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X is (the) CEO/president/... of Y
    ("LEADS",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+is\s+(?:the\s+)?'
     r'(?:CEO|president|chairman|chairwoman|head|leader|founder|director|manager)'
     r'(?:\s+and\s+(?:CEO|president|chairman|CEO|CTO|CFO))?\s+of\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X acquired Y
    ("ACQUIRED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:acquired|bought|purchased)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     r'(?:for\s+)?(\$?[\d,.]+\s*(?:billion|million|B|M)?)'),
    # X released / launched Y
    ("RELEASED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:released|launched|introduced|unveiled|announced)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X is located in Y
    ("LOCATED_IN",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:is\s+)?(?:located|based|headquartered)\s+(?:in|at)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X works (for|at) Y
    ("WORKS_AT",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:works\s+(?:for|at)|joined|employed\s+(?:at|by))\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X created / developed Y
    ("CREATED",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:created|developed|built|designed|wrote)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
    # X invested (in) Y
    ("INVESTED_IN",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:invested\s+in|led\s+(?:a\s+)?\w*\s*round\s+(?:for|in)|funded)\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b',
     r'(?:\$?([\d,.]+)\s*(?:billion|million|B|M)?)'),
    # X partnered (with) Y
    ("PARTNERED_WITH",
     r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+'
     r'(?:partnered\s+(?:with|on)|teamed\s+up\s+(?:with|on)|collaborated\s+(?:with|on))\s+'
     r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b', None),
]

def extract(text: str) -> List[Tuple[str, str, str, Dict]]:
    """从文本中提取关系三元组，返回 [(subject, relation, obj, attrs), ...]"""
    triples = []
    seen = set()
    for rel_type, pattern_str, attr_pattern in RELATION_PATTERNS:
        pattern = re.compile(pattern_str)
        attr_comp = re.compile(attr_pattern) if attr_pattern else None
        for m in pattern.finditer(text):
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            if len(subj) < 2 or len(obj) < 2:
                continue
            if subj.lower() == obj.lower():
                continue
            attrs = {}
            if attr_comp:
                am = attr_comp.search(text)
                if am and am.lastindex and am.lastindex >= 1:
                    val = am.group(1).strip()
                    if val:
                        attrs["value"] = val
            key = (rel_type, subj.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append((subj, rel_type, obj, attrs))
    return triples

# ──────────────────────────────────────────────────────────────
# 3. 写入关系（通过 episodes API，因为 POST 会自动建实体+关系）
# ──────────────────────────────────────────────────────────────
def write_triple(subj: str, rel: str, obj: str, attrs: dict) -> bool:
    """写入一条关系边"""
    # 构造一段能触发关系抽取的文本
    trigger_texts = {
        "FOUNDED": f"{subj} founded {obj}.",
        "LEADS": f"{subj} is the CEO of {obj}.",
        "ACQUIRED": f"{subj} acquired {obj}.",
        "RELEASED": f"{subj} released {obj}.",
        "LOCATED_IN": f"{subj} is located in {obj}.",
        "WORKS_AT": f"{subj} works at {obj}.",
        "CREATED": f"{subj} created {obj}.",
        "INVESTED_IN": f"{subj} invested in {obj}.",
        "PARTNERED_WITH": f"{subj} partnered with {obj}.",
    }
    content = trigger_texts.get(rel, f"{subj} {rel.lower().replace('_',' ')} {obj}.")
    
    # 用 source=batch 标记，不重复写同一条
    resp = api_post("/memories/episodes", {
        "content": content,
        "source": "batch_relation_extract",
    })
    return resp.get("status") in ("created", "ok", "updated") or "id" in resp

def write_triples_direct(triples: list) -> dict:
    """直接通过 /batch/relations 写入（如果有该端点的话）"""
    pass  # 没有现成的批量端点，用单个写入

# ──────────────────────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────────────────────
def main():
    start = time.time()
    
    logger.info("=" * 50)
    logger.info("SHM 旧数据批量关系抽取 (API版)")
    logger.info("=" * 50)
    
    # Step 1: 读取
    logger.info("[Step 1] 读取旧数据...")
    episodes = get_all_episodes()
    if not episodes:
        logger.warning("没有发现 EpisodeNode")
        return
    
    # Step 2: 抽取
    logger.info("[Step 2] 关系抽取...")
    all_triples = []
    seen: Set[tuple] = set()
    for i, ep in enumerate(episodes):
        text = ep.get("content", "") or ep.get("title", "")
        if not text or len(text) < 10:
            continue
        triples = extract(text)
        for t in triples:
            key = (t[1], t[0].lower(), t[2].lower())
            if key not in seen:
                seen.add(key)
                all_triples.append(t)
        if (i + 1) % 200 == 0:
            logger.info(f"  进度: {i+1}/{len(episodes)} 条, 发现 {len(all_triples)} 条语义边")
    
    logger.info(f"  → 共发现 {len(all_triples)} 条语义关系")
    
    if not all_triples:
        logger.info("未发现任何可抽取的语义关系（旧数据中缺少匹配模式的内容）")
        return
    
    # Step 3: 展示
    logger.info("\n发现的关系:")
    for t in all_triples[:20]:
        attrs_str = f" [{', '.join(f'{k}={v}' for k,v in t[3].items())}]" if t[3] else ""
        logger.info(f"  {t[0]:25s} ──{t[1]:12s}──> {t[2]}{attrs_str}")
    
    if len(all_triples) > 20:
        logger.info(f"  ... 共 {len(all_triples)} 条")
    
    # Step 4: 写入
    logger.info(f"\n[Step 3] 写入 {len(all_triples)} 条关系到 Kuzu...")
    written = 0
    errors = 0
    for i, (subj, rel, obj, attrs) in enumerate(all_triples):
        try:
            ok = write_triple(subj, rel, obj, attrs)
            if ok:
                written += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"  写入失败: {subj}-[{rel}]->{obj}: {e}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"  写入进度: {i+1}/{len(all_triples)} (成功:{written} 失败:{errors})")
            time.sleep(0.5)  # 避免压垮 API
    
    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"完成! 耗时: {elapsed:.1f}s")
    logger.info(f"  读取: {len(episodes)} 条 EpisodeNode")
    logger.info(f"  发现: {len(all_triples)} 条语义边")
    logger.info(f"  写入: {written} 条 (失败: {errors})")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
