"""
SHM v5.8.1 — 旧数据批量关系抽取脚本
======================================
从所有 EpisodeNode 提取语义关系，写入 Kuzu 图数据库。

用法: cd /home/admin/shm && source .venv/bin/activate && python scripts/batch_relation_extract.py
"""

import sys, os, re, json, time, logging
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s|%(message)s")
logger = logging.getLogger("batch_relation")

# ──────────────────────────────────────────────────────────────
# 1. 连接 Kuzu
# ──────────────────────────────────────────────────────────────
def get_all_episodes(db_path: str = "data/shm_kuzu_db") -> List[Dict[str, Any]]:
    """从 Kuzu 读取所有 EpisodeNode 的内容"""
    import kuzu
    
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    rows = conn.execute(
        "MATCH (e:EpisodeNode) RETURN e.id, e.content, e.title ORDER BY e.created_at"
    ).get_as_pl().to_dicts() if hasattr(conn.execute(
        "MATCH (e:EpisodeNode) RETURN e.id LIMIT 1"
    ), 'get_as_pl') else _fallback_query(conn)
    
    # fallback for older kuzu
    episodes = []
    result = conn.execute("MATCH (e:EpisodeNode) RETURN e.id, e.content, e.title, e.created_at ORDER BY e.created_at")
    while result.has_next():
        row = result.get_next()
        episodes.append({
            "id": str(row[0]),
            "content": str(row[1]) if row[1] else "",
            "title": str(row[2]) if row[2] else "",
            "created_at": str(row[3]) if row[3] else "",
        })
    
    conn.close()
    db.close()
    logger.info(f"从 Kuzu 读取 {len(episodes)} 条 EpisodeNode")
    return episodes

def _fallback_query(conn):
    """fallback 读取方法"""
    episodes = []
    result = conn.execute("MATCH (e:EpisodeNode) RETURN e.id, e.content, e.title, e.created_at ORDER BY e.created_at")
    while result.has_next():
        row = result.get_next()
        episodes.append({
            "id": str(row[0]),
            "content": str(row[1]) if row[1] else "",
            "title": str(row[2]) if row[2] else "",
            "created_at": str(row[3]) if row[3] else "",
        })
    return episodes

# ──────────────────────────────────────────────────────────────
# 2. 关系抽取（复用 core/relation_extractor.py 的逻辑但独立运行）
# ──────────────────────────────────────────────────────────────
from core.relation_extractor import RelationExtractor, RelationTriple

def extract_relations_batch(episodes: List[Dict]) -> List[RelationTriple]:
    """批量抽取关系"""
    extractor = RelationExtractor()
    all_triples: List[RelationTriple] = []
    seen: Set[tuple] = set()
    
    for i, ep in enumerate(episodes):
        text = ep.get("content", "") or ep.get("title", "")
        if not text or len(text) < 10:
            continue
        
        triples = extractor.extract(text)
        for t in triples:
            key = (t.relation, t.subject.lower(), t.obj.lower())
            if key not in seen:
                seen.add(key)
                all_triples.append(t)
        
        if (i + 1) % 200 == 0:
            logger.info(f"  进度: {i+1}/{len(episodes)}, 发现 {len(all_triples)} 条语义边")
    
    return all_triples

# ──────────────────────────────────────────────────────────────
# 3. 写入 Kuzu 图数据库
# ──────────────────────────────────────────────────────────────
def write_triples(db_path: str, triples: List[RelationTriple]) -> Dict[str, int]:
    """将抽取的关系三元组写入 Kuzu 图数据库"""
    import kuzu
    
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    stats = {"created": 0, "skipped": 0, "errors": 0}
    
    for i, t in enumerate(triples):
        try:
            # 创建/匹配源节点和目标节点（作为 OntologyEntity）
            # 先检查节点是否存在
            check = conn.execute(
                "MATCH (n:OntologyEntity) WHERE n.name = $name RETURN n.name LIMIT 1",
                {"name": t.subject}
            )
            subj_exists = check.has_next()
            
            check2 = conn.execute(
                "MATCH (n:OntologyEntity) WHERE n.name = $name RETURN n.name LIMIT 1",
                {"name": t.obj}
            )
            obj_exists = check2.has_next()
            
            if not subj_exists:
                conn.execute(
                    "MERGE (n:OntologyEntity {name: $name}) ON CREATE SET n.name = $name, "
                    "n.type = 'Concept', n.created_at = CAST(current_timestamp() AS DOUBLE)",
                    {"name": t.subject}
                )
            
            if not obj_exists:
                conn.execute(
                    "MERGE (n:OntologyEntity {name: $name}) ON CREATE SET n.name = $name, "
                    "n.type = 'Concept', n.created_at = CAST(current_timestamp() AS DOUBLE)",
                    {"name": t.obj}
                )
            
            # 写入 RELATES_TO 边（带语义 relation 属性）
            attrs = {"relation": t.relation, "confidence": t.confidence}
            if t.attributes:
                attrs.update(t.attributes)
            
            attr_str = ", ".join(f"{k}: ${k}" for k in attrs.keys())
            params = {**attrs, "subj": t.subject, "obj": t.obj}
            
            conn.execute(
                f"MATCH (a:OntologyEntity {{name: $subj}}), "
                f"(b:OntologyEntity {{name: $obj}}) "
                f"MERGE (a)-[r:RELATES_TO]->(b) "
                f"ON CREATE SET {attr_str} "
                f"ON MATCH SET r.confidence = CASE WHEN r.confidence IS NOT NULL "
                f"THEN (r.confidence + $confidence) / 2 ELSE $confidence END, "
                f"r.relation = $relation",
                params
            )
            stats["created"] += 1
            
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                logger.warning(f"  写入失败 ({t.subject}-[{t.relation}]->{t.obj}): {e}")
        
        if (i + 1) % 50 == 0:
            logger.info(f"  写入进度: {i+1}/{len(triples)}")
    
    conn.close()
    db.close()
    return stats

# ──────────────────────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────────────────────
def main():
    start = time.time()
    db_path = "data/shm_kuzu_db"
    
    logger.info("=" * 50)
    logger.info("SHM 批量关系抽取")
    logger.info(f"数据库: {db_path}")
    logger.info("=" * 50)
    
    # Step 1: 读取
    logger.info("[Step 1/3] 读取旧数据...")
    episodes = get_all_episodes(db_path)
    logger.info(f"  → {len(episodes)} 条 EpisodeNode")
    
    if not episodes:
        logger.warning("没有找到任何 EpisodeNode，跳过")
        return
    
    # Step 2: 抽取
    logger.info("[Step 2/3] 关系抽取...")
    triples = extract_relations_batch(episodes)
    logger.info(f"  → 发现 {len(triples)} 条语义边")
    
    if not triples:
        logger.info("没有发现任何语义关系（数据中无可匹配的模式）")
        return
    
    # Step 3: 写入
    logger.info("[Step 3/3] 写入 Kuzu...")
    stats = write_triples(db_path, triples)
    
    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"完成! 耗时: {elapsed:.1f}s")
    logger.info(f"  读取: {len(episodes)} 条 EpisodeNode")
    logger.info(f"  发现: {len(triples)} 条语义边")
    logger.info(f"  写入: {stats['created']} 条 (错误: {stats['errors']})")
    logger.info("=" * 50)
    
    # 展示
    if triples:
        logger.info("抽取的关系示例:")
        for t in triples[:15]:
            attrs_str = f" [{', '.join(f'{k}={v}' for k,v in t.attributes.items())}]" if t.attributes else ""
            logger.info(f"  {t.subject:25s} ──{t.relation:12s}──> {t.obj}{attrs_str}")

if __name__ == "__main__":
    main()
