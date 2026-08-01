"""
SHM v5.8.1 — OSINT 数据结构化关系抽取
=======================================
从域名/URL/IP 数据中提取语义关系，写入 GraphLite 图数据库。

抽取的关系类型:
  SUBDOMAIN_OF  — ns2.baidu.com → baidu.com
  SAME_ORG      — baidu.com → qq.com (同一公司旗下)
  URL_ON        — https://news.qq.com/xx → news.qq.com
  IN_RANGE      — 59.82.122.15 → 59.82.122.0/24
  SAME_RANGE    — IPs 在同一个 /24 段
  OWNED_BY      — baidu.com → Baidu Inc (组织推断)

用法: source .venv/bin/activate && python scripts/batch_osint_relations.py
"""

import sys, os, json, time, re, logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
from urllib.parse import urlparse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s|%(message)s")
logger = logging.getLogger("batch_osint")

API_BASE = "http://127.0.0.1:8000"

# 已知中国互联网公司域名映射
ORG_MAP: Dict[str, str] = {
    # Baidu
    "baidu.com": "Baidu",
    "baidustatic.com": "Baidu",
    "bdstatic.com": "Baidu",
    # Tencent
    "qq.com": "Tencent",
    "gtimg.com": "Tencent",
    "tencent.com": "Tencent",
    "tcdn.qq.com": "Tencent",
    # Alibaba
    "taobao.com": "Alibaba",
    "alibaba.com": "Alibaba",
    "alicdn.com": "Alibaba",
    "tmall.com": "Alibaba",
    "alipay.com": "Alibaba",
    "aliyun.com": "Alibaba",
    "1688.com": "Alibaba",
    "alibaba-inc.com": "Alibaba",
    "alibabagroup.com": "Alibaba",
    "yikuaida.com": "Alibaba",
    "etao.com": "Alibaba",
    # Alibaba Cloud
    "aliyuncs.com": "AlibabaCloud",
    # Meituan
    "meituan.com": "Meituan",
    "dianping.com": "Meituan",
    # JD
    "jd.com": "JD",
    "360buyimg.com": "JD",
    # ByteDance
    "douyin.com": "ByteDance",
    "tiktok.com": "ByteDance",
    "toutiao.com": "ByteDance",
    "pstatp.com": "ByteDance",
    # Huawei
    "huawei.com": "Huawei",
    "hicloud.com": "Huawei",
    # Xiaomi
    "xiaomi.com": "Xiaomi",
    "mi.com": "Xiaomi",
}

# ──────────────────────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────────────────────
def fetch_episodes() -> List[Dict[str, str]]:
    """通过 API 读取所有 EpisodeNode"""
    req = urllib.request.Request(
        f"{API_BASE}/query",
        data=json.dumps({"query": "MATCH (e:EpisodeNode) RETURN e.id, e.content"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read())

    rows = d.get("rows", [])
    episodes = []
    for row in rows:
        eid = str(row[0]) if len(row) > 0 else ""
        content = str(row[1]) if len(row) > 1 else ""
        episodes.append({"id": eid, "content": content})
    return episodes

# ──────────────────────────────────────────────────────────────
# 2. 关系抽取
# ──────────────────────────────────────────────────────────────
def extract_domain_relations(episodes: List[Dict]) -> List[Tuple[str, str, str, Dict]]:
    """从域名数据提取 SUBDOMAIN_OF 和 SAME_ORG 关系"""
    triples: List[Tuple[str, str, str, Dict]] = []
    seen: Set[tuple] = set()

    for ep in episodes:
        content = ep.get("content", "")
        if not content.startswith("osint|"):
            continue
        parts = content.split("|")
        if len(parts) < 3:
            continue

        data_type = parts[1]
        value = "|".join(parts[2:])  # 值可能含 |

        if data_type == "domains":
            # SUBDOMAIN_OF: 提取子域→父域关系
            domain_parts = value.split(".")
            if len(domain_parts) >= 3:
                sub = value
                parent = ".".join(domain_parts[-2:])  # 二级域名
                key = ("SUBDOMAIN_OF", sub.lower(), parent.lower())
                if key not in seen:
                    seen.add(key)
                    triples.append((sub, "SUBDOMAIN_OF", parent, {}))

                # 如果域名在 ORG_MAP 中
                if parent in ORG_MAP:
                    org = ORG_MAP[parent]
                    key2 = ("OWNED_BY", parent.lower(), org.lower())
                    if key2 not in seen:
                        seen.add(key2)
                        triples.append((parent, "OWNED_BY", org, {}))

                    # 子域也归同一组织
                    key3 = ("OWNED_BY", sub.lower(), org.lower())
                    if key3 not in seen:
                        seen.add(key3)
                        triples.append((sub, "OWNED_BY", org, {}))

            elif len(domain_parts) == 2:
                parent = value
                if parent in ORG_MAP:
                    org = ORG_MAP[parent]
                    key = ("OWNED_BY", parent.lower(), org.lower())
                    if key not in seen:
                        seen.add(key)
                        triples.append((parent, "OWNED_BY", org, {}))

        elif data_type == "urls":
            # URL_ON: URL → 主机
            try:
                parsed = urlparse(value)
                host = parsed.netloc
                if host:
                    key = ("URL_ON", content.lower(), host.lower())
                    if key not in seen:
                        seen.add(key)
                        triples.append((content, "URL_ON", host, {}))

                    # 如果 URL 主机在 ORG_MAP 中
                    host_domain = ".".join(host.split(".")[-2:]) if len(host.split(".")) >= 2 else ""
                    if host_domain in ORG_MAP:
                        org = ORG_MAP[host_domain]
                        key2 = ("OWNED_BY", host.lower(), org.lower())
                        if key2 not in seen:
                            seen.add(key2)
                            triples.append((host, "OWNED_BY", org, {}))

                    # 子域关系
                    host_parts = host.split(".")
                    if len(host_parts) >= 3:
                        parent = ".".join(host_parts[-2:])
                        key3 = ("SUBDOMAIN_OF", host, parent)
                        if key3 not in seen:
                            seen.add(key3)
                            triples.append((host, "SUBDOMAIN_OF", parent, {}))
            except Exception:
                pass

        elif data_type == "ips":
            # IN_RANGE: IP → /24 子网
            ip_parts = value.split(".")
            if len(ip_parts) >= 3:
                subnet = ".".join(ip_parts[:3]) + ".0/24"
                key = ("IN_RANGE", value, subnet)
                if key not in seen:
                    seen.add(key)
                    triples.append((value, "IN_RANGE", subnet, {}))

    return triples

def extract_org_relations(triples: List[Tuple]) -> List[Tuple[str, str, str, Dict]]:
    """从已有三组元组中提取 SAME_ORG 关系（同一组织的域名互连）"""
    extra = []
    seen: Set[tuple] = set()

    # 收集每个组织下的域名
    org_domains: Dict[str, Set[str]] = defaultdict(set)
    for subj, rel, obj, attrs in triples:
        if rel == "OWNED_BY":
            org_domains[obj].add(subj)

    for org, domains in org_domains.items():
        domain_list = list(domains)
        for i in range(len(domain_list)):
            for j in range(i + 1, min(i + 3, len(domain_list))):
                a, b = domain_list[i], domain_list[j]
                key = ("SAME_ORG", a.lower(), b.lower())
                if key not in seen:
                    seen.add(key)
                    extra.append((a, "SAME_ORG", b, {"organization": org}))

    return extra

# ──────────────────────────────────────────────────────────────
# 3. 写入 GraphLite
# ──────────────────────────────────────────────────────────────
def write_to_graph(triples: List[Tuple]) -> Dict[str, int]:
    """通过 API 写入关系边（使用 episodes 触发自动建实体+边）"""
    stats = {"written": 0, "skipped": 0, "errors": 0}

    for i, (subj, rel, obj, attrs) in enumerate(triples):
        if rel == "OWNED_BY":
            # organization→org 的名称
            content = f"{subj} is owned by {obj}."
        elif rel in ("SUBDOMAIN_OF", "URL_ON"):
            content = f"{subj} {rel.replace('_',' ').lower()} {obj}."
        elif rel == "IN_RANGE":
            content = f"{subj} is in range {obj}."
        elif rel == "SAME_ORG":
            org = attrs.get("organization", "same organization")
            content = f"{subj} and {obj} are both part of {org}."
        else:
            content = f"{subj} {rel.lower()} {obj}."

        try:
            req = urllib.request.Request(
                f"{API_BASE}/memories/episodes",
                data=json.dumps({"content": content, "source": "batch_osint_relations"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read())
                if resp_data.get("status") in ("created", "ok", "updated") or "episode_id" in resp_data:
                    stats["written"] += 1
                else:
                    stats["skipped"] += 1
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                logger.warning(f"  写入失败 {subj}─[{rel}]─>{obj}: {str(e)[:50]}")

        if (i + 1) % 50 == 0:
            logger.info(f"    写入进度: {i+1}/{len(triples)} (成功:{stats['written']})")
            time.sleep(0.3)

    return stats

# ──────────────────────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────────────────────
def main():
    start = time.time()

    logger.info("=" * 50)
    logger.info("SHM 旧数据 OSINT 结构化关系抽取")
    logger.info("=" * 50)

    # Step 1: 读取
    logger.info("[Step 1/4] 读取旧数据...")
    episodes = fetch_episodes()
    logger.info(f"  → {len(episodes)} 条 EpisodeNode")

    if not episodes:
        logger.error("没有找到数据")
        return

    # Step 2: 抽取
    logger.info("[Step 2/4] 抽取语义关系...")
    triples = extract_domain_relations(episodes)
    logger.info(f"  → 基础关系: {len(triples)} 条")

    extra = extract_org_relations(triples)
    all_triples = triples + extra
    logger.info(f"  → 同组织关系: {len(extra)} 条")
    logger.info(f"  → 共 {len(all_triples)} 条语义关系")

    # 统计类型
    by_type = Counter(t[1] for t in all_triples)
    logger.info(f"\n关系类型分布:")
    for rel, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        logger.info(f"  {rel:15s}: {cnt:4d} 条")

    # 展示样例
    logger.info(f"\n抽取样例:")
    for t in all_triples[:12]:
        logger.info(f"  {t[0]:40s} ──{t[1]:15s}──> {t[2]}")

    # Step 3: 写入
    logger.info(f"\n[Step 3/4] 写入 GraphLite ({len(all_triples)} 条)...")
    stats = write_to_graph(all_triples)

    # Step 4: 验证
    logger.info(f"\n[Step 4/4] 验证...")
    time.sleep(2)
    try:
        req = urllib.request.Request(
            f"{API_BASE}/query",
            data=json.dumps({"query": "MATCH ()-[r:RELATES_TO]->() RETURN r.relation, COUNT(*) as cnt GROUP BY r.relation ORDER BY cnt DESC"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
            logger.info("GraphLite 关系边统计 (抽样):")
            for row in d.get("rows", [])[:10]:
                r = row[0] if isinstance(row, list) else ""
                c = row[1] if isinstance(row, list) and len(row) > 1 else 0
                logger.info(f"  {str(r):15s}: {c} 条")
    except Exception as e:
        logger.warning(f"验证失败: {e}")

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"完成! 耗时: {elapsed:.1f}s")
    logger.info(f"  读取: {len(episodes)} 条 EpisodeNode")
    logger.info(f"  抽取: {len(all_triples)} 条语义关系")
    logger.info(f"  写入: {stats['written']} 条 (错误: {stats['errors']})")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
