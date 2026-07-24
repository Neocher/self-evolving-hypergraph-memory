"""
SHM Daily Briefing v1 — 多源日报引擎
======================================
采集策略: 代理可用→走代理，不可用→直连/备选
        每个板块至少2个独立信源，避免单点依赖
"""

import json, subprocess, sys, os, re, time, logging, html
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("日报")

PROXY = "socks5://127.0.0.1:1081"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def fetch(url, proxy=None, timeout=12, follow=True, referer=None):
    cmd = ["curl", "-s", "--max-time", str(timeout), "-L" if follow else "",
           "-H", f"User-Agent: {UA}",
           *(("-e", referer) if referer else []),
           *(("-x", proxy) if proxy else []),
           url]
    r = subprocess.run([c for c in cmd if c], capture_output=True, text=True, timeout=timeout+3)
    return r.stdout

def try_json(text):
    try: return json.loads(text)
    except: return None

# ═══════════════════════════════════════════════════════════════
# 采集器
# ═══════════════════════════════════════════════════════════════

def collect_hn():
    """HN 热门 — 国际科技/AI"""
    items = []
    txt = fetch("https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=15", PROXY, 10)
    d = try_json(txt)
    if d:
        for h in d.get("hits", []):
            items.append({
                "title": h.get("title",""), "url": h.get("url","") or f"https://news.ycombinator.com/item?id={h.get('objectID','')}",
                "points": h.get("points",0), "source": "HackerNews"
            })
    log.info(f"  HN: {len(items)} 条")
    return items

def collect_github():
    """GitHub 趋势 — 开源动态"""
    items = []
    txt = fetch("https://api.github.com/search/repositories?q=created:>2026-07-15&sort=stars&order=desc&per_page=10", PROXY, 10)
    d = try_json(txt)
    if d and "items" in d:
        for r in d["items"][:10]:
            items.append({
                "title": r.get("full_name",""), "url": r.get("html_url",""),
                "desc": (r.get("description") or "")[:80],
                "stars": r.get("stargazers_count",0), "source": "GitHub"
            })
    log.info(f"  GitHub: {len(items)} 条")
    return items

def collect_google_news():
    """Google News — 全球新闻聚合"""
    items = []
    # Google News RSS 走代理（302重定向可跟）
    txt = fetch("https://news.google.com/rss?hl=en-US&gl=US", PROXY, 10)
    if txt:
        for title, link in re.findall(r'<title>([^<]+)</title>.*?<link>([^<]+)</link>', txt, re.DOTALL):
            if title and 'Google News' not in title:
                items.append({"title": title.strip(), "url": link, "source": "Google News"})
    log.info(f"  Google News: {len(items)} 条")
    return items[:10]

def collect_arxiv(categories=["cs.AI","cs.RO","q-bio.BM"]):
    """arXiv — 学术前沿"""
    items = defaultdict(list)
    for cat in categories:
        txt = fetch(f"https://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending", timeout=15)
        if txt:
            for entry in re.finditer(r'<entry>.*?</entry>', txt, re.DOTALL):
                e = entry.group()
                title = re.search(r'<title>(.*?)</title>', e, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', e, re.DOTALL)
                link = re.search(r'<id>(.*?)</id>', e)
                if title:
                    items[cat].append({
                        "title": title.group(1).strip(),
                        "summary": (summary.group(1).strip()[:100] + "...") if summary else "",
                        "url": link.group(1) if link else "",
                        "source": f"arXiv/{cat}"
                    })
        log.info(f"  arXiv/{cat}: {len(items[cat])} 条")
    return items

def collect_tophub():
    """今日热榜聚合 — 国内多平台热点"""
    items = []
    txt = fetch("https://tophub.today/", timeout=10)
    if txt:
        for a in re.finditer(r'<a[^>]*href="([^"]*)"[^>]*class="[^"]*"[^>]*>([^<]+)</a>', txt):
            url, title = a.group(1), a.group(2).strip()
            if title and len(title) > 8 and 'nav' not in title.lower():
                items.append({"title": title, "url": url if url.startswith("http") else f"https://tophub.today{url}", "source": "热榜聚合"})
    log.info(f"  热榜聚合: {len(items)} 条")
    return items[:15]

def collect_baidu():
    """百度新闻 — 国内新闻"""
    items = []
    txt = fetch("https://news.baidu.com/", timeout=8)
    if txt:
        for a in re.finditer(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', txt):
            url, title = a.group(1), re.sub(r'<[^>]+>','',a.group(2)).strip()
            if title and len(title) > 8 and '百度' not in title:
                items.append({"title": title, "url": url, "source": "百度新闻"})
    log.info(f"  百度新闻: {len(items)} 条")
    return items[:10]

def collect_bilibili(rid=188):
    """B站分区热门 — rid=188科技, rid=1全站"""
    items = []
    txt = fetch(f"https://api.bilibili.com/x/web-interface/ranking/v2?rid={rid}&type=all", timeout=8, referer="https://www.bilibili.com")
    d = try_json(txt)
    if d and isinstance(d, dict) and "data" in d:
        data = d["data"]
        vlist = data.get("list", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for v in vlist[:8]:
            items.append({
                "title": v.get("title",""), "url": f"https://www.bilibili.com/video/{v.get('bvid','')}",
                "author": v.get("owner",{}).get("name",""),
                "play": v.get("stat",{}).get("view",0),
                "source": "B站"
            })
    elif d and isinstance(d, list):
        for v in d[:8]:
            items.append({
                "title": v.get("title",""), "url": f"https://www.bilibili.com/video/{v.get('bvid','')}",
                "author": v.get("owner",{}).get("name",""),
                "play": v.get("stat",{}).get("view",0),
                "source": "B站"
            })
    log.info(f"  B站: {len(items)} 条")
    return items

# ═══════════════════════════════════════════════════════════════
# 归类引擎
# ═══════════════════════════════════════════════════════════════
# 通过关键词把采集到的条目归类到六大板块

SECTIONS = {
    "🌍 全球政治军事": ["war","military","defense","ukraine","russia","china","us","nato","conflict",
                       "sanction","election","diplomat","geopolitik","nuclear","中东","制裁",
                       "军事","国防","外交","冲突","朝鲜","台湾","trump","biden","putin",
                       "army","navy","air force","security","intelligence","terror",
                       "politics","political","government","congress","senate","law",
                       "protest","riot","refugee","border","treaty","alliance",
                       "attack","strike","bomb","missile","cyber","espionage",
                       "spacex","nasa","esa","rocket","satellite","launch",
                       "prison","sentence","court","trial","legal","supreme"],
    "💰 全球经济产业": ["market","stock","economy","trade","tariff","inflation","fed","interest rate",
                       "ipo","funding","acquisition","startup","venture","crypto","bitcoin",
                       "经济","股市","贸易","关税","通胀","IPO","融资","投资","bank",
                       "revenue","profit","sales","growth","debt","bond","yield",
                       "merger","spinoff","dividend","buyback","layoff","裁员",
                       "manufacturing","supply chain","logistics","retail","ecommerce",
                       "oil","gas","energy","commodity","gold","silver","metal",
                       "housing","real estate","mortgage","rent","property"],
    "🧬 生物制药医疗": ["drug","biotech","pharma","clinical","fda","gene","therapy","vaccine",
                       "brain-computer","脑机","制药","生物","医疗","临床","基因","疫苗",
                       "hospital","surgery","diagnosis","disease","cancer","tumor",
                       "protein","cell","dna","rna","genome","crispr","enzyme",
                       "patient","treatment","cure","medicine","health","wellness",
                       "脑机接口","医疗器械","病理","靶向","抗体","免疫"],
    "🤖 AI与具身智能": ["ai","artificial intelligence","llm","gpt","claude","qwen","gemini",
                       "robot","embodied","autonomous","neural","deep learning",
                       "人工智能","大模型","机器人","具身","自动驾驶","深度学习",
                       "machine learning","transformer","diffusion","reinforcement",
                       "computer vision","nlp","speech","recognition","generative",
                       "openai","anthropic","google deepmind","meta ai","xai",
                       "llama","mistral","mixtral","falcon","qwen","kimi","doubao",
                       "coding agent","code generation","copilot","cursor",
                       "drone","humanoid","quadruped","boston dynamics","tesla bot",
                       "self-driving","lidar","slam","navigation","manipulation"],
    "🇨🇳 国内热点": ["中国","北京","上海","深圳","政策","监管","数据","互联网","科技",
                     "华为","阿里","腾讯","字节","百度","小米","比亚迪",
                     "新能源汽车","芯片","半导体","5g","6g","量子",
                     "高考","教育","房价","养老","医保","社保",
                     "中美","贸易战","技术封锁","国产替代",
                     "粤港澳","长三角","一带一路","乡村振兴"],
}

def classify(item):
    """把条目归入最匹配的板块"""
    text = (item.get("title","") + " " + item.get("desc","")).lower()
    scores = {}
    for section, keywords in SECTIONS.items():
        scores[section] = sum(1 for kw in keywords if kw.lower() in text)
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "📡 其他"

# ═══════════════════════════════════════════════════════════════
# 产出 HTML
# ═══════════════════════════════════════════════════════════════

def render_html(sections_data, stats):
    """生成暗色主题日报 HTML"""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_parts = ["""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHM 全球要闻简报</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:20px;max-width:960px;margin:0 auto}
h1{font-size:1.5em;margin-bottom:4px;background:linear-gradient(90deg,#58a6ff,#bc8cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:#8b949e;font-size:.85em;margin-bottom:20px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;font-size:.8em}
.stat-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px}
.stat-box span{color:#58a6ff;font-weight:bold}
.section{margin-bottom:24px}
.section h2{font-size:1.1em;padding:8px 12px;background:#161b22;border-left:3px solid #58a6ff;border-radius:4px;margin-bottom:8px}
.item{display:flex;align-items:flex-start;padding:8px 12px;gap:8px;border-bottom:1px solid #21262d;transition:background .15s}
.item:hover{background:#161b22}
.item .idx{color:#484f58;min-width:24px;font-size:.85em}
.item a{color:#58a6ff;text-decoration:none;flex:1;font-size:.9em;line-height:1.4}
.item a:hover{color:#79c0ff}
.item .meta{color:#8b949e;font-size:.75em;white-space:nowrap;margin-top:1px}
.footer{margin-top:30px;padding:12px;text-align:center;color:#484f58;font-size:.8em;border-top:1px solid #21262d}
</style></head><body>
"""]

    html_parts.append(f'<h1>🌐 SHM 全球要闻简报</h1>')
    html_parts.append(f'<div class="sub">{date_str} · 多源自动聚合 · 覆盖{stats["sources"]}个信源</div>')
    
    html_parts.append('<div class="stats">')
    html_parts.append(f'<div class="stat-box">📰 条目 <span>{stats["total"]}</span></div>')
    html_parts.append(f'<div class="stat-box">📡 信源 <span>{stats["sources"]}</span></div>')
    html_parts.append(f'<div class="stat-box">📂 板块 <span>{stats["sections"]}</span></div>')
    html_parts.append('</div>')

    section_icons = {
        "🌍 全球政治军事": "🌍", "💰 全球经济产业": "💰", "🧬 生物制药医疗": "🧬",
        "🤖 AI与具身智能": "🤖", "🇨🇳 国内热点": "🇨🇳", "📡 其他": "📡"
    }
    
    for sec_name in ["🌍 全球政治军事", "💰 全球经济产业", "🧬 生物制药医疗", "🤖 AI与具身智能", "🇨🇳 国内热点", "📡 其他"]:
        items = sections_data.get(sec_name, [])
        if not items:
            continue
        icon = section_icons.get(sec_name, "📡")
        html_parts.append(f'<div class="section"><h2>{icon} {sec_name[2:]} <span style="font-weight:normal;color:#8b949e;font-size:.8em">({len(items)})</span></h2>')
        for i, item in enumerate(items[:12], 1):
            title = html.escape(item.get("title",""))
            url = item.get("url","#")
            src = html.escape(item.get("source",""))
            meta = f"[{src}"
            if item.get("points"):
                meta += f" · {item['points']}pts"
            elif item.get("stars"):
                meta += f" · {item['stars']}★"
            elif item.get("play"):
                meta += f" · {item['play']//10000}万播放"
            meta += "]"
            html_parts.append(
                f'<div class="item"><span class="idx">{i}.</span>'
                f'<a href="{url}" target="_blank">{html.escape(title)}</a>'
                f'<span class="meta">{meta}</span></div>'
            )
        html_parts.append('</div>')

    html_parts.append(f'<div class="footer">SHM v5.8.2 · 数据来源: HN/GitHub/arXiv/B站/百度/热榜聚合 · {date_str}</div>')
    html_parts.append('</body></html>')
    
    return "\n".join(html_parts)

# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    start = time.time()
    log.info("=" * 50)
    log.info("SHM 日报引擎 v1 启动")
    log.info("=" * 50)
    
    # Step 1: 多源并行采集
    log.info("\n[采集]")
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        f1 = ex.submit(collect_hn)
        f2 = ex.submit(collect_github)
        f3 = ex.submit(collect_google_news)
        f4 = ex.submit(collect_arxiv)
        f5 = ex.submit(collect_tophub)
        f6 = ex.submit(collect_baidu)
        f7 = ex.submit(collect_bilibili, 188)  # 科技
        f8 = ex.submit(collect_bilibili, 1)     # 全站
        
        hn_items = f1.result()
        gh_items = f2.result()
        gn_items = f3.result()
        arxiv_items = f4.result()
        th_items = f5.result()
        bd_items = f6.result()
        bl_tech = f7.result()
        bl_all = f8.result()
    
    log.info(f"\n[归类]")
    # 合并所有条目
    all_items = hn_items + gh_items + gn_items + th_items + bd_items + bl_tech + bl_all
    
    # arXiv 条目展开
    for cat, items in arxiv_items.items():
        all_items.extend(items)
    
    # 去重
    seen = set()
    unique_items = []
    for item in all_items:
        key = item.get("title","")[:40].lower()
        if key not in seen and len(item.get("title","")) > 5:
            seen.add(key)
            unique_items.append(item)
    
    log.info(f"  去重后: {len(unique_items)} 条 (原始: {len(all_items)} 条)")
    
    # 归类
    sections = defaultdict(list)
    for item in unique_items:
        sec = classify(item)
        sections[sec].append(item)
    
    for sec, items in sections.items():
        log.info(f"  {sec}: {len(items)} 条")
    
    # Step 3: 生成 HTML
    log.info(f"\n[生成]")
    sources = set(item.get("source","") for item in unique_items)
    stats = {
        "total": len(unique_items),
        "sources": len(sources),
        "sections": len([s for s in sections.values() if s]),
    }
    html_content = render_html(sections, stats)
    
    date_tag = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"/tmp/shm_daily_{date_tag}.html"
    with open(out_path, "w") as f:
        f.write(html_content)
    log.info(f"  HTML: {out_path} ({len(html_content)} chars)")
    
    elapsed = time.time() - start
    log.info(f"\n✅ 完成! 耗时 {elapsed:.1f}s")
    log.info(f"  条目: {stats['total']} | 信源: {stats['sources']} | 板块: {stats['sections']}")
    
    # 打印摘要
    log.info(f"\n📋 摘要:")
    for sec_name in ["🌍 全球政治军事", "💰 全球经济产业", "🧬 生物制药医疗", "🤖 AI与具身智能", "🇨🇳 国内热点"]:
        items = sections.get(sec_name, [])
        if items:
            log.info(f"  {sec_name}:")
            for item in items[:3]:
                log.info(f"    • {item['title'][:60]} [{item.get('source','')}]")
    
    return out_path

if __name__ == "__main__":
    main()
