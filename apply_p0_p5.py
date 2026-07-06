#!/usr/bin/env python3
"""Apply P0-P5 ontology changes — one shot, correct boundaries."""
import ast, sys, re

PATH = '/home/admin/shm/core/ontology_validator.py'
with open(PATH) as f:
    c = f.read()

# ─── EXACT BOUNDARIES ───
s = c.find('    ENTITY_TYPE_MAP: dict[str, str] = {')
m_end = c.find('\n    }\n\n    # 实体类型', s) + len('\n    }\n')

cat_start = c.find('    ENTITY_TYPE_CATEGORIES: dict[str, str] = {', s)
cat_end = c.find('\n    }\n\n    def _extract_types', cat_start) + len('\n    }\n')

print(f"MAP:  L{c[:s].count(chr(10))+1} - L{c[:m_end].count(chr(10))}")
print(f"CAT:  L{c[:cat_start].count(chr(10))+1} - L{c[:cat_end].count(chr(10))}")

# ─── NEW MAP ───
new_map = '''    ENTITY_TYPE_MAP: dict[str, str] = {
        # --- 深度学习框架 ---
        "pytorch": "deep_learning_framework",
        "tensorflow": "deep_learning_framework",
        "jax": "deep_learning_framework",
        "mxnet": "deep_learning_framework",
        "paddlepaddle": "deep_learning_framework",
        "onnx": "deep_learning_framework",
        "keras": "deep_learning_framework",
        "theano": "deep_learning_framework",
        "caffe": "deep_learning_framework",
        # --- 机器学习模型/架构 ---
        "transformer": "ml_model",
        "bert": "ml_model",
        "gpt": "ml_model",
        "gpt-4": "ml_model",
        "gpt-4o": "ml_model",
        "gpt-3.5": "ml_model",
        "llama": "ml_model",
        "llama3": "ml_model",
        "mistral": "ml_model",
        "qwen": "ml_model",
        "deepseek": "ml_model",
        "deepseek-v3": "ml_model",
        "claude": "ml_model",
        "gemini": "ml_model",
        "clip": "ml_model",
        "vit": "ml_model",
        "resnet": "ml_model",
        "textencoder": "ml_model",
        "sentencetransformer": "ml_model",
        "all-minilm-l6-v2": "ml_model",
        "word2vec": "ml_model",
        "whisper": "ml_model",
        "stable-diffusion": "ml_model",
        "dalle": "ml_model",
        # --- 硬件 ---
        "cpu": "hardware",
        "gpu": "hardware",
        "tpu": "hardware",
        "nvidia": "hardware",
        "amd": "hardware",
        "intel": "hardware",
        "cuda": "hardware",
        "rocm": "hardware",
        "mps": "hardware",
        "asic": "hardware",
        "fpga": "hardware",
        # --- 数据库/向量搜索 ---
        "faiss": "vector_database",
        "milvus": "vector_database",
        "pinecone": "vector_database",
        "weaviate": "vector_database",
        "chromadb": "vector_database",
        "qdrant": "vector_database",
        "kuzu": "graph_database",
        "neo4j": "graph_database",
        "arangodb": "graph_database",
        "redis": "database",
        "postgresql": "database",
        "mysql": "database",
        "mongodb": "database",
        "clickhouse": "database",
        "elasticsearch": "database",
        # --- 知名公司 ---
        "tesla": "company",
        "spacex": "company",
        "openai": "company",
        "google": "company",
        "apple": "company",
        "microsoft": "company",
        "meta": "company",
        "amazon": "company",
        "aws": "cloud_platform",
        "azure": "cloud_platform",
        "gcp": "cloud_platform",
        "alibaba": "company",
        "tencent": "company",
        "baidu": "company",
        "bytedance": "company",
        "huawei": "company",
        "xiaomi": "company",
        "ibm": "company",
        "oracle": "company",
        "salesforce": "company",
        "netflix": "company",
        "uber": "company",
        "airbnb": "company",
        "spotify": "company",
        "shopify": "company",
        "twitter": "company",
        "linkedin": "company",
        "github": "company",
        "gitlab": "company",
        "redhat": "company",
        # --- 知名人物 ---
        "elon musk": "person",
        "sam altman": "person",
        "tim cook": "person",
        "satya nadella": "person",
        "sundar pichai": "person",
        "mark zuckerberg": "person",
        "jeff bezos": "person",
        "bill gates": "person",
        "steve jobs": "person",
        "larry page": "person",
        "sergey brin": "person",
        "jack ma": "person",
        # --- 中文互联网平台 ---
        "qq": "internet_platform",
        "weixin": "internet_platform",
        "wechat": "internet_platform",
        "taobao": "internet_platform",
        "douyin": "internet_platform",
        "tiktok": "internet_platform",
        "bilibili": "internet_platform",
        "sina": "internet_platform",
        "sohu": "internet_platform",
        "netease": "internet_platform",
        "zhihu": "internet_platform",
        "xiaohongshu": "internet_platform",
        "meituan": "internet_platform",
        "didi": "internet_platform",
        "jd": "internet_platform",
        "pinduoduo": "internet_platform",
        "kuaishou": "internet_platform",
        # --- 基础设施 ---
        "docker": "infrastructure",
        "kubernetes": "infrastructure",
        "k8s": "infrastructure",
        "terraform": "infrastructure",
        "ansible": "infrastructure",
        "jenkins": "infrastructure",
        "fastapi": "web_framework",
        "flask": "web_framework",
        "django": "web_framework",
        "spring": "web_framework",
        "uvicorn": "web_server",
        "nginx": "web_server",
        "apache": "web_server",
        # --- 系统/AI ---
        "shm": "memory_system",
        "hermes": "ai_agent",
        "cursor": "ide",
        # --- 中文技术术语 ---
        "\u6df1\u5ea6\u5b66\u4e60": "chinese_tech",
        "\u5411\u91cf\u6570\u636e\u5e93": "chinese_tech",
        "\u77e5\u8bc6\u56fe\u8c31": "chinese_tech",
        "\u641c\u7d22\u5f15\u64ce": "chinese_tech",
        "\u63a8\u8350\u7cfb\u7edf": "chinese_tech",
        "\u81ea\u7136\u8bed\u8a00": "chinese_tech",
        "\u673a\u5668\u5b66\u4e60": "chinese_tech",
        "\u56fe\u6570\u636e\u5e93": "chinese_tech",
        "\u795e\u7ecf\u7f51\u7edc": "chinese_tech",
        "\u7f16\u7801\u5668": "chinese_tech",
        "\u89e3\u7801\u5668": "chinese_tech",
        # --- 编程语言 ---
        "python": "programming_language",
        "rust": "programming_language",
        "go": "programming_language",
        "javascript": "programming_language",
        "typescript": "programming_language",
        "java": "programming_language",
        "swift": "programming_language",
        "kotlin": "programming_language",
        # --- 操作系统 ---
        "linux": "os",
        "ubuntu": "os",
        "centos": "os",
        "debian": "os",
        "alpine": "os",
        "macos": "os",
        "windows": "os",
        "freebsd": "os",
        # --- 数据格式/处理 ---
        "json": "data_format",
        "yaml": "data_format",
        "toml": "data_format",
        "csv": "data_format",
        "parquet": "data_format",
        "xml": "data_format",
        "numpy": "data_processing",
        "pandas": "data_processing",
        "polars": "data_processing",
        "spark": "data_processing",
        # --- 网络协议/服务 ---
        "http": "network_protocol",
        "https": "network_protocol",
        "smtp": "network_protocol",
        "websocket": "network_protocol",
        "grpc": "network_protocol",
        "dns": "network_service",
        "cdn": "network_service",
        "ddos": "network_service",
        "vpn": "network_service",
    }'''

new_cat = '''    # \u5b9e\u4f53\u7c7b\u578b -> \u7c7b\u522b\uff08\u7528\u4e8e\u6cdb\u5316\u5339\u914d\uff09
    ENTITY_TYPE_CATEGORIES: dict[str, str] = {
        "deep_learning_framework": "ml_infra",
        "ml_model": "ml_infra",
        "hardware": "infrastructure",
        "vector_database": "data_infra",
        "graph_database": "data_infra",
        "database": "data_infra",
        "company": "organization",
        "person": "people",
        "cloud_platform": "organization",
        "internet_platform": "web_service",
        "infrastructure": "infrastructure",
        "web_framework": "software",
        "web_server": "software",
        "memory_system": "system",
        "ai_agent": "ai_software",
        "ai_assistant": "ai_software",
        "ai_platform": "ai_software",
        "chinese_tech": "technology",
        "data_format": "data",
        "data_processing": "data_infra",
        "os": "platform",
        "programming_language": "language",
        "ide": "software",
        "network_protocol": "infrastructure",
        "network_service": "infrastructure",
    }'''

# Replace MAP first
c = c[:s] + new_map + c[m_end:]

# Recompute CAT boundaries after MAP shift
cat_start = c.find('    ENTITY_TYPE_CATEGORIES: dict[str, str] = {')
cat_close = c.find('\n    }\n\n    def _extract_types', cat_start)
if cat_close < 0:
    print("ERROR: cat close not found")
    print(repr(c[cat_start:cat_start+200]))
    sys.exit(1)
cat_end = cat_close + len('\n    }\n')

# Replace CAT
c = c[:cat_start] + new_cat + c[cat_end:]

# ─── P1: CJK in _extract_types ───
old_rx = '                    if re.search(r\'\\b\' + re.escape(entity) + r\'\\b\', text_lower, re.ASCII):'
new_rx = '                    has_cjk = any(ord(c) > 0x2E80 for c in entity)\n                    if has_cjk or re.search(r\'\\b\' + re.escape(entity) + r\'\\b\', text_lower, re.ASCII):'
for _ in range(2):
    i = c.find(old_rx)
    if i >= 0: c = c[:i] + new_rx + c[i+len(old_rx):]

# ─── P4: OSINT types ───
osint_block = '''    "domain_info": {
        "description": "\u57df\u540d\u6ce8\u518c/IP\u6620\u5c04/Whois\u4fe1\u606f",
        "conflict_keys": ["domain", "dns", "ip", "\u89e3\u6790", "\u6ce8\u518c"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "ip_address": {
        "description": "IP\u5730\u5740\u5730\u7406\u4f4d\u7f6e/\u5f52\u5c5e",
        "conflict_keys": ["ip", "\u5730\u7406\u4f4d\u7f6e", "\u5f52\u5c5e", "asn"],
        "contradiction_pattern": "same_entity_diff_value",
    },
    "url_link": {
        "description": "URL\u7ed3\u6784/\u5185\u5bb9\u7c7b\u578b/\u72b6\u6001",
        "conflict_keys": ["url", "http", "https", "status", "\u54cd\u5e94"],
        "contradiction_pattern": "same_entity_diff_value",
    },
'''
gf = '    "generic_fact": {\n        "description": "\u901a\u7528\u4e8b\u5b9e\uff08\u65e0\u6cd5\u5f52\u7c7b\u65f6\u4f7f\u7528\uff09",\n        "conflict_keys": [],\n        "contradiction_pattern": "embedding_contradiction",\n    },\n}'
if gf in c:
    # Remove the final closing "}" of generic_fact section
    c = c.replace(gf, '    "generic_fact": {\n        "description": "\u901a\u7528\u4e8b\u5b9e\uff08\u65e0\u6cd5\u5f52\u7c7b\u65f6\u4f7f\u7528\uff09",\n        "conflict_keys": [],\n        "contradiction_pattern": "embedding_contradiction",\n    },' + osint_block[:-1] + '\n}', 1)

# ─── P3: shared entity topology ───
old_topo_lines = '''                except Exception as e:
                    logger.debug("Topology path query failed: %s", e)

        if total_checked == 0:
            return 1.0
'''
new_topo_lines = '''                except Exception as e:
                    logger.debug("Topology path query failed: %s", e)

        # P3: also check paths between shared entities
        shared_list = list(shared)
        for i in range(len(shared_list)):
            for j in range(i + 1, len(shared_list)):
                total_checked += 1
                try:
                    result = self.kuzu.execute_cypher(
                        "MATCH (a:OntologyEntity {name: $a_name}) "
                        "MATCH (b:OntologyEntity {name: $b_name}) "
                        "OPTIONAL MATCH (a)-[:RELATES_TO*1..3]-(b) "
                        "RETURN count(*) AS cnt",
                        {"a_name": shared_list[i], "b_name": shared_list[j]},
                    )
                    if result and len(result) > 0:
                        row = result[0]
                        cnt = row.get("cnt", 0) if isinstance(row, dict) else int(row[0]) if isinstance(row, (list, tuple)) else 0
                        if cnt > 0:
                            path_found += 1
                except Exception:
                    pass

        if total_checked == 0:
            return 1.0
'''
if old_topo_lines in c:
    c = c.replace(old_topo_lines, new_topo_lines, 1)

# ─── P5: entity learning ───
c = c.replace(
    "self._ontology_synced = False  # lazy sync on first use\n",
    "self._ontology_synced = False  # lazy sync on first use\n        self._candidate_entities: dict[str, int] = {}  # P5\n",
)
learn_meth = '''    def _learn_candidate_entities(self, content: str, threshold: int = 3) -> None:
        """P5: discover unknown entity candidates."""
        import re
        candidates = re.findall(r'\\b[A-Z][a-zA-Z0-9._-]{2,49}\\b', content)
        for c in candidates:
            c_lower = c.lower()
            if c_lower not in self.ENTITY_TYPE_MAP and c_lower not in self._candidate_entities:
                self._candidate_entities[c_lower] = 1
            elif c_lower not in self.ENTITY_TYPE_MAP:
                self._candidate_entities[c_lower] = self._candidate_entities.get(c_lower, 0) + 1
                freq = self._candidate_entities[c_lower]
                if freq == threshold:
                    logger.info("P5: New entity candidate '%s' appeared %d times", c, freq)

'''
er_marker = '    def extract_and_relate(self, content: str) -> int:\n        """\u5199\u5165\u65f6\u63d0\u53d6\u5b9e\u4f53\u5171\u73b0\u5173\u7cfb'
c = c.replace(er_marker, learn_meth + er_marker, 1)
er_idx = c.find(learn_meth) + len(learn_meth)
er_body = c.find('        entities = self._extract_entity_cooccurrence(content)', er_idx)
c = c[:er_body] + '        self._learn_candidate_entities(content)\n' + c[er_body:]

# ─── VERIFY ───
try:
    ast.parse(c)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR L{e.lineno}: {e.msg}")
    lines = c.split('\n')
    for i in range(max(0, e.lineno-4), min(len(lines), e.lineno+3)):
        print(f"  {i+1}|{lines[i]}")
    sys.exit(1)

with open(PATH, 'w') as f:
    f.write(c)
print("Written OK")

# Quick verify
from core.ontology_validator import OntologyValidator
ov = OntologyValidator()
print(f"Entity count: {len(ov.ENTITY_TYPE_MAP)}")
for text in ["elon musk tesla spacex", "PyTorch CPU FAISS", "\u6df1\u5ea6\u5b66\u4e60 \u641c\u7d22\u5f15\u64ce", "baidu tencent dns"]:
    t = ov._extract_types(text)
    print(f"  {text:30s} -> {[x['entity'] for x in t]}")
