"""GraphLite CONTAINS 文本搜索契约测试（P0-3）。

背景：GraphLite Rust lexer 不支持 UTF-8（实测 panic `not a char boundary;
inside '记'`），中文写入经 _interpolate/_gql_value 做 b64 编码存储。
b64 块编码无子串保持性 → 中文 CONTAINS 不保证命中。

契约：
  - 英文 content CONTAINS 英文子串 → 保证命中
  - 中文 content CONTAINS 中文子串 → 不保证命中，但查询不崩溃
  - 中文检索主通道依赖向量（FAISS）/ BM25（字符 n-gram），非 GraphLite CONTAINS
"""
import uuid
import pytest


@pytest.fixture
def gstore(graphlite_store):
    return graphlite_store


class TestGraphLiteContainsContract:

    def test_english_contains_hit(self, gstore):
        """英文 content CONTAINS 英文子串 → 保证命中"""
        ep_id = str(uuid.uuid4())
        gstore.create_episode({
            "id": ep_id,
            "content": "hello world from test suite",
            "created_at": 1.0,
            "tau_initial": 1.0,
            "source": "test",
        })

        rows = gstore.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.content CONTAINS $term "
            "RETURN e.id",
            {"term": "world"},
        )
        ids = [r[0] if isinstance(r, (list, tuple)) else r.get("e.id", "") for r in rows]
        assert ep_id in ids, "英文 CONTAINS 应命中"

    def test_english_contains_miss(self, gstore):
        """英文 content CONTAINS 不存在的子串 → 返回空"""
        ep_id = str(uuid.uuid4())
        gstore.create_episode({
            "id": ep_id,
            "content": "hello world",
            "created_at": 1.0,
            "tau_initial": 1.0,
            "source": "test",
        })

        rows = gstore.query_cypher(
            "MATCH (e:EpisodeNode) WHERE e.content CONTAINS $term "
            "RETURN e.id",
            {"term": "nonexistent"},
        )
        assert len(rows) == 0, "不存在的子串应返回空"

    def test_chinese_contains_does_not_crash(self, gstore):
        """中文 CONTAINS 查询不崩溃（b64 编码下不保证命中）"""
        ep_id = str(uuid.uuid4())
        gstore.create_episode({
            "id": ep_id,
            "content": "这是一段中文测试内容",
            "created_at": 1.0,
            "tau_initial": 1.0,
            "source": "test",
        })

        # 查询不抛异常即为通过（b64 编码下命中为 bonus，不保证）
        try:
            rows = gstore.query_cypher(
                "MATCH (e:EpisodeNode) WHERE e.content CONTAINS $term "
                "RETURN e.id",
                {"term": "中文"},
            )
            # 不崩溃即通过；命中率不做硬断言
            ids = [r[0] if isinstance(r, (list, tuple)) else r.get("e.id", "") for r in rows]
            # b64 块编码无子串保持性：可能命中也可能不命中，不崩溃即可
            assert isinstance(rows, list)
        except Exception as e:
            pytest.fail(f"中文 CONTAINS 查询不应崩溃: {e}")
