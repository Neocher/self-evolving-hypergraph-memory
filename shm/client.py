"""
SHM Python SDK — Self-evolving Hypergraph Memory 客户端
=======================================================
用法:
    from shm.client import SHMClient

    client = SHMClient(base_url="http://127.0.0.1:8000")
    client.add_episode("今天天气很好", source="user", namespace="chat1")
    results = client.search("天气", top_k=3)

依赖: httpx (pip install httpx)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class SHMClient:
    """SHM REST API 客户端"""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        # Config 自动注入（P2-2）
        try:
            from config.settings import get_settings
            cfg = get_settings().shm_client
            self.base_url = (base_url or cfg.base_url).rstrip("/")
            self.timeout = timeout or int(cfg.timeout)
        except Exception:
            self.base_url = (base_url or "http://127.0.0.1:8000").rstrip("/")
            self.timeout = timeout or 30

    # ─── 内部 HTTP 方法 ─────────────────────────────────────

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"} if data else {}
        body = json.dumps(data) if data else None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.request(method, url, content=body, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text if e.response else str(e)
            raise RuntimeError(f"HTTP {e.response.status_code}: {detail}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Request failed: {e}") from e

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, data: dict) -> dict:
        return self._request("POST", path, data)

    def _delete(self, path: str) -> dict:
        return self._request("DELETE", path)

    # ═══════════════════════════════════════════════════════════
    # 记忆写入
    # ═══════════════════════════════════════════════════════════

    def add_sensory(self, content: str, source: str = "user",
                    namespace: str = "") -> dict:
        """写入感觉缓冲区 (Layer1)"""
        return self._post("/memories/sensory", {
            "content": content, "source": source, "namespace": namespace,
        })

    def add_episode(self, content: str, source: str = "user",
                    namespace: str = "", force_promote: bool = False) -> dict:
        """直接创建情节节点 (Layer2)"""
        return self._post("/memories/episodes", {
            "content": content, "source": source,
            "namespace": namespace, "force_promote": force_promote,
        })

    # ═══════════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 5,
               namespace: str = "") -> List[Dict[str, Any]]:
        """三级融合检索（向量+BM25+实体匹配）"""
        payload: dict = {"query": query, "top_k": top_k}
        if namespace:
            payload["namespace"] = namespace
        result = self._post("/memories/retrieve", payload)
        return result.get("results", [])

    def search_vector(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """纯向量检索（直通 FAISS）"""
        result = self._post("/search/vector", {"query": query, "top_k": top_k})
        return result.get("results", [])

    def query_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """执行 Cypher 查询（只读代理）"""
        result = self._post("/query", {"query": cypher})
        return result.get("results", [])

    # ═══════════════════════════════════════════════════════════
    # 命名空间管理
    # ═══════════════════════════════════════════════════════════

    def delete_namespace(self, namespace: str) -> dict:
        """删除命名空间下所有节点"""
        return self._delete(f"/memories/namespace/{namespace}")

    # ═══════════════════════════════════════════════════════════
    # 本体系统 (Ontology v2)
    # ═══════════════════════════════════════════════════════════

    def register_entity_type(self, name: str, description: str = "",
                             parent: Optional[str] = None,
                             attributes: Optional[List[Dict]] = None) -> dict:
        """注册实体类型"""
        return self._post("/ontology/types", {
            "name": name, "description": description, "parent": parent,
            "attributes": attributes or [],
        })

    def list_entity_types(self) -> List[Dict]:
        """列出所有实体类型"""
        result = self._get("/ontology/types")
        return result.get("entity_types", [])

    def get_entity_type(self, name: str) -> dict:
        """查询实体类型详情"""
        return self._get(f"/ontology/types/{name}")

    def delete_entity_type(self, name: str) -> dict:
        """删除实体类型"""
        return self._delete(f"/ontology/types/{name}")

    def register_edge_type(self, name: str, source_types: Optional[List[str]] = None,
                           target_types: Optional[List[str]] = None,
                           symmetry: bool = False) -> dict:
        """注册边类型"""
        return self._post("/ontology/edges", {
            "name": name,
            "source_types": source_types or [],
            "target_types": target_types or [],
            "symmetry": symmetry,
        })

    def list_edge_types(self) -> List[Dict]:
        """列出所有边类型"""
        result = self._get("/ontology/edges")
        return result.get("edge_types", [])

    def get_edge_type(self, name: str) -> dict:
        """查询边类型详情"""
        return self._get(f"/ontology/edges/{name}")

    def delete_edge_type(self, name: str) -> dict:
        """删除边类型"""
        return self._delete(f"/ontology/edges/{name}")

    # ═══════════════════════════════════════════════════════════
    # 超边 (Hyperedge)
    # ═══════════════════════════════════════════════════════════

    def create_hyperedge(self, node_ids: List[str],
                         hyperedge_type: str = "episode",
                                         name: str = "") -> dict:
        """创建超边"""
        return self._post("/hyperedges", {
            "node_ids": node_ids,
            "type": hyperedge_type,
            "name": name or f"hyperedge_{hyperedge_type}",
        })

    # ═══════════════════════════════════════════════════════════
    # 系统
    # ═══════════════════════════════════════════════════════════

    def health(self) -> dict:
        """健康检查"""
        return self._get("/health")

    def stats(self) -> dict:
        """获取统计信息"""
        h = self.health()
        return h.get("stats", h)

    def trigger_dream(self) -> dict:
        """显式触发梦境"""
        return self._post("/memories/dream/trigger", {})

    def rebuild_index(self) -> dict:
        """重建 FAISS 索引"""
        return self._post("/index/rebuild", {})
