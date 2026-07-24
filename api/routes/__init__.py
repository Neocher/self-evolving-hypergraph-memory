"""
SHM API Routes — 模块化路由组织
================================
路由实现保留在 api/_routes.py (向后兼容)，本包提供模块化导入入口。
下一阶段：按功能域拆分为独立的 route 模块。

模块划分:
  - routes/write.py       → 记忆写入 (sensory, episodes, promote)
  - routes/search.py      → 检索 (retrieve, vector, cypher, namespace)
  - routes/dream.py       → 梦境 (trigger, reset, notify, candidates)
  - routes/hyperedges.py  → 超边管理 (CRUD)
  - routes/communities.py → 社区 + 冲突 (communities, conflicts)
  - routes/ontology.py    → 本体系统 (types, edges, discover)
  - routes/visual.py      → 视觉记忆 (visual CRUD, heatmap)
  - routes/system.py      → 系统端点 (health, metrics, audit, sessions, batch)
"""

from api._routes import router, Services, init_services, get_services
from api._routes import flush_faiss_buffer, incremental_faiss_update, rebuild_index

__all__ = ["router", "Services", "init_services", "get_services",
           "flush_faiss_buffer", "incremental_faiss_update", "rebuild_index"]
