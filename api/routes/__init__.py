"""
SHM API Routes — 模块化路由组织
===============================
子模块托管所有 handler，本文件聚合导出。
"""

from api.routes._deps import (
    router, Services, init_services, get_services,
    flush_faiss_buffer, incremental_faiss_update,
)

# 导入子模块触发 @router 注册
import api.routes.write
import api.routes.search
import api.routes.hyperedges
import api.routes.communities
import api.routes.dream
import api.routes.system
import api.routes.ontology
import api.routes.visual

# 聚合导出
from api.routes.system import rebuild_index
from api.routes.write import _process_embed_queue

__all__ = [
    "router", "Services", "init_services", "get_services",
    "flush_faiss_buffer", "incremental_faiss_update",
    "rebuild_index",
]
