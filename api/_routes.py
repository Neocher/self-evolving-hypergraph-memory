"""
FastAPI 路由注册 — 向后兼容转发层
=================================
所有端点已迁移到 api/routes/ 子模块。
本文件仅作向后兼容导入，不定义任何 handler。
"""

# 触发子模块 @router 注册（通过 __init__.py 导入子模块）
from api.routes import router
from api.routes._deps import Services, init_services, get_services
from api.routes._deps import flush_faiss_buffer, incremental_faiss_update
from api.routes.system import rebuild_index
from api.routes.write import _process_embed_queue
