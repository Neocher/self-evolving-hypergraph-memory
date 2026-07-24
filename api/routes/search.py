"""
检索路由 (retrieve, vector, cypher, namespace)
=============================
路由实现当前在 api/_routes.py，本模块提供未来拆分入口。

拆分后，各域端点从 _routes.py 迁移到此文件，
app.py 将注册这些子路由而非单一 _routes.router。
"""

# TODO: 迁移端点从 api/_routes.py 到此文件
# from api._routes import router  # 当前路由仍集中在 _routes.py
