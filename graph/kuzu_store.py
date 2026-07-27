"""
KuzuStore — 向后兼容别名（已迁移至 RyuStore）
==========================================
```python
# 新代码请使用：
from graph.ryu_store import RyuStore, RyuConfig

# 旧代码仍可用：
from graph.kuzu_store import KuzuStore, KuzuConfig  # 自动映射到 RyuStore
```
"""
import warnings
warnings.warn(
    "graph.kuzu_store 已重命名为 graph.ryu_store。"
    "请使用: from graph.ryu_store import RyuStore, RyuConfig",
    DeprecationWarning, stacklevel=2
)

from graph.ryu_store import RyuStore, RyuConfig

# 向后兼容别名
KuzuStore = RyuStore
KuzuConfig = RyuConfig

__all__ = ["KuzuStore", "KuzuConfig", "RyuStore", "RyuConfig"]
