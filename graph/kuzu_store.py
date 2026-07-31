"""
KuzuStore — 已废弃（引擎迁移至 GraphLite）
========================================
历史: KuzuStore → RyuStore（Kuzu fork）→ GraphLiteStore（当前）

```python
# 新代码请使用：
from graph.graphlite_store import GraphLiteStore, CircuitBreakerOpen
```
"""
import warnings
warnings.warn(
    "graph.kuzu_store 已废弃。当前图引擎为 GraphLite，"
    "请使用: from graph.graphlite_store import GraphLiteStore",
    DeprecationWarning, stacklevel=2
)

# 兼容导入（若 ryu_store 可用则提供；不可用则给占位）
try:
    from graph.ryu_store import RyuStore, RyuConfig
    KuzuStore = RyuStore
    KuzuConfig = RyuConfig
except ImportError:  # ryugraph 未安装（GraphLite 时代）
    KuzuStore = None
    KuzuConfig = None

__all__ = ["KuzuStore", "KuzuConfig", "RyuStore", "RyuConfig"]
