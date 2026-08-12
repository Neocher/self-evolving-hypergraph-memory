# Hermes ↔ SHM 对接指南（安装人员必读）

> **为什么需要这份文档**：2026-08-12 实证——shm_v5 MemoryProvider 插件曾配置 2 周但**从未被 Hermes 加载**，
> 根因是插件目录放错位置（`plugins/memory/` 多一层），导致"写得多、读得少"（SHM 持续写入但 Hermes 从不自动检索）。
> 安装/集成后**必须运行检测脚本** `scripts/check_hermes_integration.py` 验证闭环。

## 架构

```
Hermes Agent (gateway 进程)
  └→ MemoryManager
       └→ MemoryProvider 插件 (plugins/shm_v5/)
            ├── prefetch(query)      # 每轮对话前自动检索 SHM（P0 核心！）
            ├── system_prompt_block() # 系统提示注入记忆状态
            ├── sync_turn()          # 每 N 轮持久化对话
            └── fact_store/search/feedback  # 记忆工具
                    │  HTTP :8000
                    ▼
              SHM v5.26 (GraphLite + FAISS)
```

## 安装步骤

### 1. 插件目录（最容易错的一步）

插件**必须**放在 `$HERMES_HOME/plugins/shm_v5/`（**直接子目录**）：

```bash
# ✅ 正确
~/.hermes/plugins/shm_v5/
    ├── __init__.py      # MemoryProvider 实现 (SHMv5MemoryProvider)
    └── plugin.yaml      # name: shm_v5

# ❌ 错误（Hermes 发现机制不扫描这一层，插件静默不加载）
~/.hermes/plugins/memory/shm_v5/
```

Hermes 的插件发现逻辑（`plugins/memory/__init__.py::_iter_provider_dirs`）只扫描
`$HERMES_HOME/plugins/` 的**直接子目录**。多一层 `memory/` 就永远发现不了。

### 2. 配置文件

`~/.hermes/config.yaml`：

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 5000
  user_char_limit: 800
  provider: shm_v5          # ← 必须与插件目录名/类名一致
  flush_min_turns: 6
  nudge_interval: 10
```

`plugin.yaml` 的 `name` 字段也必须与目录一致（shm_v5）。

### 3. 重启 Hermes Gateway

```bash
systemctl --user restart hermes-gateway   # 或 hermes gateway restart
```

⚠️ 从 gateway 进程内部执行会被守卫拦截（SIGTERM 传播自杀），须从外部 shell 执行。

### 4. 新会话生效

**插件注入只对 gateway 重启后的新会话生效**——已有会话不重建 system prompt
（prompt caching 原则）。验证时请开新会话。

## 对接成功的标志

新会话的 system prompt 应包含：

```
记忆系统: SHM v5.26 (N 节点已索引)
工具: fact_store (存储), fact_search (检索), fact_feedback (反馈)
## SHM v5 记忆检索
[0.63] <相关记忆内容> ...
```

## 常见坑（全部实测踩过）

| # | 坑 | 症状 | 修复 |
|:--|:--|:--|:--|
| 1 | **插件目录位置错**（多一层 memory/）| 配置了 provider 但从未加载；SHM 日志无 retrieve 调用 | 移到 `~/.hermes/plugins/shm_v5/` |
| 2 | **prefetch 参数名不匹配**（`k` vs `top_k`）| Pydantic 忽略未知字段 → 返回默认 20 条（上下文膨胀） | 插件用 `top_k`（SHM API 字段） |
| 3 | 插件 yaml name 与目录不一致 | 配置匹配失败 | `name: shm_v5` 与目录同名 |
| 4 | 在 gateway 内重启 gateway | 守卫拦截 / SIGTERM 自杀 | 外部 shell 执行或 `start_new_session=True` |
| 5 | 验证时用旧会话 | 看不到注入（system prompt 快照）| 开新会话验证 |
| 6 | SHM 认证 | 非 dev_mode 时 API 需 Bearer token | 插件 `_api_call` 需带 `Authorization` 头 |

## 验证方法（自动）

```bash
# 一键检测：插件目录/配置/可加载性/SHM 可达/prefetch 端到端
python3 scripts/check_hermes_integration.py

# 手动验证（可选）：
# 1. 插件可见
python3 -c "import sys; sys.path.insert(0, '$HERMES_HOME/hermes-agent'); from plugins.memory import discover_memory_providers; print([n for n,d,a in discover_memory_providers() if n=='shm_v5'])"
# 2. 新会话触发自动检索
#    CLI: hermes chat -q "hi"  后查 SHM 日志:
#    journalctl -u shm-server --since '1 minute ago' | grep 'POST /memories/retrieve'
```

## 检测脚本说明

`scripts/check_hermes_integration.py` 检查 6 项（全部 PASS 才表示对接成功）：

1. SHM 服务存活（`GET /health`）
2. 插件目录存在且位置正确
3. 插件可被 Hermes 发现并加载（`load_memory_provider("shm_v5")`）
4. config.yaml `memory.provider == shm_v5`
5. prefetch 实测返回非空（语义检索连通）
6. system_prompt_block 输出节点状态

任何一项 FAIL 都会给出修复建议。安装人员必须在部署后运行此脚本确认。
