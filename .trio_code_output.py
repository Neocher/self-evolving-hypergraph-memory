Now let me check what's still unfixed and review the actual current state of each finding.
## 分析

- **已修复**: Finding 1-6, 8 (acp_bridge.py, core/llm_client.py, gateway/auth.py, config/settings.py, gateway/gateway_api.py)
- **未修复**: Finding 9 (embedding/encoder.py - 从 os.environ 读取 key 每次调用), Finding 10 (config/settings.py - 空默认 key 静默无认证)
- **方案**:
  - A: 在 TextEncoder 中预加载 key 到私有实例变量，传给 `_cloud_embed` 避免每次调用读 os.environ
  - B: 修改 `_cloud_embed` 函数签名接受 api_keys 可选参数

采用方案 A（侵入最小，与 llm_client.py 的 `_env_keys` 模式一致）
Now modify `_cloud_embed` to accept pre-loaded keys:
Now update `_do_embed` to pass `self._api_keys` to `_cloud_embed`:
Now update `create_encoder` to use the preloaded keys:
Fix Finding 10: Add empty API key validation to `LLMConfig`:
Now run tests to verify no regressions:
All 246 tests pass. Let me verify syntax of the modified files specifically:
Let me verify the final state of both modified files:
## 通过状态: 有条件通过

## 发现的问题

**已修复 (上一轮)**: Finding 1-6, 8 — 6 个文件已应用修复（acp_bridge.py, core/llm_client.py, gateway/auth.py, config/settings.py, gateway/gateway_api.py）

**本轮新增修复**:

| # | 严重度 | 文件 | 问题 | 修复 |
|---|--------|------|------|------|
| 9 | **中** | `embedding/encoder.py:56-90` | `_cloud_embed` 每次调用从 `os.environ` 读 API key；TextEncoder 无 `__repr__` 保护 | 预加载 key 到 `self._api_keys` 私有 dict，传给 `_cloud_embed`；添加 `__repr__` 不暴露 key 值 |
| 10 | **低** | `config/settings.py:174` | 空默认 `api_key=""` 切换到云服务时静默发空 `Authorization` | 添加 `__post_init__` 检查：endpoint 非本地时空 key 打印 warning |

**未修复 (超出本轮范围)**: Finding 7 (TLS/HTTPS) — 需基础设施改动，已备注到 `defaults.yaml`

## 改进建议

无 — 全部安全发现已修复或已记录。
