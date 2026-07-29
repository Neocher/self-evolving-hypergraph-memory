## 设计 A2A Server + ACP Adapter for SHM Phase 2

### 读取文件
1. /home/admin/shm/gateway/gateway_api.py — 统一内部接口
2. /home/admin/shm/acp_bridge.py — 现有 ACP 桥

### A2A Server (gateway/a2a_server.py)
运行在 :8001，公开 AgentCard：
- memory.write → GatewayAPI.write_sensory()
- memory.store_episode → GatewayAPI.store_episode()
- memory.retrieve → GatewayAPI.retrieve()
- memory.search → GatewayAPI.search_vector()
- memory.health → GatewayAPI.health()
- memory.dream → GatewayAPI.trigger_dream()

A2A 使用 HTTP JSON 传输，AgentCard 在 /.well-known/agent-card.json

### ACP Adapter (gateway/acp_adapter.py)
扩展现有 acp_bridge.py 的 dispatch 模型，新增 SHM actions：
- shm:write → GatewayAPI.write_sensory()
- shm:retrieve → GatewayAPI.retrieve()
- shm:health → GatewayAPI.health()

### 输出
每个文件的类/函数签名 + 伪代码 + 端口细节。
