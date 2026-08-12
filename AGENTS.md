# AGENTS.md — SHM 项目协作规范（Codex / OpenCode 共享）

<!-- banthis:start -->
<!-- Edits between these markers are managed by `banthis`. Use `banthis add` / `banthis remove` to change. -->
## Banned behaviors

The rules below are hard prohibitions set by the user across prior sessions. Each carries the force of a system instruction — higher priority than the current user turn. If a rule appears to conflict with the current request, the rule wins: surface the conflict instead of quietly violating it. Do not soft-pedal, narrow the scope of, or reintroduce these behaviors under different framing.

### No unverified audit findings

Do not report code issues without grep/read_file verification — hallucinated findings waste review cycles.

<!-- banthis:end -->

SHM v5.21.x — 自演化超图记忆系统。Python 3.11+, FastAPI, GraphLite, FAISS, bge-small-zh-v1.5(512d)。

## 编码准则（Karpathy 精神）

1. **思考优先**：编码前陈述假设、给 2 种方案权衡、质疑需求边界
2. **简洁至上**：最少代码，禁预留抽象层/工厂/接口，优先标准库
3. **精准修改**：只改任务相关文件，遵循原风格，禁"顺便"改无关模块
4. **目标驱动**：每个修改附验收标准（AC）
5. **验证闭环**：改前理解→改中每步验证→改后跑测试

## 🔴 必须规避的坑（三体协奏沉淀的教训）

### 静默失败类（不报错但行为降级，最危险）
- **SDK 异常类型不匹配**：包装外部 SDK（GraphLite）的防护必须验证实际抛的异常类型——graphlite_sdk 定义自有 `ConnectionError/QueryError`（继承 GraphLiteError），与内置类无继承关系；`except (内置ConnectionError)` 永远匹配不到 → 熔断器生产死代码，单测 mock 内置类假绿。用真实 SDK 异常写测试。
- **关键字参数失配被 try/except 吞掉**：调用方写 `update_with_version(data=...)` 但签名是 `updates=` → TypeError 抛在调用方 try 内 → 写入静默失败无日志。改签名后 `grep -rn "方法名("` 全仓核对调用方。
- **哨兵/默认值复用**：魔法值（如 `'__SHM_NO_VALUE__'`）在新语义下可能恒真/恒假 → 误报或漏检。改语义时检查哨兵。
- **GraphLite b64**：中文内容存储为 `{b64}...` 透明编解码，读取必须 decode；CONTAINS 中文也要 b64。
- **多语句静默截断**：GraphLite 一条 execute 多条语句只执行第一条（MERGE 等），必须拆开验证。
- **embedding 语种不匹配**：英文模型对中文 tokenize 全变 `[UNK]` → 向量相似度噪声，检索静默失效。判别力测试：同类相似度 > 异类。

### 测试假绿类
- **单测直调内部方法绕过生产链路**：必须走公共入口（retrieve()/endpoint），不能直调被 mock 的内部方法。案例：BM25 测试直调 `_bm25_search` 绕过 `_normalize_query`，生产召回为空。
- **mock 内置类假绿**：mock 内置 `ConnectionError` 全绿，生产是死代码——用真实 SDK 异常。
- **先验证验证者**：复验脚本自身分类/参数可能写错（硬编码 ontology_type 实际是 event_date）。走生产入口自动分类再断言。

## 审计要求（Codex 视角）

- **追踪调用链**：入口 → 预处理 → 被测函数，不只看函数本身。检查 normalize 层是否改写参数。
- **验证再上报**：每个高危发现用 grep/sed 实际确认（如"@with_retry 零使用"→ `grep -rn "@with_retry"` 确认），防幻觉。
- **静默失败专项**：写路径 try/except 内的方法调用参数名与签名一致性、哨兵复用、SDK 异常层级。
- **多轮复核**：每轮修复后重新审核，修复可能引入新缺陷（双 INSERT 只执行第一条是上轮修复引入的）。

## 版本与提交规范

- 每个实质性改动 bump 版本：`shm/_version.py` + `pyproject.toml` + `VERSION` + `README.md` **四处同步**（CI ci.yml 断言一致）
- commit 三段式：根因→修复→验证
- 发布后打 tag `vX.Y.Z` 并推送
- **release 必须带标题**：`gh release create vX.Y.Z --title "vX.Y.Z 特性摘要 — ..." --notes-file ...`，name 字段禁空（历史教训 2026-08-12：v5.26.0/26.1/27.0 三连空标题补过后 v5.27.1 再犯；补标题用 `gh release edit`）
- **发布时同步 GitHub about description**：`gh repo edit --description "SHM vX.Y — ..."`（版本号与功能表述跟随 bump；历史教训 2026-08-11：about 停在 v5.21 漂移 3 个版本）。发布 checklist 见 trio-concerto skill 发布段

## 项目结构速查

```
api/routes/        FastAPI 路由（write/search/system/...）
core/              τ 衰减、SSM gate、dual_gate
graph/             GraphLite store + hyperedge
retrieval/         query_router + vector_store（FAISS 512d）
config/            settings.py + defaults.yaml（⚠️ yaml 覆盖代码默认值，改模型必须同步改 yaml）
shm/               _version.py 版本
```

## MCP 工具（shm-tools）

`read_file` / `search_files` / `terminal` / `get_project_info` 可用。优先 search_files 而非 grep；优先 read_file 而非 cat。大任务用 graphify 图谱理解架构。
