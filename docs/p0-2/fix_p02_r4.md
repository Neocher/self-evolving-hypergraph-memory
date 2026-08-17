# P0-2 Agentic 检索 R4 修复 — OpenCode 任务书

## 背景
Codex R4 复核判定：需再修改（1 P2 + 3 P3）。收敛中，核心 4/5 已通过。

## 修复清单

### P2-1: 收缩表补 let's + let 停用词
- **问题**：`_EN_CONTRACTION_MAP`（query_router.py:103 附近）缺 `let's` → "Let's find Apple" 被 :2016 切成 "Let"，:2022 不停用 "let" → 伪实体
- **修复**：
  - `_EN_CONTRACTION_MAP` 加 `"let's": "let us"`
  - `_PROPERTY_CANDIDATE_STOPWORDS` 加 `"let"`（仅加还原不够，let 当前不是停用词）
  - 补 `o'clock`/`ain't`/`y'all` 或对切分 token 停用

### P3-1: _expand_contractions 词边界
- **问题**：`re.sub` 无词边界（:122），命中所有格内部（cache's 中 he's → he is）
- **修复**：re.sub 加 `\b` 防护（`\b(?:...)\b`）

### P3-2: test_mcp_session_ts 补 v2
- **问题**：test_mcp_session_ts.py 只覆盖 v1 mcp_server.py；mcp_server_v2.py:116 FastMCP 透传无测试
- **修复**：补 v2 静态 schema 断言（或 mock import 后验证透传）

### P3-3: _extract_property_terms 撇号死条件
- **问题**：`_extract_property_terms` 未调 `_expand_contractions`（:2233 `[a-z]{2,}` 不会产出撇号 token，don't 停用是死条件）
- **修复**：调用 `_expand_contractions` 或注释说明（实际影响极低，因后续按 attr_names 过滤）

## 约束
- 遵循 AGENTS.md 编码准则；版本号不 bump
- 全部测试通过（984 基线无回归，-p no:randomly）

## 验收
1. P2-1 有测试（"Let's find Apple" 不产出 "Let" 伪实体）
2. 全量 pytest 无回归
