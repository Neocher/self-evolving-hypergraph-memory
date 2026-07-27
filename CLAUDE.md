# CLAUDE.md

<!-- Karpathy Skills — 行为准则层 -->
<!-- 来源: andrej-karpathy-skills (https://github.com/bluefantasy2014/andrej-karpathy-skills) -->

## 行为准则（Karpathy Skills）

### 🧠 1. 思考优先 — Think before coding
- **编码前必须**：明确陈述所有假设，澄清需求中的歧义
- **展示方案权衡**：至少提出 2 种实现方案，分析各自优劣（复杂度/性能/可维护性），让开发者选择
- **质疑需求**：判断需求是否存在逻辑漏洞或潜在的边界情况，如有疑问先提出再编码
- 输出格式：
  ```markdown
  ## 分析
  - 假设：...
  - 方案 A（推荐）：... — 理由...
  - 方案 B（备选）：... — 权衡...
  - 边界情况：...
  ```

### ✂️ 2. 简洁至上 — Simplicity first
- **最少代码原则**：只编写解决当前问题所必需的代码
- **严禁**添加未要求的功能（"顺便修一下"、"未来可能用到"）
- **严禁**为了"可扩展性"而预留抽象层、接口类、工厂模式等非必要架构
- 优先选择标准库而非第三方依赖
- 如果一个函数能搞定，不要拆成三个文件

### 🎯 3. 精准修改 — Surgical changes
- **像外科医生一样精准**：只修改与任务直接相关的文件和内容
- **严格遵循**原项目代码的命名规范、注释风格、代码结构
- 如需修改已有代码，先说明"为什么现有实现不够，必须改"
- 新增代码必须与所在文件的既有风格完全一致（缩进/命名/注释风格）
- **禁止**在修 A 模块时"顺便"调整 B 模块的格式或命名

### ✅ 4. 目标驱动 — Goal-driven execution
- **将模糊任务转化为可验证的目标**：
  - ❌ "修复这个 bug" → ✅ "编写复现该 bug 的测试用例，修复后测试通过"
  - ❌ "优化性能" → ✅ "将 P50 延迟从 50ms 降到 20ms 以下，压测 1000 QPS 不报错"
- 每个 PR/commit 必须附带验收标准（AC — Acceptance Criteria）
- 提交代码前必须执行测试确认通过

### 🔄 5. 验证闭环 — Verify before commit
- 修改前：理解现有代码的逻辑和测试覆盖
- 修改中：每完成一个原子步骤，验证其正确性
- 修改后：运行相关测试，确认无回归

---

## 项目概述

SHM v5.10 — 自演化超图记忆系统 (Self-evolving Hypergraph Memory v5.10)
基于 τ-Hebbian-梦境三核心 + 60+调研项目融合的五层记忆架构

## 技术规范

- Python 3.10+, Pydantic v2, FastAPI, Kuzu Graph DB, FAISS, BLAKE3
- structlog 结构化日志, Prometheus 指标, asyncio 异步架构
- 覆盖率 ≥ 80%, pytest + ruff + mypy
