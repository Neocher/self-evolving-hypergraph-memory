# SHM v7 任务书：τ 衰减形式 A/B 实验（指数 → 幂律 / 分数阶）

**类型**：实验性 P1，非重写。加一个 `tau_form` 配置开关，默认 `exp` 完全向后兼容，A/B 分支启用 `pow` / `frac`。
**仓库**：`/home/user/self-evolving-hypergraph-memory`
**分支**：`exp/tau-form-ab`（从 main 拉）
**版本**：改完四处同步 bump（`shm/_version.py` + `pyproject.toml` + `VERSION` + `README.md`），建议 `v7.0.0-alpha` 或 `v6.24.0`（视改动规模自定，CI 断言一致）。

---

## 【缺口】现状实证

核心衰减公式在 `core/tau_decay.py:279-282`：

```python
exponent = -dt / tau_decay
return self.config.tau_initial * math.exp(exponent)   # ← 纯指数衰减
```

`_get_effective_tau_decay`（同文件 :215-259）里所有自适应机制——importance 调制、core 轨 ×2、访问频次 boost、AdaMem 可学习遗忘（v2.5，:126-166）——**全部只缩放 `tau_decay` 这个时间常数**（钳在 [300s, 7200s]），**没人改过"指数"这个衰减形式**。

数值验证（SHM 实际参数，τc=300s，threshold=0.1）：
- 现状下，一条 3 天没被访问的记忆 τ 直接 underflow 归零（代码注释承认这是"预期行为"）
- 幂律 α=0.5 下 3 天保留 1.3e-6 残值，14 天后还有 6e-8 长尾
- 这就是"长记忆"的实际含义——幂律/分数阶把"指数遗忘"换成"幂律长记忆"

理论参照：Frac（arXiv 2609.36314, 2026-09）——分数阶 SSM，用"有限个对数间隔的指数模式加权求和"近似重尾幂律核，1.3B LM 长上下文超 Mamba/Gated DeltaNet。

## 【新能力】要加的东西

### 1. 配置开关（`config/settings.py` + `defaults.yaml`）

`TauDecayConfig` 加一个字段：

```python
tau_form: str = "exp"   # "exp" | "pow" | "frac"
# "exp":  现状，τ₀·exp(-dt/τc)（默认，完全向后兼容）
# "pow":  τ₀·(1 + dt/τc)^(-1/α)，幂律长记忆
# "frac": Frac 近似，K 个对数间隔的指数模式求和（重尾核近似）

# pow 模式专用
alpha: float = 0.5       # 幂律强度，α∈(0,1]，越小长记忆越强
# frac 模式专用
frac_K: int = 8          # 指数模式数
frac_scale_factor: float = 10.0  # 对数间隔的最大尺度倍数
```

**同步 `defaults.yaml`**（SHM AGENTS.md 坑：yaml 覆盖代码默认值，改默认必须同步 yaml）。

环境变量：`SHM_TAU__TAU_FORM=pow` 等，按现有 `SHM_TAU__*` 命名规则加。

### 2. 核心实现（`core/tau_decay.py`）

`compute_tau` 内按 `tau_form` 分支：

```python
def compute_tau(self, node_id, created_at=None, force_now=None, fact_track="active"):
    ...
    dt = max(0, now - created)
    tau_decay = self._get_effective_tau_decay(node_id, fact_track=fact_track)
    if self.config.tau_form == "pow":
        alpha = self._effective_alpha(node_id)  # 支持 AdaMem 学 alpha
        return self.config.tau_initial * (1.0 + dt/tau_decay) ** (-1.0/alpha)
    elif self.config.tau_form == "frac":
        return self._frac_decay(node_id, dt, tau_decay)
    else:  # exp（默认）
        exponent = -dt / tau_decay
        if exponent < -700:
            return 0.0
        return self.config.tau_initial * math.exp(exponent)

def _frac_decay(self, node_id, dt, tau_decay):
    K = self.config.frac_K
    sf = self.config.frac_scale_factor
    # 对数间隔的 τc 尺度：tau_decay 到 tau_decay*sf
    scales = [tau_decay * (sf ** (k/(K-1))) for k in range(K)]
    weights = [1.0/(k+1) for k in range(K)]
    W = sum(weights)
    return sum(w * math.exp(-dt/s) for w, s in zip(weights, scales)) / W
```

**pow 模式下 AdaMem 学 α 而不是学 τc**（最小改动接入点）：
- `NodeMemoryInfo` 加 `alpha_custom: Optional[float] = None`
- `AdaptiveDecayLearner.update_decay` 在 `tau_form=="pow"` 时走 alpha 更新分支（学 `alpha_custom`，范围钳在 [0.1, 1.0]）
- `set_custom_alpha(node_id, alpha)` 新方法，仿照 `set_custom_decay`
- `tau_form=="exp"` 时 alpha 分支不启用，行为不变

### 3. 关键兼容性处理（诚实标注的三个风险）

**风险 1：`exponent < -700 → return 0.0` 的 underflow 哨兵**
- 现状：exp 模式下 dt 足够大时直接 return 0.0
- pow/frac 模式下 0.0 几乎不再出现（幂律长尾）
- **处理**：`exponent < -700` 这行只在 `tau_form=="exp"` 分支生效，pow/frac 不走这个 early-return
- **要审计**：全仓 grep 依赖"τ 归零=完全衰减"语义的下游（gate 判定、prune 触发等），确认 pow/frac 下 τ 不归零不会破坏逻辑

**风险 2：`is_decay_candidate` 的 threshold 语义**
- 现状：`tau < decay_threshold(0.1)` → 修剪候选
- pow/frac 模式下 τ 下降更慢，0.1 阈值会让"修剪候选"时刻大幅后移
- **处理（保守）**：本次**不改** `decay_threshold`，让 dream 的 prune 节奏自然漂移，A/B 时观察 prune 数量变化
- **可选（激进）**：加 `decay_threshold_form_aware` 开关，pow/frac 下自动把 threshold 下调到 0.01，保持 prune 频率接近 exp 模式
- 建议先做保守版，激进版留 A/B 数据再定

**风险 3：`refresh_on_access` 的 reset 语义**
- 现状：访问时"重置 dt≈0 → τ 回初始强度"，且审计 P2-12 已发现 refresh 是死代码（compute_tau 仍按旧 created_at 衰减）
- 幂律哲学是"历史全保留"，reset 到 0 会丢长尾
- **处理**：本次**不修** refresh（它是既有死代码，独立任务），pow/frac 分支下 refresh 行为与 exp 保持一致（同样"无效"），A/B 时 refresh 因素被控制变量
- **TODO 注释**：在 pow/frac 分支加一行注释说明"refresh 死代码独立任务，本次不修"

### 4. 测试（`tests/`）

**新增 `tests/test_tau_form_ab.py`**（仿照 `test_tau_wiring.py` 结构）：

- `test_exp_unchanged`：`tau_form="exp"` 时 `compute_tau` 与现有公式逐点一致（回归保护）
- `test_pow_long_tail`：`tau_form="pow", alpha=0.5` 时 3d/7d/14d 的 τ 值 > 0（指数同条件下为 0）
- `test_pow_alpha_extremes`：alpha→0.1 时 τ 下降更慢（长记忆更强），alpha→1.0 时退化为 (1+x)^-1 形式
- `test_frac_matches_pow_roughly`：frac 模式 K=8 的曲线应与 pow α≈0.5 同量级（验证近似正确）
- `test_frac_K1_equals_exp`：frac 模式 K=1 时退化为单指数（与 exp 一致）
- `test_ada_mem_learn_alpha`：pow 模式下 AdaMem 的 SGD 更新 `alpha_custom` 而非 `tau_decay_custom`
- `test_ada_mem_exp_unchanged`：exp 模式下 AdaMem 仍更新 `tau_decay_custom`，`alpha_custom` 始终 None
- `test_config_yaml_override`：`defaults.yaml` 里设 `tau_form: pow` 时 config 读到 pow（yaml 覆盖代码默认值的坑）
- `test_env_override`：`SHM_TAU__TAU_FORM=pow` 生效

**跑全量回归**：`pytest tests/ -x -q`，确保 `tau_form="exp"` 默认下零回归。

## 【判据】A/B 验证

### 离线仿真（第一道门，不依赖 LoCoMo 全量）

在仓库内加一个 `scripts/tau_form_sim.py`，用 SHM 实际节点时间戳（从现有 DB 抽样 1000 条）离线对比三种 form 的 τ 曲线：
- 输出：每条记忆在 1h/1d/3d/7d/14d 的 τ 值
- 断言：pow/frac 在 3d+ 的 τ 显著 > exp（幂律长尾生效）
- 这是**快速自检**，证明公式实现正确

### LoCoMo-Refined 长程召回（第二道门，决定性）

按用户既有评测框架（shm-eval-calibration skill）：

- **判据**：LoCoMo-Refined 的**长程召回子集**（gap>3d 的题）上，pow/frac vs exp 的相对 DiD
- **重复**：≥3 次独立 run（每次 fresh server，τ 状态从零开始）
- **对照**：run 内对照（同一次 server 启动下，先跑 exp 基线再跑 pow/frac，控制 reader 版本/embedding 版本一致）
- **噪声地板**：4.4pp（用户既有的同代码漂移基线）
- **通过线**：pow 或 frac 在长程召回上**超过 exp 4.4pp 以上**，且短程召回（gap<1d）不降超过 2pp

### 失效条件（止损）

- 如果 pow 和 frac 在长程召回上都**没超过 exp 4.4pp 噪声地板**，判定"SHM 瓶颈不在衰减形式"，本次实验止损，合入但默认保持 `tau_form="exp"`
- 此时结论指向用户已知方向：瓶颈在 reader 证据使用层（71% 错题 gold 已在 ctx），应转攻 reader 而非衰减

## 验收标准（AC）

- [ ] `tau_form` 配置开关 + `alpha`/`frac_K`/`frac_scale_factor` 参数就位，`defaults.yaml` 同步
- [ ] `compute_tau` 三分支实现，exp 分支**逐点回归一致**
- [ ] pow 模式下 AdaMem 学 `alpha_custom`，exp 模式下学 `tau_decay_custom`（不混）
- [ ] 风险 1/2/3 按"保守版"处理（underflow 哨兵只在 exp 分支、threshold 不动、refresh 不修）
- [ ] 9 个新测试全绿，全量 `pytest tests/ -x -q` 零回归
- [ ] `scripts/tau_form_sim.py` 离线仿真跑通，pow/frac 在 3d+ 长尾生效
- [ ] 版本四处同步 bump，`VERSION` 文件更新
- [ ] LoCoMo-Refined 长程召回 A/B 跑完，出报告（含 DiD + 噪声地板判定 + 失效条件是否触发）

## 提交规范

- 三段式 commit：`根因: τ 衰减固定指数形式，3d+ 记忆 underflow 归零` → `修复: 加 tau_form 开关支持 pow/frac 幂律长记忆` → `验证: 9 新测试 + 全量回归零破坏 + LoCoMo A/B`
- 合入 main 前：LoCoMo A/B 报告附在 PR 描述
- 打 tag `vX.Y.Z` 并推送（如果合入）

## 2026-10-02 零效应核查（A/B 启动前拦截）

启动多臂 A/B 前核查了 tau_form 进入 bench 检索的路径，结论：**LoCoMo bench
口径下 τ 形式 A/B 是零效应实验**，机制证据：

1. bench 检索 boost 用库内静态属性 `e.tau_initial`（query_router:1296/1345/1466/1498/1734），
   而 eval_db_p1 的 40 条 EpisodeNode `tau_initial` **全为 NULL** → 所有 doc 恒定
   boost=1.5，对排序零影响。
2. `compute_tau`（v6.24.0 改的衰减函数）调用点仅 gateway/dream_pipeline/仿真脚本，
   bench 检索链路完全不经过它。
3. dream 归档（τ 唯一有效影响点，decay_threshold 驱逐）在 bench 中从不触发
   （pkl 快照恢复、无 dream 循环）。

推论：任务书"LoCoMo 长程召回 A/B"AC 在 bench 口径下测不到 tau_form 效应。
τ 改动在 SHM 生产层（SHM server 检索 + dream 归档）才有意义。
待决：改 AC 口径（生产层长程召回）或先合入默认 exp 待生产层验证。
