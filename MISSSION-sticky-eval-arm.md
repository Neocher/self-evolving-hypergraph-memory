
## 2026-10-02 AC-4 首跑 FAIL + 语义修正（Hermes 亲跑）

HYDE_STICKY=1 STICKY_SUFF=1, 26 题×2【独立进程】: ctx 完全相同 5/26 (19%) FAIL。
根因 = 任务书语义歧义被暴露: 进程内 dict 缓存对"两次独立进程各跑一遍 26 题"
**天然零命中** (每进程首跑全 cache miss → LLM 照样重生成) → sticky 在跨 run 场景
完全没生效, 19% 与非 sticky 基线无差别。
**修正**: sticky 缓存必须【文件级 per-run 落盘】(run1 写, run2 同 run-id 命中,
不同 run-id 互不影响), 才能真正消跨 run 漂移。进程内语义仅作单测。
判据不变: 文件 sticky on, 同 run-id 两次独立进程, ctx 完全相同率 ≥23/26。
