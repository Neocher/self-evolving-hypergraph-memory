---
status: active
owner_mode: goal
objective: "SHM v5.29.0 审计: 1) 查找最新版 bug (含 v5.28/5.29 并发修复的残留问题) 2) 性能提升建议 (附实测依据与 ROI 排序)"
updated_at: 2026-08-13T15:15:33+08:00
adapter_id: shm-audit-20260813
---

# Active Goal State

## Objective

SHM v5.29.0 审计: 1) 查找最新版 bug (含 v5.28/5.29 并发修复的残留问题) 2) 性能提升建议 (附实测依据与 ROI 排序)

## Authority Sources

- No explicit goal document was provided during bootstrap.

## Operating Contract

- Treat this file as the durable goal state for future agent ticks.
- Treat the authority sources above as the first context to inspect before acting.
- Read current project evidence before choosing the next action.
- Run a bounded progress segment when useful; it does not have to be one tiny step.
- Keep private evidence, credentials, local paths, and raw logs out of public commits.
- End each tick with changed files, validation, residual risk, and the next action.

## Execution Profile

- `cadence=bounded_progress_segment minimum=multi_surface_or_implementation include=coherent_artifact,targeted_validation,state_writeback spend_rule=spend_only_after_artifact_validation_writeback small_streak_threshold=2`
- Repeated small-scale follow-through should expand the next delivery batch or report a blocker before spending quota.

## Non-Goals

- Do not perform irreversible production operations without explicit approval.
- Do not publish private project evidence.
- Do not optimize for activity if no useful artifact or decision can be produced.


## User Todo / Owner Review Reading Queue

## Agent Todo

- [ ] [P1] Run `loopx check` against the project registry and record the first project-specific adapter signal or an explicit no-follow-up rationale.
  <!-- loopx:todo todo_id=todo_fa501099a20c status=open task_class=advancement_task action_kind=onboarding_connection_validation updated_at=2026-08-13T15:15:30%2B08:00 -->
- [ ] P0 codex/opencode 结构摸底
  <!-- loopx:todo todo_id=todo_09481de89f02 status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:31%2B08:00 -->
- [ ] P0 pi 广度扫描热点文件
  <!-- loopx:todo todo_id=todo_e94641e61a85 status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:32%2B08:00 -->
- [ ] P0 prime 长考深度 bug 猎杀
  <!-- loopx:todo todo_id=todo_83bad38323a0 status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:32%2B08:00 -->
- [ ] P0 claude 并发深潜 (写队列/梦境)
  <!-- loopx:todo todo_id=todo_6100ff20c228 status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:32%2B08:00 -->
- [ ] P1 bug 清单核验 (读源码逐条)
  <!-- loopx:todo todo_id=todo_7c3d477edb16 status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:33%2B08:00 -->
- [ ] P1 性能建议 ROI 排序
  <!-- loopx:todo todo_id=todo_12b796353a0b status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:33%2B08:00 -->
- [ ] P2 审计报告交付
  <!-- loopx:todo todo_id=todo_cb78731b145f status=open task_class=advancement_task action_kind=shell claimed_by=prime-commander updated_at=2026-08-13T15:15:33%2B08:00 -->

## Next Action

- [P1] Run `loopx check` against the project registry and record the first project-specific adapter signal or an explicit no-follow-up rationale.

## Recent User Feedback

- Initialized by `loopx bootstrap`.

## Progress Ledger

- Created the initial goal state and registry connection.
