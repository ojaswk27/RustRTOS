# RTOS RL Scheduler — Clean Parity Rewrite

**Date:** 2026-03-28
**Status:** Approved

## Problem Summary

The existing codebase has four bugs that make both the training and deployment incorrect:

1. **Python: deadline misses at period boundaries are silently dropped.** `_do_releases()` runs before `_check_deadlines()` in `step()`. For all implicit-deadline tasks (period == deadline), this means the release overwrites `abs_deadline` before the check runs — misses are never counted, corrupting the reward signal.
2. **Rust: multi-tick tasks can never complete.** The scheduler sets `state = Running` after 1 tick, but the execution condition requires `state == Ready`. All 6 tasks have WCET ≥ 2, so no task ever completes in Rust.
3. **Rust: running tasks skip deadline checks.** `check_deadline()` only fires for `state == Ready`. Tasks in `Running` state can silently miss deadlines.
4. **Rust: `time_since_scheduled` is always 1.0.** The second feature is hardcoded to `Q10`, permanently mismatching training.

## Approach

Rewrite both sides so they are structurally identical — same task model, same tick ordering, same feature normalization. Automate weight export so `policy.rs` is generated rather than manually edited.

## Task Model

Both Python and Rust track the same five fields per task:

| Field | Type | Meaning |
|---|---|---|
| `ready` | bool | Whether a job is pending (replaces the `TaskState` enum) |
| `remaining` | u32 / int | Ticks of work left for the current job |
| `abs_deadline` | u32 / int | Absolute deadline tick of the current job |
| `next_release` | u32 / int | Tick when the next job releases |
| `last_scheduled` | i32 / int | Last tick this task received CPU (-1 if never) |

No `Running` state. A task is `ready` until it either completes (`remaining` hits 0) or misses its deadline. This matches the Python model exactly and eliminates the state-machine bugs.

## Tick Lifecycle

Every tick follows this sequence in both implementations, in this exact order:

1. **Check deadlines** — for each task: if `ready && tick >= abs_deadline`, increment miss counter, set `ready = false`, clear `remaining`.
2. **Release** — for each task: if `tick >= next_release`, set `remaining = wcet`, `abs_deadline = tick + deadline`, `next_release = tick + period`, `ready = true`.
3. **Build observation** — construct the 24-element state vector.
4. **Infer / act** — policy selects action (0–5 = run task, 6 = idle).
5. **Execute** — if the selected action is a ready task: decrement `remaining`, set `last_scheduled = tick`; if `remaining` hits 0, set `ready = false`, increment completions.
6. **Increment tick.**

**Deadline check before release** is the critical ordering fix. A task that reaches its period boundary without finishing is correctly flagged as a miss before its job is refreshed.

## Observation Vector

24 elements (6 tasks × 4 features). Non-ready tasks emit all zeros.

| Index (per task) | Feature | Formula | Range |
|---|---|---|---|
| 0 | `time_to_deadline` | `(abs_deadline - tick) / MAX_DEADLINE` | 0–1 |
| 1 | `time_since_scheduled` | `(tick - last_scheduled) / MAX_PERIOD` (1.0 if never) | 0–1 |
| 2 | `remaining / wcet` | `remaining / wcet` | 0–1 |
| 3 | `is_ready` | 1.0 if ready, else 0.0 | 0 or 1 |

`MAX_DEADLINE = 100`, `MAX_PERIOD = 100` (constants derived from the tasksets, same in Python and Rust).

In Rust, all features are encoded as Q10 fixed-point integers (multiply by 1024, round). `time_since_scheduled` uses the actual `last_scheduled` value — not hardcoded.

## Neural Network

Architecture unchanged: 24 → 32 (ReLU) → 32 (ReLU) → 7 (linear, argmax).

Rust inference unchanged: Q10 fixed-point, `saturating_add` / `saturating_mul`, divide by 1024 after each layer's accumulate.

## Weight Export — Automated Codegen

`export_weights.py` is extended to write `src/policy.rs` directly, replacing the manual paste step:

- Loads trained model, extracts actor weights (same logic as now).
- Quantizes to Q10 (same formula: `round(w * 1024).astype(int)`).
- Renders a complete `src/policy.rs` file with weights embedded as `static` arrays.
- Writes the file to disk.

`src/policy.rs` will carry a header comment: `// GENERATED — do not edit by hand. Run: uv run python export_weights.py`

The generated file has the same structure as the current one (W1/B1/W2/B2/W3/B3 statics + `infer()` function), so nothing else in the Rust crate needs to change.

## Files Changed

| File | Change |
|---|---|
| `rtos_env.py` | Fix tick ordering (deadlines before releases), replace `TaskSim` with proper dataclass, add `last_scheduled` tracking |
| `train.py` | No changes required |
| `export_weights.py` | Add codegen step that writes `src/policy.rs` |
| `src/task.rs` | Replace `TaskState` enum + `state` field with `ready: bool`; add `last_scheduled: i32` |
| `src/scheduler.rs` | Fix tick ordering (deadlines before releases); fix execution loop (check `ready` not `state`); track `last_scheduled`; fix `build_state` to use real `last_scheduled` |
| `src/policy.rs` | Becomes generated — placeholder zeros replaced by codegen |

`src/main.rs`, `Cargo.toml`, `memory.x`, `.cargo/config.toml` are unchanged.

## Reward Function

Unchanged:

| Signal | Value |
|---|---|
| Task completion | +1.0 |
| Deadline miss | -2.0 |
| Per-tick cost | -0.01 |
| Context switch (task-to-task) | -0.05 |

## Tasksets

Unchanged:

- `NORMAL_TASKSET` — U ≈ 1.03
- `STRESSED_TASKSET` — U ≈ 1.15

## Success Criteria

- Python training runs without error; reward curve trends upward over 500k steps.
- `export_weights.py` writes `src/policy.rs` without manual intervention.
- `cargo build --release` succeeds.
- `cargo run --release` under QEMU shows non-zero completions and plausible miss counts.
- PPO outperforms or matches EDF on the normal taskset in the comparison plot.
