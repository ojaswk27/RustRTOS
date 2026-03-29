# RTOS RL Scheduler — Clean Parity Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four correctness bugs (deadline miss detection, broken multi-tick execution, Running-state deadline blindspot, hardcoded time_since_scheduled) by rewriting Python env and Rust scheduler with identical task models and tick ordering.

**Architecture:** Python `RTOSEnv` and Rust `Scheduler` share the same 5-field task model (`ready` bool, `remaining`, `abs_deadline`, `next_release`, `last_scheduled`) and the same tick lifecycle: check deadlines → release → observe → act → execute → increment. `export_weights.py` generates `src/policy.rs` directly instead of requiring manual paste.

**Tech Stack:** Python 3.11, gymnasium, stable-baselines3, pytest; Rust `no_std` `thumbv7em-none-eabihf`, cortex-m-rt, cortex-m-semihosting, QEMU.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add pytest dev dependency |
| `tests/test_rtos_env.py` | Create | Python unit tests (TDD) |
| `rtos_env.py` | Modify | Fix tick ordering; remove `completed_this_period` |
| `src/main.rs` | Modify | `cfg_attr` gates so crate compiles on host for `cargo test` |
| `src/task.rs` | Rewrite | Replace `TaskState` enum with `ready: bool`; add `last_scheduled`; add `#[cfg(test)]` suite |
| `src/scheduler.rs` | Rewrite | Fix tick ordering; fix execution loop; fix `build_state`; add `hprintln` test stub |
| `export_weights.py` | Modify | Add `generate_policy_rs()` that writes `src/policy.rs` |
| `src/policy.rs` | Generated | Written by `export_weights.py` after training — never hand-edited |

---

## Task 1: Add pytest and write failing Python tests

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_rtos_env.py`

- [ ] **Step 1: Add pytest**

```bash
uv add --dev pytest
```

Expected output: `pyproject.toml` updated, lockfile updated.

- [ ] **Step 2: Create tests directory and empty init**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 3: Write tests/test_rtos_env.py**

```python
"""
Tests for rtos_env.py correctness.

Two tests catch real bugs and fail before the fix:
  - test_deadline_miss_at_period_boundary  (Bug 1: wrong tick ordering)
  - test_deadline_check_before_release     (Bug 1: same root cause)

Three tests are regression guards that pass before and after:
  - test_no_false_deadline_miss_on_completion
  - test_nonready_task_emits_all_zeros
  - test_time_since_scheduled_dynamic
"""

import pytest
import numpy as np
from rtos_env import RTOSEnv, IDLE_ACTION


# ── helpers ──────────────────────────────────────────────────────────────────

def run_env(taskset, actions, max_ticks=None):
    """Run env with a list of actions. Returns final info dict."""
    if max_ticks is None:
        max_ticks = len(actions)
    env = RTOSEnv(taskset=taskset, max_ticks=max_ticks)
    env.reset()
    info = {}
    for a in actions:
        _, _, done, _, info = env.step(a)
        if done:
            break
    return info


# ── Bug 1 tests (should FAIL before fix, PASS after) ─────────────────────────

def test_deadline_miss_at_period_boundary():
    """
    A task that never receives CPU must record a miss at every period boundary.
    Bug 1: _do_releases ran before _check_deadlines, overwriting abs_deadline
    so the miss was never detected. Expects 2 misses in 10 ticks with period=5.
    """
    info = run_env(
        taskset=[(5, 5, 2)],
        actions=[IDLE_ACTION] * 10,
        max_ticks=10,
    )
    assert info["misses"] == 2, (
        f"Expected 2 misses (at tick 5 and tick 10), got {info['misses']}. "
        "Likely caused by releases overwriting abs_deadline before the check."
    )


def test_deadline_check_before_release():
    """
    Misses must be counted even when the period boundary and deadline coincide.
    (All tasks in both tasksets have implicit deadlines: period == deadline.)
    """
    info = run_env(
        taskset=[(5, 5, 3)],
        actions=[IDLE_ACTION] * 15,
        max_ticks=15,
    )
    assert info["misses"] == 3, (
        f"Expected 3 misses (ticks 5, 10, 15), got {info['misses']}."
    )
    assert info["completions"] == 0


# ── Regression guards (should PASS before and after) ─────────────────────────

def test_no_false_deadline_miss_on_completion():
    """A task that completes before its deadline must not be counted as a miss."""
    # task: period=10, deadline=10, wcet=2. Run it on the first 2 ticks of each period.
    actions = [0 if i % 10 < 2 else IDLE_ACTION for i in range(30)]
    info = run_env(taskset=[(10, 10, 2)], actions=actions, max_ticks=30)
    assert info["misses"] == 0
    assert info["completions"] == 3


def test_nonready_task_emits_all_zeros():
    """A task with no pending job must produce a zero feature vector."""
    env = RTOSEnv(taskset=[(10, 10, 1)], max_ticks=20)
    env.reset()
    # Run task 0 (wcet=1): completes after 1 tick, ready becomes False
    obs, _, _, _, _ = env.step(0)
    # Features [0:4] belong to task 0 — all must be 0.0 once completed
    assert list(obs[0:4]) == [0.0, 0.0, 0.0, 0.0], (
        f"Expected zeros for completed task, got {obs[0:4]}"
    )


def test_time_since_scheduled_dynamic():
    """
    time_since_scheduled (feature index 1) must reflect actual last_scheduled,
    not a hardcoded constant.
    """
    # Use period=100 so max_period=100 and math is clean.
    env = RTOSEnv(taskset=[(100, 100, 5)], max_ticks=200)
    obs, _ = env.reset()
    # Before any execution: last_scheduled=-1 → feature = max_period/max_period = 1.0
    assert obs[1] == pytest.approx(1.0), (
        f"Expected 1.0 before first schedule, got {obs[1]}"
    )
    # Run task 0 at tick 0
    obs, _, _, _, _ = env.step(0)
    # After tick 0 → tick 1: time_since_scheduled = (1-0)/100 = 0.01
    assert obs[1] == pytest.approx(0.01), (
        f"Expected 0.01 after scheduling at tick 0 (now tick 1), got {obs[1]}"
    )
```

- [ ] **Step 4: Run tests — expect 2 failures**

```bash
uv run pytest tests/test_rtos_env.py -v
```

Expected output:
```
FAILED tests/test_rtos_env.py::test_deadline_miss_at_period_boundary
FAILED tests/test_rtos_env.py::test_deadline_check_before_release
PASSED tests/test_rtos_env.py::test_no_false_deadline_miss_on_completion
PASSED tests/test_rtos_env.py::test_nonready_task_emits_all_zeros
PASSED tests/test_rtos_env.py::test_time_since_scheduled_dynamic
2 failed, 3 passed
```

If all 5 pass, the bugs are already fixed. If more than 2 fail, investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "test: add failing Python tests for deadline miss ordering bug"
```

---

## Task 2: Fix rtos_env.py

**Files:**
- Modify: `rtos_env.py`

The two changes are: (1) remove `completed_this_period` from `TaskSim` and all call sites, (2) swap the order of `_check_deadlines` and `_do_releases` in `step()`.

- [ ] **Step 1: Replace TaskSim class**

In `rtos_env.py`, replace the entire `TaskSim` class (lines 42–66) with:

```python
class TaskSim:
    """Internal task state for simulation."""

    __slots__ = (
        "period",
        "deadline",
        "wcet",
        "remaining",
        "next_release",
        "abs_deadline",
        "ready",
        "last_scheduled",
    )

    def __init__(self, period: int, deadline: int, wcet: int):
        self.period = period
        self.deadline = deadline
        self.wcet = wcet
        self.remaining = 0
        self.next_release = 0
        self.abs_deadline = 0
        self.ready = False
        self.last_scheduled = -1
```

- [ ] **Step 2: Simplify _do_releases**

Replace `_do_releases` with:

```python
    def _do_releases(self):
        """Release tasks whose period boundary has arrived. Runs AFTER _check_deadlines."""
        for t in self.tasks:
            if self.tick >= t.next_release:
                t.remaining = t.wcet
                t.abs_deadline = self.tick + t.deadline
                t.ready = True
                t.next_release = self.tick + t.period
```

- [ ] **Step 3: Simplify _check_deadlines**

Replace `_check_deadlines` with:

```python
    def _check_deadlines(self) -> int:
        """Check for deadline misses. Must run BEFORE _do_releases."""
        misses = 0
        for t in self.tasks:
            if t.ready and self.tick >= t.abs_deadline:
                misses += 1
                t.ready = False
                t.remaining = 0
        return misses
```

- [ ] **Step 4: Fix ordering and remove completed_this_period in step()**

Replace the `step()` method with:

```python
    def step(self, action: int):
        reward = -0.01  # small per-tick cost encourages urgency

        # Execute action
        completions = 0
        if action != IDLE_ACTION and action < self.n_tasks:
            t = self.tasks[action]
            if t.ready and t.remaining > 0:
                t.remaining -= 1
                t.last_scheduled = self.tick
                if t.remaining == 0:
                    t.ready = False
                    completions = 1
                    reward += 1.0

        # Context switch penalty (task-to-task only)
        if (
            action != self.last_action
            and action != IDLE_ACTION
            and self.last_action != IDLE_ACTION
        ):
            reward -= 0.05
        self.last_action = action

        self.tick += 1

        # 1. Check deadlines (before releases — catches misses at period boundaries)
        misses = self._check_deadlines()
        reward -= 2.0 * misses

        # 2. Release new jobs
        self._do_releases()

        self.deadline_misses += misses
        self.completions += completions

        obs = self._build_obs()
        terminated = self.tick >= self.max_ticks
        return (
            obs,
            reward,
            terminated,
            False,
            {
                "misses": self.deadline_misses,
                "completions": self.completions,
            },
        )
```

- [ ] **Step 5: Run tests — expect all 5 to pass**

```bash
uv run pytest tests/test_rtos_env.py -v
```

Expected output:
```
PASSED tests/test_rtos_env.py::test_deadline_miss_at_period_boundary
PASSED tests/test_rtos_env.py::test_deadline_check_before_release
PASSED tests/test_rtos_env.py::test_no_false_deadline_miss_on_completion
PASSED tests/test_rtos_env.py::test_nonready_task_emits_all_zeros
PASSED tests/test_rtos_env.py::test_time_since_scheduled_dynamic
5 passed
```

- [ ] **Step 6: Commit**

```bash
git add rtos_env.py
git commit -m "fix(env): check deadlines before releases; remove completed_this_period"
```

---

## Task 3: Make Rust crate testable on host

**Files:**
- Modify: `src/main.rs`

`cargo test` compiles for the host (`x86_64-unknown-linux-gnu`), but `main.rs` has `#![no_std]` / `#![no_main]` and pulls in embedded-only crates. Wrapping these with `cfg_attr(not(test), ...)` lets the test build succeed without touching the release build.

- [ ] **Step 1: Replace src/main.rs**

```rust
//! RL-Based Adaptive RTOS Scheduler — bare-metal entry point.
//!
//! Runs on ARM Cortex-M4 (STM32F411) under QEMU. Defines the same 6-task
//! periodic taskset used in Python training, then runs the scheduler for
//! one hyperperiod (300 ticks). Output goes via semihosting to the QEMU console.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

mod policy;
mod scheduler;
mod task;

#[cfg(not(test))]
use cortex_m_rt::entry;
#[cfg(not(test))]
use cortex_m_semihosting::{debug, hprintln};
#[cfg(not(test))]
use panic_halt as _;

#[cfg(not(test))]
#[entry]
fn main() -> ! {
    let _ = hprintln!("========================================");
    let _ = hprintln!("  RL-RTOS Scheduler — Cortex-M4 Demo");
    let _ = hprintln!("========================================\n");

    // Same taskset as Python training: (period, deadline, wcet)
    // Total utilization ≈ 1.03 — intentionally overloaded.
    let tasks = [
        task::Task::new(0, 10, 10, 2),
        task::Task::new(1, 15, 15, 3),
        task::Task::new(2, 20, 20, 4),
        task::Task::new(3, 30, 30, 5),
        task::Task::new(4, 50, 50, 8),
        task::Task::new(5, 100, 100, 10),
    ];

    let mut sched = scheduler::Scheduler::new(tasks);

    // Run for one hyperperiod: LCM(10,15,20,30,50,100) = 300 ticks
    sched.run(300);

    let _ = hprintln!("\nScheduler finished. Halting.");
    debug::exit(debug::EXIT_SUCCESS);

    loop {}
}
```

- [ ] **Step 2: Verify it still compiles for the embedded target**

```bash
cargo build --release
```

Expected: compiles without errors or warnings about unused imports.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "chore(rust): cfg_attr gates to allow cargo test on host"
```

---

## Task 4: Rewrite src/task.rs

**Files:**
- Rewrite: `src/task.rs`

Removes `TaskState` enum, replaces `state: TaskState` with `ready: bool`, adds `last_scheduled: i32`, passes `tick` to `tick_execute`, and adds a `#[cfg(test)]` suite.

- [ ] **Step 1: Write the new src/task.rs**

```rust
/// Represents a periodic real-time task.
///
/// A task has a job pending when `ready == true`. It stays ready until it
/// either completes (remaining hits 0) or misses its deadline. There is no
/// intermediate "Running" state — the scheduler simply calls `tick_execute`
/// each tick it allocates CPU to this task.
#[derive(Clone, Copy)]
pub struct Task {
    pub id: usize,
    pub period: u32,
    pub deadline: u32,
    pub wcet: u32,
    pub remaining: u32,
    pub next_release: u32,
    pub abs_deadline: u32,
    pub ready: bool,
    pub last_scheduled: i32,  // tick of last CPU allocation; -1 if never scheduled
    pub deadline_misses: u32,
}

impl Task {
    pub const fn new(id: usize, period: u32, deadline: u32, wcet: u32) -> Self {
        Self {
            id,
            period,
            deadline,
            wcet,
            remaining: 0,
            next_release: 0,
            abs_deadline: 0,
            ready: false,
            last_scheduled: -1,
            deadline_misses: 0,
        }
    }

    /// Release a new job. Called when the period boundary arrives.
    pub fn release(&mut self, tick: u32) {
        self.remaining = self.wcet;
        self.abs_deadline = tick + self.deadline;
        self.next_release = tick + self.period;
        self.ready = true;
    }

    /// Execute one tick of CPU work. Returns true if the task just completed.
    pub fn tick_execute(&mut self, tick: u32) -> bool {
        self.last_scheduled = tick as i32;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.ready = false;
            true
        } else {
            false
        }
    }

    /// Check for a deadline miss. Returns true if a miss occurred.
    /// Must be called BEFORE release() on the same tick.
    pub fn check_deadline(&mut self, tick: u32) -> bool {
        if self.ready && tick >= self.abs_deadline {
            self.deadline_misses += 1;
            self.ready = false;
            self.remaining = 0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_release_sets_fields() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0);
        assert!(t.ready);
        assert_eq!(t.remaining, 3);
        assert_eq!(t.abs_deadline, 10);
        assert_eq!(t.next_release, 10);
        assert_eq!(t.last_scheduled, -1); // not yet scheduled
    }

    #[test]
    fn test_tick_execute_partial() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0);
        let done = t.tick_execute(0);
        assert!(!done);
        assert_eq!(t.remaining, 2);
        assert!(t.ready);
        assert_eq!(t.last_scheduled, 0);
    }

    #[test]
    fn test_tick_execute_completes() {
        let mut t = Task::new(0, 10, 10, 2);
        t.release(0);
        t.tick_execute(0);
        let done = t.tick_execute(1);
        assert!(done);
        assert_eq!(t.remaining, 0);
        assert!(!t.ready);
        assert_eq!(t.last_scheduled, 1);
    }

    #[test]
    fn test_check_deadline_no_miss_before_boundary() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline = 10
        assert!(!t.check_deadline(9));
        assert!(t.ready);
        assert_eq!(t.deadline_misses, 0);
    }

    #[test]
    fn test_check_deadline_miss_at_boundary() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline = 10
        assert!(t.check_deadline(10));
        assert!(!t.ready);
        assert_eq!(t.remaining, 0);
        assert_eq!(t.deadline_misses, 1);
    }

    #[test]
    fn test_check_deadline_not_ready() {
        let mut t = Task::new(0, 10, 10, 3);
        // Never released — ready is false
        assert!(!t.check_deadline(100));
        assert_eq!(t.deadline_misses, 0);
    }

    #[test]
    fn test_check_before_release_catches_miss() {
        // Critical ordering test: check_deadline at period boundary fires
        // and records a miss BEFORE release() overwrites abs_deadline.
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline=10, next_release=10
        // At tick 10: check first, then release
        let miss = t.check_deadline(10);
        assert!(miss, "miss must be recorded at the period boundary");
        assert_eq!(t.deadline_misses, 1);
        // Now release the next job (scheduler calls this second)
        t.release(10);
        assert!(t.ready);
        assert_eq!(t.abs_deadline, 20);
        // Miss count survives the release
        assert_eq!(t.deadline_misses, 1);
    }
}
```

- [ ] **Step 2: Run Rust tests on host target**

```bash
cargo test --target x86_64-unknown-linux-gnu
```

Expected output:
```
running 7 tests
test task::tests::test_release_sets_fields ... ok
test task::tests::test_tick_execute_partial ... ok
test task::tests::test_tick_execute_completes ... ok
test task::tests::test_check_deadline_no_miss_before_boundary ... ok
test task::tests::test_check_deadline_miss_at_boundary ... ok
test task::tests::test_check_deadline_not_ready ... ok
test task::tests::test_check_before_release_catches_miss ... ok
test result: ok. 7 passed; 0 failed
```

If the build fails with linker errors, make sure Task 3 is complete (cfg_attr gates in main.rs).

- [ ] **Step 3: Verify embedded build still works**

```bash
cargo build --release
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/task.rs
git commit -m "fix(task): replace TaskState enum with ready bool; add last_scheduled; add tests"
```

---

## Task 5: Rewrite src/scheduler.rs

**Files:**
- Rewrite: `src/scheduler.rs`

Fixes tick ordering (deadlines before releases), fixes execution (check `ready` not a state enum), adds real `last_scheduled` to `build_state`, removes the now-unnecessary `max_deadline` field (replaced by the constant `MAX_DEADLINE = 100`), and adds a `hprintln!` stub so the file compiles under `cargo test`.

- [ ] **Step 1: Write the new src/scheduler.rs**

```rust
/// Tick-based preemptive scheduler.
///
/// Tick lifecycle (mirrors rtos_env.py exactly):
///   1. check_deadlines  — before releases, catches implicit-deadline misses
///   2. do_releases      — refresh jobs whose period boundary arrived
///   3. build_state      — construct 24-element Q10 observation
///   4. policy::infer    — NN picks action
///   5. execute          — run selected task for one tick
///   6. tick += 1
use crate::policy;
use crate::task::Task;

#[cfg(not(test))]
use cortex_m_semihosting::hprintln;

/// No-op stub so run() compiles under `cargo test --target x86_64-unknown-linux-gnu`.
#[cfg(test)]
macro_rules! hprintln {
    ($($arg:tt)*) => {
        {}
    };
}

const NUM_TASKS: usize = 6;
const STATE_SIZE: usize = NUM_TASKS * 4;
const Q10: i32 = 1024;
/// Largest deadline across both tasksets — used for normalization.
const MAX_DEADLINE: i32 = 100;
/// Largest period across both tasksets — used for normalization.
const MAX_PERIOD: i32 = 100;

pub struct Scheduler {
    pub tasks: [Task; NUM_TASKS],
    pub tick: u32,
    pub current_task: Option<usize>,
    pub total_misses: u32,
    pub total_completions: u32,
    pub context_switches: u32,
}

impl Scheduler {
    pub fn new(tasks: [Task; NUM_TASKS]) -> Self {
        Self {
            tasks,
            tick: 0,
            current_task: None,
            total_misses: 0,
            total_completions: 0,
            context_switches: 0,
        }
    }

    /// Step 1: record deadline misses. Must run before do_releases().
    fn check_deadlines(&mut self) {
        for t in self.tasks.iter_mut() {
            if t.check_deadline(self.tick) {
                self.total_misses += 1;
            }
        }
    }

    /// Step 2: release tasks whose period boundary has arrived.
    fn do_releases(&mut self) {
        for t in self.tasks.iter_mut() {
            if self.tick >= t.next_release {
                t.release(self.tick);
            }
        }
    }

    /// Step 3: build the Q10-encoded state vector sent to the policy.
    /// Layout: [ttd, tss, rem_ratio, is_ready] × 6 tasks.
    /// Non-ready tasks emit all zeros.
    fn build_state(&self) -> [i32; STATE_SIZE] {
        let mut state = [0i32; STATE_SIZE];
        for (i, t) in self.tasks.iter().enumerate() {
            let base = i * 4;
            if t.ready {
                // time_to_deadline: (abs_deadline - tick) / MAX_DEADLINE
                let ttd = if t.abs_deadline > self.tick {
                    (t.abs_deadline - self.tick) as i32 * Q10 / MAX_DEADLINE
                } else {
                    0
                };
                state[base] = ttd.clamp(0, Q10);

                // time_since_scheduled: (tick - last_scheduled) / MAX_PERIOD
                // 1.0 (Q10) if this task has never been scheduled.
                let since = if t.last_scheduled >= 0 {
                    ((self.tick as i32 - t.last_scheduled) * Q10 / MAX_PERIOD).clamp(0, Q10)
                } else {
                    Q10
                };
                state[base + 1] = since;

                // remaining / wcet
                state[base + 2] = (t.remaining as i32 * Q10 / t.wcet as i32).clamp(0, Q10);

                // is_ready
                state[base + 3] = Q10;
            }
        }
        state
    }

    /// Execute one scheduler tick.
    pub fn tick_once(&mut self) {
        // 1. Deadlines before releases
        self.check_deadlines();
        // 2. Release new jobs
        self.do_releases();
        // 3. Observe and decide
        let state = self.build_state();
        let action = policy::infer(&state);

        // 4. Count context switches (only when the new task is actually ready)
        if action < NUM_TASKS && self.tasks[action].ready {
            if let Some(prev) = self.current_task {
                if prev != action {
                    self.context_switches += 1;
                }
            }
        }

        // 5. Execute selected task (check ready, not a state enum)
        if action < NUM_TASKS && self.tasks[action].ready {
            if self.tasks[action].tick_execute(self.tick) {
                self.total_completions += 1;
            }
            self.current_task = Some(action);
        } else {
            self.current_task = None;
        }

        // 6. Advance tick
        self.tick += 1;
    }

    /// Run the scheduler for `total_ticks` ticks, logging every 50.
    pub fn run(&mut self, total_ticks: u32) {
        let _ = hprintln!("Scheduler starting for {} ticks", total_ticks);

        for _ in 0..total_ticks {
            self.tick_once();

            if self.tick % 50 == 0 {
                let _ = hprintln!(
                    "tick={} misses={} completions={} switches={}",
                    self.tick,
                    self.total_misses,
                    self.total_completions,
                    self.context_switches
                );
            }
        }

        let _ = hprintln!("\n=== Final Stats ===");
        let _ = hprintln!("Total ticks:     {}", self.tick);
        let _ = hprintln!("Completions:     {}", self.total_completions);
        let _ = hprintln!("Deadline misses: {}", self.total_misses);
        let _ = hprintln!("Context switches:{}", self.context_switches);
        for t in &self.tasks {
            let _ = hprintln!("  Task {}: misses={}", t.id, t.deadline_misses);
        }
    }
}
```

- [ ] **Step 2: Run Rust tests (all 7 from task.rs must still pass)**

```bash
cargo test --target x86_64-unknown-linux-gnu
```

Expected: same 7 passing tests, no new failures.

- [ ] **Step 3: Verify embedded build**

```bash
cargo build --release
```

Expected: no errors. If you see `error[E0425]: cannot find value TaskState`, you have a stale import — remove `use crate::task::TaskState` from scheduler.rs.

- [ ] **Step 4: Commit**

```bash
git add src/scheduler.rs
git commit -m "fix(scheduler): deadlines before releases; fix execution; real last_scheduled in state"
```

---

## Task 6: Add policy.rs codegen to export_weights.py

**Files:**
- Modify: `export_weights.py`

Add `generate_policy_rs()` that writes `src/policy.rs` in full. The JSON export is kept for debugging. `main()` calls both.

- [ ] **Step 1: Replace export_weights.py**

```python
"""
Exports trained PPO actor weights to JSON (for inspection) and generates
src/policy.rs (for embedding in Rust) in one step.

Network architecture: 24 -> 32 (ReLU) -> 32 (ReLU) -> 7.
Q10 format: weights multiplied by 1024 and rounded to i32.

Usage:  uv run python export_weights.py
Output: policy_weights.json  (human-readable debug dump)
        src/policy.rs        (generated Rust source — do not edit by hand)
"""

import json
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "ppo_rtos_model/ppo_rtos.zip"
OUTPUT_JSON = "policy_weights.json"
OUTPUT_RUST = "src/policy.rs"
Q10_SCALE = 1024


def extract_actor_weights(model):
    """Pull weight matrices and bias vectors from the PPO actor network."""
    policy = model.policy
    layers = []
    for module in policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            layers.append((w, b))
    w = policy.action_net.weight.detach().cpu().numpy()
    b = policy.action_net.bias.detach().cpu().numpy()
    layers.append((w, b))
    return layers


def to_q10(arr: np.ndarray) -> np.ndarray:
    return (arr * Q10_SCALE).round().astype(int)


def fmt_2d(arr: np.ndarray) -> str:
    rows = ",\n    ".join(
        "[" + ", ".join(str(x) for x in row) + "]"
        for row in arr
    )
    return "[\n    " + rows + "\n]"


def fmt_1d(arr: np.ndarray) -> str:
    return "[" + ", ".join(str(x) for x in arr) + "]"


RUST_TEMPLATE = """\
// GENERATED — do not edit by hand. Run: uv run python export_weights.py

const SCALE: i32 = 1024;
const IN: usize = {in_size};
const H: usize = {h_size};
const OUT: usize = {out_size};

static W1: [[i32; {in_size}]; {h_size}] = {w1};
static B1: [i32; {h_size}] = {b1};

static W2: [[i32; {h_size}]; {h_size}] = {w2};
static B2: [i32; {h_size}] = {b2};

static W3: [[i32; {h_size}]; {out_size}] = {w3};
static B3: [i32; {out_size}] = {b3};

#[inline]
fn relu(x: i32) -> i32 {{
    if x > 0 {{ x }} else {{ 0 }}
}}

/// Run the policy network on a Q10-encoded state vector.
/// Returns action index 0-{n_tasks_max}: 0-{n_tasks} = run task, {out_size_minus1} = idle.
pub fn infer(state: &[i32; {in_size}]) -> usize {{
    let mut h1 = [0i32; {h_size}];
    for j in 0..{h_size} {{
        let mut acc: i32 = 0;
        for i in 0..{in_size} {{
            acc = acc.saturating_add(W1[j][i].saturating_mul(state[i]));
        }}
        h1[j] = relu(acc / SCALE + B1[j]);
    }}

    let mut h2 = [0i32; {h_size}];
    for j in 0..{h_size} {{
        let mut acc: i32 = 0;
        for i in 0..{h_size} {{
            acc = acc.saturating_add(W2[j][i].saturating_mul(h1[i]));
        }}
        h2[j] = relu(acc / SCALE + B2[j]);
    }}

    let mut best_idx: usize = 0;
    let mut best_val: i32 = i32::MIN;
    for j in 0..{out_size} {{
        let mut acc: i32 = 0;
        for i in 0..{h_size} {{
            acc = acc.saturating_add(W3[j][i].saturating_mul(h2[i]));
        }}
        let val = acc / SCALE + B3[j];
        if val > best_val {{
            best_val = val;
            best_idx = j;
        }}
    }}

    best_idx
}}
"""


def generate_policy_rs(layers: list, path: str) -> None:
    assert len(layers) == 3, f"Expected 3 layers (two hidden + output), got {len(layers)}"
    (w1, b1), (w2, b2), (w3, b3) = layers

    in_size = w1.shape[1]   # 24
    h_size = w1.shape[0]    # 32
    out_size = w3.shape[0]  # 7

    assert w2.shape == (h_size, h_size), f"Layer 2 shape mismatch: {w2.shape}"
    assert w3.shape[1] == h_size, f"Layer 3 input size mismatch: {w3.shape}"

    content = RUST_TEMPLATE.format(
        in_size=in_size,
        h_size=h_size,
        out_size=out_size,
        out_size_minus1=out_size - 1,
        n_tasks_max=out_size - 2,
        n_tasks=out_size - 2,
        w1=fmt_2d(to_q10(w1)),
        b1=fmt_1d(to_q10(b1)),
        w2=fmt_2d(to_q10(w2)),
        b2=fmt_1d(to_q10(b2)),
        w3=fmt_2d(to_q10(w3)),
        b3=fmt_1d(to_q10(b3)),
    )

    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  ({in_size}→{h_size}→{h_size}→{out_size})")


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    layers = extract_actor_weights(model)

    print("Network architecture:")
    for i, (w, b) in enumerate(layers):
        print(f"  Layer {i}: weight {w.shape}, bias {b.shape}")

    # JSON dump (for debugging / inspection)
    export = {"q10_scale": Q10_SCALE, "layers": []}
    for w, b in layers:
        export["layers"].append({
            "weight_shape": list(w.shape),
            "weights": w.tolist(),
            "biases": b.tolist(),
            "weights_q10": to_q10(w).tolist(),
            "biases_q10": to_q10(b).tolist(),
        })
    with open(OUTPUT_JSON, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Written: {OUTPUT_JSON}")

    # Generate src/policy.rs
    generate_policy_rs(layers, OUTPUT_RUST)
    print("\nNext step: cargo build --release")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs cleanly (no model needed yet)**

The script needs a trained model to run fully, but we can verify the codegen logic is importable:

```bash
uv run python -c "from export_weights import generate_policy_rs, fmt_1d, fmt_2d, to_q10; import numpy as np; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add export_weights.py
git commit -m "feat(export): generate src/policy.rs directly instead of manual paste"
```

---

## Task 7: End-to-end verification

**Files:** none (verification only)

Two-phase check: first with placeholder weights to confirm the execution loop is fixed, then with trained weights to confirm policy transfer.

### Phase A — Placeholder weights (sanity check)

The current `src/policy.rs` has all-zero weights. With zeros, `infer()` always returns 0 (first task). Task 0 has `wcet=2`, so it should complete every 10 ticks → 30 completions in 300 ticks. Other tasks miss every period.

- [ ] **Step 1: Build for QEMU**

```bash
cargo build --release
```

Expected: no errors.

- [ ] **Step 2: Run under QEMU**

```bash
qemu-system-arm \
  -cpu cortex-m4 \
  -machine mps2-an386 \
  -nographic \
  -semihosting-config enable=on,target=native \
  -kernel target/thumbv7em-none-eabihf/release/os_project
```

Expected output (exact numbers will vary slightly by scheduling granularity):
```
========================================
  RL-RTOS Scheduler — Cortex-M4 Demo
========================================

tick=50  misses=... completions=5  switches=0
tick=100 misses=... completions=10 switches=0
...
=== Final Stats ===
Total ticks:     300
Completions:     30
Deadline misses: <non-zero>
Context switches:0
  Task 0: misses=0
  Task 1: misses=<non-zero>
  ...
```

**Key assertion:** `Completions` must be non-zero (was 0 with the old broken execution loop). Task 0 misses must be 0 (it always gets CPU). Context switches = 0 (always same task).

### Phase B — Trained weights

- [ ] **Step 3: Train the policy**

```bash
uv run python train.py
```

Expected: training runs for 500k steps, prints comparison table at the end, saves model to `ppo_rtos_model/`. This takes several minutes.

- [ ] **Step 4: Export weights and generate policy.rs**

```bash
uv run python export_weights.py
```

Expected output:
```
Loading model from ppo_rtos_model/ppo_rtos.zip...
Network architecture:
  Layer 0: weight (32, 24), bias (32,)
  Layer 1: weight (32, 32), bias (32,)
  Layer 2: weight (7, 32),  bias (7,)
Written: policy_weights.json
Written: src/policy.rs  (24→32→32→7)
```

If you see a shape other than `(32, 24)`, `(32, 32)`, `(7, 32)`, the SB3 model architecture differs from what was configured. Check `train.py`'s `policy_kwargs` to ensure `net_arch=[32, 32]`.

- [ ] **Step 5: Build and run with trained weights**

```bash
cargo build --release
qemu-system-arm \
  -cpu cortex-m4 \
  -machine mps2-an386 \
  -nographic \
  -semihosting-config enable=on,target=native \
  -kernel target/thumbv7em-none-eabihf/release/os_project
```

Expected: output with non-zero completions across multiple tasks and context switches > 0 (the trained policy distributes CPU across tasks).

- [ ] **Step 6: Commit final state**

```bash
git add src/policy.rs policy_weights.json
git commit -m "feat: trained policy weights; full end-to-end verified under QEMU"
```

---

## Self-Review

**Spec coverage check:**
- Bug 1 (Python deadline miss): fixed in Task 2, tested in Task 1. ✓
- Bug 2 (Running state blocks execution): fixed in Task 4/5 (no Running state, check `ready`). ✓
- Bug 3 (Running state blocks deadline detection): fixed in Task 4/5 (check_deadline uses `ready`). ✓
- Bug 4 (time_since_scheduled hardcoded): fixed in Task 5 (`build_state` uses real `last_scheduled`). ✓
- Minor: context switch counted before confirming ready: fixed in Task 5. ✓
- Automated weight export: Task 6. ✓
- All normalization constants (MAX_DEADLINE=100, MAX_PERIOD=100): in Task 5. ✓

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:**
- `tick_execute(&mut self, tick: u32)` — called with `self.tick` in scheduler.rs Task 5. ✓
- `check_deadline(&mut self, tick: u32)` — called with `self.tick` in scheduler.rs Task 5. ✓
- `Task::new(id, period, deadline, wcet)` — called in main.rs Task 3 with correct argument order. ✓
- `IDLE_ACTION = MAX_TASKS = 6` — used consistently in Python tests (Task 1) and rtos_env.py (Task 2). ✓
