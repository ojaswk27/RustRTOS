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
