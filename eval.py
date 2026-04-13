"""
Evaluate a saved PPO model against classical schedulers on any taskset.

Evaluates on both uniform and mixed-criticality settings.
With mixed criticality, T0-T2 are critical (5x weight), T3-T5 are soft.

Usage:
    uv run python eval.py                          # evaluates ppo_rtos_model/ppo_rtos.zip
    uv run python eval.py sweep_results/best_model/ppo_rtos.zip
"""

import sys
import numpy as np
from stable_baselines3 import PPO
from rtos_env import (
    RTOSEnv,
    NORMAL_TASKSET,
    STRESSED_TASKSET,
    HARD_TASKSET,
    VERY_HARD_TASKSET,
    VESTAL_TASKSET,
    MIXED_CRITICALITY,
    IDLE_ACTION,
)

EVAL_EPISODES = 200

model_path = sys.argv[1] if len(sys.argv) > 1 else "ppo_rtos_model/ppo_rtos.zip"


def round_robin(tasks, last_idx):
    n = len(tasks)
    for offset in range(1, n + 1):
        idx = (last_idx + offset) % n
        if tasks[idx].ready and tasks[idx].remaining > 0:
            return idx
    return IDLE_ACTION


def rate_monotonic(tasks):
    best, best_period = IDLE_ACTION, float("inf")
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0 and t.period < best_period:
            best, best_period = i, t.period
    return best


def edf(tasks, tick):
    best, best_dl = IDLE_ACTION, float("inf")
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0 and t.abs_deadline < best_dl:
            best, best_dl = i, t.abs_deadline
    return best


def run_eval(taskset, scheduler, model, episodes, criticality=None, variable_exec=True):
    """Run evaluation, return dict of metrics."""
    all_misses, all_crit, all_soft, all_ctx, all_starv, all_rew = [], [], [], [], [], []
    for _ in range(episodes):
        env = RTOSEnv(
            taskset=taskset, max_ticks=300, variable_exec=variable_exec,
            criticality=criticality,
        )
        obs, _ = env.reset()
        done, rr_idx, total_r = False, -1, 0.0
        while not done:
            if scheduler == "ppo":
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            elif scheduler == "rr":
                action = round_robin(env.tasks, rr_idx)
                if action != IDLE_ACTION:
                    rr_idx = action
            elif scheduler == "rms":
                action = rate_monotonic(env.tasks)
            else:
                action = edf(env.tasks, env.tick)
            obs, r, done, _, info = env.step(action)
            total_r += r
        all_misses.append(info["misses"])
        all_crit.append(info["critical_misses"])
        all_soft.append(info["soft_misses"])
        all_ctx.append(info["context_switches"])
        all_starv.append(info["starvation_total"] / 300)
        all_rew.append(total_r)
    return {
        "misses": np.mean(all_misses), "misses_std": np.std(all_misses),
        "crit_misses": np.mean(all_crit), "crit_std": np.std(all_crit),
        "soft_misses": np.mean(all_soft), "soft_std": np.std(all_soft),
        "ctx": np.mean(all_ctx),
        "starv": np.mean(all_starv),
        "reward": np.mean(all_rew), "reward_std": np.std(all_rew),
    }


def print_table(taskset_name, results, show_crit=False):
    if show_crit:
        print(f"\n{'─'*90}")
        print(f"  {taskset_name}")
        print(f"{'─'*90}")
        print(f"  {'Scheduler':<14} {'Total':>10} {'Critical':>10} {'Soft':>10} {'Ctx-Sw':>7} {'Starv':>7} {'Reward':>12}")
        print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*7} {'─'*12}")
        for name, d in results:
            print(
                f"  {name:<14} {d['misses']:>6.1f}±{d['misses_std']:<3.1f}"
                f" {d['crit_misses']:>6.1f}±{d['crit_std']:<3.1f}"
                f" {d['soft_misses']:>6.1f}±{d['soft_std']:<3.1f}"
                f" {d['ctx']:>7.1f} {d['starv']:>7.2f}"
                f" {d['reward']:>8.1f}±{d['reward_std']:<3.1f}"
            )
        print(f"{'─'*90}")
    else:
        print(f"\n{'─'*74}")
        print(f"  {taskset_name}")
        print(f"{'─'*74}")
        print(f"  {'Scheduler':<14} {'Misses':>10} {'Ctx-Sw':>7} {'Starv':>7} {'Reward':>12}")
        print(f"  {'─'*14} {'─'*10} {'─'*7} {'─'*7} {'─'*12}")
        for name, d in results:
            print(
                f"  {name:<14} {d['misses']:>6.1f}±{d['misses_std']:<3.1f}"
                f" {d['ctx']:>7.1f} {d['starv']:>7.2f}"
                f" {d['reward']:>8.1f}±{d['reward_std']:<3.1f}"
            )
        print(f"{'─'*74}")


print(f"Loading model: {model_path}")
model = PPO.load(model_path, device="cpu")

schedulers = [("EDF", "edf"), ("RMS", "rms"), ("Round Robin", "rr"), ("PPO", "ppo")]

# ── Uniform criticality (baseline comparison) ──
for taskset_name, taskset in [("Normal (U≈1.03)", NORMAL_TASKSET), ("Stressed (U≈1.15)", STRESSED_TASKSET)]:
    results = []
    for label, fn in schedulers:
        d = run_eval(taskset, fn, model, EVAL_EPISODES)
        results.append((label, d))
    results.sort(key=lambda x: x[1]["misses"])
    print_table(f"{taskset_name} — Uniform Criticality, Variable Exec", results)

# ── Mixed criticality on hard tasksets (where PPO wins) ──
for taskset_name, taskset in [
    ("Hard (U_nom≈1.57)", HARD_TASKSET),
    ("Very Hard (U_nom≈1.87)", VERY_HARD_TASKSET),
]:
    results = []
    for label, fn in schedulers:
        d = run_eval(taskset, fn, model, EVAL_EPISODES, criticality=MIXED_CRITICALITY)
        results.append((label, d))
    results.sort(key=lambda x: x[1]["crit_misses"])
    print_table(f"{taskset_name} — Mixed Criticality, Variable Exec", results, show_crit=True)

# ── Vestal (2007) inverted-priority MC benchmark ──
# Variable exec (avg 75% WCET): effective U_LO < 1, EDF handles it.
results = []
for label, fn in schedulers:
    d = run_eval(VESTAL_TASKSET, fn, model, EVAL_EPISODES, criticality=MIXED_CRITICALITY)
    results.append((label, d))
results.sort(key=lambda x: x[1]["crit_misses"])
print_table(
    "Vestal MC — Inverted Priority, Variable Exec (avg U_LO≈0.81)",
    results, show_crit=True
)

# Fixed exec: exact WCET every job — U_LO=1.075, matches xv6 behavior.
# This is the canonical Vestal failure scenario for EDF.
results_fx = []
for label, fn in schedulers:
    d = run_eval(VESTAL_TASKSET, fn, model, EVAL_EPISODES,
                 criticality=MIXED_CRITICALITY, variable_exec=False)
    results_fx.append((label, d))
results_fx.sort(key=lambda x: x[1]["crit_misses"])
print_table(
    "Vestal MC — Inverted Priority, Fixed Exec (U_LO=1.075, EDF failure mode)",
    results_fx, show_crit=True
)
