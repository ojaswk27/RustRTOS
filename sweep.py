"""
Parallel reward-function sweep for the PPO RTOS scheduler.

Trains one PPO agent per reward configuration using all available CPU cores,
evaluates each with default reward params (fair comparison), and prints a
ranked table sorted by deadline misses on the stressed taskset.

Usage:
    uv run python sweep.py

Outputs:
    sweep_results/config_N/  ppo_rtos.zip + results.json  (per config)
    sweep_results/best_model/ppo_rtos.zip                 (winner)
"""

import json
import os
import shutil
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from stable_baselines3 import PPO

from rtos_env import RTOSEnv, RandomRTOSEnv, NORMAL_TASKSET, STRESSED_TASKSET
from train import evaluate_ppo

# ── Sweep configuration ────────────────────────────────────────────────
MISS_PENALTIES = [-2.0, -3.0, -5.0]
COMPLETION_REWARDS = [1.0, 1.5, 2.0]
URGENCY_WEIGHTS = [0.0, 0.1, 0.3, 0.5]
CONTEXT_SW_PENALTIES = [0.0, -0.02, -0.05]
STARVATION_WEIGHTS = [0.0, 0.05]
JITTER_WEIGHTS = [0.0, 0.02]

TICK_COSTS = [-0.01]

SWEEP_STEPS = 300_000   # per worker (winner can be retrained at 2M via train.py)
EVAL_EPISODES = 50
RESULTS_DIR = "sweep_results"


def build_grid():
    configs = []
    for i, (mp, cr, uw, cs, sw, jw, tc) in enumerate(
        itertools.product(
            MISS_PENALTIES, COMPLETION_REWARDS, URGENCY_WEIGHTS,
            CONTEXT_SW_PENALTIES, STARVATION_WEIGHTS, JITTER_WEIGHTS, TICK_COSTS,
        )
    ):
        configs.append({
            "config_id": i,
            "miss_penalty": mp,
            "completion_reward": cr,
            "urgency_weight": uw,
            "tick_cost": tc,
            "context_switch_penalty": cs,
            "starvation_weight": sw,
            "jitter_weight": jw,
        })
    return configs


# ── Worker (must be top-level for pickle / ProcessPoolExecutor) ────────

def train_one(cfg: dict) -> dict:
    import torch
    torch.set_num_threads(1)  # prevent each worker from spawning multiple torch threads

    out_dir = os.path.join(RESULTS_DIR, f"config_{cfg['config_id']}")
    os.makedirs(out_dir, exist_ok=True)

    # Train on random tasksets (mixed utilization) with this config's reward params
    env = RandomRTOSEnv(
        utilization_range=(0.75, 1.10),
        max_ticks=300,
        variable_exec=True,
        completion_reward=cfg["completion_reward"],
        miss_penalty=cfg["miss_penalty"],
        urgency_weight=cfg["urgency_weight"],
        tick_cost=cfg["tick_cost"],
        context_switch_penalty=cfg["context_switch_penalty"],
        starvation_weight=cfg["starvation_weight"],
        jitter_weight=cfg["jitter_weight"],
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[32, 32]),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=SWEEP_STEPS)
    model.save(os.path.join(out_dir, "ppo_rtos"))

    # Evaluate with DEFAULT reward params — apples-to-apples across all configs
    normal_rewards, normal_misses = evaluate_ppo(model, RTOSEnv, NORMAL_TASKSET, EVAL_EPISODES)
    stressed_rewards, stressed_misses = evaluate_ppo(model, RTOSEnv, STRESSED_TASKSET, EVAL_EPISODES)

    result = {
        "config_id": cfg["config_id"],
        "miss_penalty": cfg["miss_penalty"],
        "completion_reward": cfg["completion_reward"],
        "urgency_weight": cfg["urgency_weight"],
        "tick_cost": cfg["tick_cost"],
        "context_switch_penalty": cfg["context_switch_penalty"],
        "starvation_weight": cfg["starvation_weight"],
        "jitter_weight": cfg["jitter_weight"],
        "normal_misses_mean": float(normal_misses.mean()),
        "normal_misses_std": float(normal_misses.std()),
        "stressed_misses_mean": float(stressed_misses.mean()),
        "stressed_misses_std": float(stressed_misses.std()),
        "stressed_reward_mean": float(stressed_rewards.mean()),
        "model_path": os.path.join(out_dir, "ppo_rtos.zip"),
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Reporting ──────────────────────────────────────────────────────────

def print_ranked_table(results: list):
    ranked = sorted(
        results,
        key=lambda r: (r["stressed_misses_mean"], -r["stressed_reward_mean"]),
    )
    header = (
        f"{'Rank':>4}  {'ID':>3}  {'miss_pen':>8}  {'comp_rew':>8}  {'urgency':>7}  |  "
        f"{'normal_miss':>11}  {'stress_miss':>11}  {'stress_rew':>10}"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:>4}  {r['config_id']:>3}  {r['miss_penalty']:>8.2f}  "
            f"{r['completion_reward']:>8.2f}  {r['urgency_weight']:>7.2f}  |  "
            f"{r['normal_misses_mean']:>7.2f}±{r['normal_misses_std']:<4.1f}  "
            f"{r['stressed_misses_mean']:>7.2f}±{r['stressed_misses_std']:<4.1f}  "
            f"{r['stressed_reward_mean']:>10.1f}"
        )
    print("─" * len(header))


def save_best_model(results: list):
    if not results:
        return
    best = sorted(
        results,
        key=lambda r: (r["stressed_misses_mean"], -r["stressed_reward_mean"]),
    )[0]
    best_dir = os.path.join(RESULTS_DIR, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    shutil.copy2(best["model_path"], os.path.join(best_dir, "ppo_rtos.zip"))
    print(f"\nBest: config_{best['config_id']}  "
          f"miss_penalty={best['miss_penalty']}  "
          f"completion_reward={best['completion_reward']}")
    print(f"  stressed_misses={best['stressed_misses_mean']:.2f}  "
          f"stressed_reward={best['stressed_reward_mean']:.1f}")
    print(f"  Saved to {best_dir}/ppo_rtos.zip")
    print(f"\nTo retrain best config at 2M steps, set these in RTOSEnv:")
    print(f"  miss_penalty={best['miss_penalty']}  "
          f"completion_reward={best['completion_reward']}  "
          f"urgency_weight={best['urgency_weight']}")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = build_grid()
    n_workers = min(len(configs), os.cpu_count() or 4)
    print(f"Starting sweep: {len(configs)} configs across {n_workers} workers "
          f"({SWEEP_STEPS:,} steps each)...")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(train_one, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"  [done] config_{cfg['config_id']:>2}  "
                    f"miss_pen={cfg['miss_penalty']:>5.1f}  "
                    f"comp_rew={cfg['completion_reward']:>4.1f}  "
                    f"urgency={cfg['urgency_weight']:>4.2f}  "
                    f"stressed_misses={result['stressed_misses_mean']:.1f}"
                )
            except Exception as exc:
                print(f"  [FAIL] config_{cfg['config_id']}: {exc}")

    print_ranked_table(results)
    save_best_model(results)


if __name__ == "__main__":
    main()
