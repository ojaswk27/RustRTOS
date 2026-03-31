"""
Extended training script for high-stress RTOS scheduler with:
  - Resumable checkpointing every 100k steps
  - 4-stage curriculum with extreme tasksets (U up to 1.50)
  - Variable WCET to simulate real execution variability
  - Periodic evaluation on fixed extreme tasksets
  - Advanced baselines: LSF, Budget Burn Rate
  - Logging of training progress with multiple metrics

Usage:
    uv run python train_extended.py [--resume]

Outputs:
    ppo_rtos_extended_model/       Model directory
    checkpoints/                   Periodic snapshots for resumption
    training_extended.json         Detailed metrics per checkpoint
    comparison_extended.png        Enhanced baseline comparison
    extreme_tasksets/              Fixed U=1.2, 1.3, 1.4, 1.5 for benchmarking
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rtos_env import (
    IDLE_ACTION,
    NORMAL_TASKSET,
    STRESSED_TASKSET,
    RandomRTOSEnv,
    RTOSEnv,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

MODEL_DIR = "ppo_rtos_extended_model"
CHECKPOINT_DIR = "checkpoints"
METRICS_FILE = "training_extended.json"
EVAL_EPISODES = 50
CHECKPOINT_INTERVAL = 100_000  # Save every 100k steps

# Enhanced reward configuration for high-stress
REWARD_KWARGS = dict(
    miss_penalty=-3.0,
    completion_reward=2.0,
    urgency_weight=0.5,  # Increased from 0.1 to prioritize near-deadline tasks
    context_switch_penalty=0.0,
    variable_exec=True,
)

# 4-stage curriculum
TRAINING_STAGES = [
    {
        "name": "Light-Task Learning",
        "steps": 1_000_000,
        "utilization_range": (0.70, 0.90),
        "wcet_variability": 0.0,  # Fixed WCET - make light task advantage obvious
        "description": "Learn light-task preference (fixed execution time, feasible region)",
    },
    {
        "name": "Deadline Pressure",
        "steps": 1_500_000,
        "utilization_range": (0.85, 1.10),
        "wcet_variability": 0.10,  # ±10% deadline variance - show why it matters
        "description": "Mild pressure - light tasks become critical under deadlines",
    },
    {
        "name": "Robust Prioritization",
        "steps": 1_500_000,
        "utilization_range": (1.05, 1.25),
        "wcet_variability": 0.20,  # ±20% execution time variability
        "description": "Overload - robust light-task prioritization under uncertainty",
    },
    {
        "name": "Extreme Stress",
        "steps": 1_000_000,
        "utilization_range": (1.30, 1.50),
        "wcet_variability": 0.30,  # ±30% execution time variability
        "description": "Extreme stress - survive by completing light tasks",
    },
]

# ──────────────────────────────────────────────────────────────────────
# Advanced Baselines
# ──────────────────────────────────────────────────────────────────────


def least_slack_first(tasks, tick):
    """Least Slack First: choose task with minimum (deadline - remaining)."""
    best_idx, best_slack = IDLE_ACTION, float("inf")
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0:
            slack = (t.abs_deadline - tick) - t.remaining
            if slack < best_slack:
                best_idx, best_slack = i, slack
    return best_idx


def budget_burn_rate(tasks, tick):
    """Budget Burn Rate: prioritize tasks that burn work fastest to deadline."""
    best_idx, best_rate = IDLE_ACTION, 0.0
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0:
            time_left = max(1, t.abs_deadline - tick)
            rate = t.remaining / time_left  # work per tick until deadline
            if rate > best_rate:
                best_idx, best_rate = i, rate
    return best_idx


def round_robin(tasks, last_idx):
    """Cycle through tasks in index order, skipping non-ready ones."""
    n = len(tasks)
    for offset in range(1, n + 1):
        idx = (last_idx + offset) % n
        if tasks[idx].ready and tasks[idx].remaining > 0:
            return idx
    return IDLE_ACTION


def rate_monotonic(tasks):
    """Static-priority: smallest period = highest priority (RMS)."""
    best, best_period = IDLE_ACTION, float("inf")
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0 and t.period < best_period:
            best, best_period = i, t.period
    return best


def edf(tasks, tick):
    """Dynamic-priority: earliest absolute deadline first."""
    best, best_dl = IDLE_ACTION, float("inf")
    for i, t in enumerate(tasks):
        if t.ready and t.remaining > 0 and t.abs_deadline < best_dl:
            best, best_dl = i, t.abs_deadline
    return best


# ──────────────────────────────────────────────────────────────────────
# Fixed Extreme Tasksets
# ──────────────────────────────────────────────────────────────────────


def create_extreme_tasksets():
    """Create predetermined ultra-overloaded tasksets for benchmarking."""
    tasksets = {
        "U_1_20": [  # U ≈ 1.20
            (10, 10, 3),
            (15, 15, 3),
            (20, 20, 5),
            (30, 30, 5),
            (50, 50, 9),
            (100, 100, 11),
        ],
        "U_1_30": [  # U ≈ 1.30
            (10, 10, 3),
            (15, 15, 4),
            (20, 20, 5),
            (30, 30, 6),
            (50, 50, 9),
            (100, 100, 12),
        ],
        "U_1_40": [  # U ≈ 1.40
            (10, 10, 3),
            (15, 15, 4),
            (20, 20, 6),
            (30, 30, 6),
            (50, 50, 10),
            (100, 100, 13),
        ],
        "U_1_50": [  # U ≈ 1.50
            (10, 10, 4),
            (15, 15, 4),
            (20, 20, 6),
            (30, 30, 7),
            (50, 50, 10),
            (100, 100, 14),
        ],
    }
    return tasksets


def compute_utilization(taskset):
    """Compute total utilization U = sum(wcet / period)."""
    return sum(wcet / period for period, _, wcet in taskset)


# ──────────────────────────────────────────────────────────────────────
# Evaluation & Checkpointing
# ──────────────────────────────────────────────────────────────────────


def evaluate_baseline(
    env_cls, taskset, scheduler_fn, episodes, max_ticks=300
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a baseline scheduler and return per-episode (reward, misses)."""
    rewards, misses = [], []
    for _ in range(episodes):
        env = env_cls(taskset=taskset, max_ticks=max_ticks)
        obs, _ = env.reset()
        total_r, done, rr_idx = 0.0, False, -1
        while not done:
            if scheduler_fn == "rr":
                action = round_robin(env.tasks, rr_idx)
                if action != IDLE_ACTION:
                    rr_idx = action
            elif scheduler_fn == "rms":
                action = rate_monotonic(env.tasks)
            elif scheduler_fn == "edf":
                action = edf(env.tasks, env.tick)
            elif scheduler_fn == "lsf":
                action = least_slack_first(env.tasks, env.tick)
            elif scheduler_fn == "bbr":
                action = budget_burn_rate(env.tasks, env.tick)
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_fn}")
            obs, r, done, _, info = env.step(action)
            total_r += r
        rewards.append(total_r)
        misses.append(info["misses"])
    return np.array(rewards), np.array(misses)


def evaluate_ppo(
    model, env_cls, taskset, episodes, max_ticks=300
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate PPO model and return per-episode (reward, misses)."""
    rewards, misses = [], []
    for _ in range(episodes):
        env = env_cls(taskset=taskset, max_ticks=max_ticks)
        obs, _ = env.reset()
        total_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(int(action))
            total_r += r
        rewards.append(total_r)
        misses.append(info["misses"])
    return np.array(rewards), np.array(misses)


def evaluate_on_extreme_tasksets(model, extreme_tasksets, eval_episodes=30):
    """Evaluate model on all extreme tasksets; return results dict."""
    results = {}
    for name, taskset in extreme_tasksets.items():
        u = compute_utilization(taskset)
        print(f"  Evaluating on {name} (U={u:.2f})...")
        rews, misses = evaluate_ppo(model, RTOSEnv, taskset, eval_episodes)
        results[name] = {
            "utilization": u,
            "misses_mean": misses.mean(),
            "misses_std": misses.std(),
            "reward_mean": rews.mean(),
            "reward_std": rews.std(),
        }
    return results


class TrainingMetricsCallback(BaseCallback):
    """Log training metrics for later analysis."""

    def __init__(self, extreme_tasksets, eval_interval=100_000):
        super().__init__()
        self.extreme_tasksets = extreme_tasksets
        self.eval_interval = eval_interval
        self.episode_rewards = []
        self.episode_misses = []
        self.steps_since_eval = 0
        self.eval_results = []  # List of (step, results) tuples

    def _on_step(self) -> bool:
        # Track episode rewards/misses
        for info in self.locals.get("infos", []):
            if info:
                ep = info.get("episode")
                if ep:
                    self.episode_rewards.append(ep["r"])
                if "misses" in info and info.get("terminal_observation") is not None:
                    self.episode_misses.append(info["misses"])

        # Periodic evaluation on extreme tasksets
        self.steps_since_eval += 1
        if self.steps_since_eval >= self.eval_interval:
            print(
                f"\n  [Checkpoint] Step {self.num_timesteps}: evaluating on extreme tasksets..."
            )
            results = evaluate_on_extreme_tasksets(
                self.model, self.extreme_tasksets, eval_episodes=20
            )
            self.eval_results.append((self.num_timesteps, results))
            self.steps_since_eval = 0

        return True


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def smooth(data, window=50):
    """Smooth data using moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(callback, extreme_results, filename="training_extended.png"):
    """Plot training reward, misses, and extreme taskset performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Training reward
    ax = axes[0, 0]
    if callback.episode_rewards:
        ax.plot(smooth(callback.episode_rewards), linewidth=0.8, color="blue")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward Curve")
        ax.grid(True, alpha=0.3)

    # Panel 2: Training deadline misses
    ax = axes[0, 1]
    if callback.episode_misses:
        ax.plot(smooth(callback.episode_misses), linewidth=0.8, color="red")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Deadline Misses")
        ax.set_title("Deadline Misses Over Training")
        ax.grid(True, alpha=0.3)

    # Panel 3: Performance on extreme tasksets
    ax = axes[1, 0]
    if callback.eval_results:
        steps_list = []
        u120_misses = []
        u130_misses = []
        u140_misses = []
        u150_misses = []

        for step, results in callback.eval_results:
            steps_list.append(step / 1_000_000)  # Convert to millions
            u120_misses.append(results.get("U_1_20", {}).get("misses_mean", 0))
            u130_misses.append(results.get("U_1_30", {}).get("misses_mean", 0))
            u140_misses.append(results.get("U_1_40", {}).get("misses_mean", 0))
            u150_misses.append(results.get("U_1_50", {}).get("misses_mean", 0))

        ax.plot(steps_list, u120_misses, label="U=1.20", marker="o")
        ax.plot(steps_list, u130_misses, label="U=1.30", marker="s")
        ax.plot(steps_list, u140_misses, label="U=1.40", marker="^")
        ax.plot(steps_list, u150_misses, label="U=1.50", marker="d")
        ax.set_xlabel("Training Steps (millions)")
        ax.set_ylabel("Avg Deadline Misses")
        ax.set_title("Performance on Extreme Tasksets During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 4: Final comparison (standard tasksets)
    ax = axes[1, 1]
    if extreme_results:
        names = list(extreme_results.keys())
        u_values = [extreme_results[n]["utilization"] for n in names]
        misses = [extreme_results[n]["misses_mean"] for n in names]
        stds = [extreme_results[n]["misses_std"] for n in names]

        ax.bar(range(len(names)), misses, yerr=stds, capsize=5, color="steelblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"U={u:.2f}" for u in u_values], rotation=45)
        ax.set_ylabel("Avg Deadline Misses")
        ax.set_title("Final Performance on Extreme Tasksets")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_baseline_comparison(comparison_results, filename="comparison_extended.png"):
    """Bar chart comparing all schedulers on standard + extreme tasksets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Standard tasksets
    ax = axes[0]
    tasksets = ["Normal (U≈1.03)", "Stressed (U≈1.15)"]
    schedulers = ["PPO", "Round Robin", "RMS", "EDF", "LSF", "Budget Burn Rate"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#00BCD4"]

    x = np.arange(len(schedulers))
    width = 0.35

    for i, taskset_name in enumerate(tasksets):
        if taskset_name in comparison_results:
            data = comparison_results[taskset_name]
            misses = [data.get(s, {}).get("misses", 0) for s in schedulers]
            ax.bar(x + i * width, misses, width, label=taskset_name)

    ax.set_ylabel("Avg Deadline Misses")
    ax.set_title("Standard Tasksets")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Extreme tasksets
    ax = axes[1]
    extreme_names = ["U_1_20", "U_1_30", "U_1_40", "U_1_50"]
    u_labels = ["U=1.20", "U=1.30", "U=1.40", "U=1.50"]

    x = np.arange(len(schedulers))
    width = 0.15

    for i, extreme_name in enumerate(extreme_names):
        if extreme_name in comparison_results:
            data = comparison_results[extreme_name]
            misses = [data.get(s, {}).get("misses", 0) for s in schedulers]
            ax.bar(x + i * width, misses, width, label=u_labels[i])

    ax.set_ylabel("Avg Deadline Misses")
    ax.set_title("Extreme Tasksets")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ──────────────────────────────────────────────────────────────────────
# Main Training Loop with Checkpointing
# ──────────────────────────────────────────────────────────────────────


def find_latest_checkpoint():
    """Find the most recent checkpoint directory."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    checkpoints = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("step_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if checkpoints:
        latest = checkpoints[-1]
        return {
            "step": int(latest.split("_")[1]),
            "path": os.path.join(CHECKPOINT_DIR, latest),
        }
    return None


def save_checkpoint(model, step, metrics_log):
    """Save model and metrics to a checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save(os.path.join(ckpt_dir, "model"))
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"  Checkpoint saved: {ckpt_dir}")


def train_extended(resume=False):
    """Train with 4-stage curriculum, checkpointing, and periodic evaluation."""
    print("=" * 70)
    print("Extended RTOS Scheduler Training with Checkpointing")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    extreme_tasksets = create_extreme_tasksets()

    # Track overall training metrics
    metrics_log = {"stages": []}

    # Check for existing checkpoint
    latest_ckpt = find_latest_checkpoint() if resume else None
    if latest_ckpt:
        print(f"\nResuming from checkpoint at step {latest_ckpt['step']}")
        model = PPO.load(
            os.path.join(latest_ckpt["path"], "model"),
            env=None,
        )
        with open(os.path.join(latest_ckpt["path"], "metrics.json"), "r") as f:
            metrics_log = json.load(f)
        total_steps_done = latest_ckpt["step"]
        start_stage = len(metrics_log["stages"])
    else:
        print("\nStarting fresh training")
        model = None
        total_steps_done = 0
        start_stage = 0

    # Train through stages
    for stage_idx, stage_config in enumerate(TRAINING_STAGES[start_stage:]):
        abs_stage_idx = start_stage + stage_idx
        print(f"\n{'=' * 70}")
        print(
            f"Stage {abs_stage_idx + 1}/{len(TRAINING_STAGES)}: {stage_config['name']}"
        )
        print(f"  Description: {stage_config['description']}")
        print(f"  Steps: {stage_config['steps']:,}")
        print(f"  Utilization range: {stage_config['utilization_range']}")
        print(f"  WCET variability: {stage_config['wcet_variability'] * 100:.0f}%")
        print("=" * 70)

        # Create environment with variable execution support
        env = RandomRTOSEnv(
            utilization_range=stage_config["utilization_range"],
            max_ticks=300,
            **REWARD_KWARGS,
        )

        # Create or reload model
        ppo_kwargs = dict(
            policy_kwargs=dict(net_arch=[32, 32]),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            device="cpu",
            verbose=1,
        )

        if model is None:
            model = PPO("MlpPolicy", env, **ppo_kwargs)
        else:
            model.set_env(env)

        # Setup callback for periodic evaluation
        callback = TrainingMetricsCallback(
            extreme_tasksets, eval_interval=CHECKPOINT_INTERVAL
        )

        # Learn for this stage
        print(f"Training for {stage_config['steps']:,} steps...")
        model.learn(
            total_timesteps=stage_config["steps"],
            callback=callback,
            reset_num_timesteps=False,
        )

        # Save stage metrics
        stage_metrics = {
            "stage_name": stage_config["name"],
            "steps_completed": stage_config["steps"],
            "episode_rewards": [float(x) for x in callback.episode_rewards],
            "episode_misses": [float(x) for x in callback.episode_misses],
            "extreme_taskset_results": callback.eval_results,
        }
        metrics_log["stages"].append(stage_metrics)
        total_steps_done += stage_config["steps"]

        # Save checkpoint after each stage
        save_checkpoint(model, total_steps_done, metrics_log)

    # Final model save
    model.save(os.path.join(MODEL_DIR, "ppo_rtos_extended"))
    print(f"\nModel saved to {MODEL_DIR}/ppo_rtos_extended")

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")

    return model, metrics_log, extreme_tasksets


def evaluate_all_schedulers(model, extreme_tasksets):
    """Evaluate all schedulers on all tasksets."""
    print("\n" + "=" * 70)
    print("Final Evaluation: All Schedulers on All Tasksets")
    print("=" * 70)

    all_tasksets = {
        "Normal (U≈1.03)": NORMAL_TASKSET,
        "Stressed (U≈1.15)": STRESSED_TASKSET,
        "U_1_20": extreme_tasksets["U_1_20"],
        "U_1_30": extreme_tasksets["U_1_30"],
        "U_1_40": extreme_tasksets["U_1_40"],
        "U_1_50": extreme_tasksets["U_1_50"],
    }

    schedulers = {
        "PPO": None,
        "Round Robin": "rr",
        "RMS": "rms",
        "EDF": "edf",
        "LSF": "lsf",
        "Budget Burn Rate": "bbr",
    }

    comparison_results = {}

    for taskset_name, taskset in all_tasksets.items():
        u = compute_utilization(taskset)
        print(f"\n{taskset_name} (U={u:.2f}):")
        taskset_results = {}

        for scheduler_name, scheduler_fn in schedulers.items():
            if scheduler_fn is None:
                rews, misses = evaluate_ppo(model, RTOSEnv, taskset, EVAL_EPISODES)
            else:
                rews, misses = evaluate_baseline(
                    RTOSEnv, taskset, scheduler_fn, EVAL_EPISODES
                )

            taskset_results[scheduler_name] = {
                "misses": float(misses.mean()),
                "misses_std": float(misses.std()),
                "reward": float(rews.mean()),
                "reward_std": float(rews.std()),
            }

            print(
                f"  {scheduler_name:18s}: misses={misses.mean():.1f}±{misses.std():.1f}  "
                f"reward={rews.mean():.1f}±{rews.std():.1f}"
            )

        comparison_results[taskset_name] = taskset_results

    return comparison_results


# ──────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────


def main():
    resume = "--resume" in sys.argv

    # Train
    model, metrics_log, extreme_tasksets = train_extended(resume=resume)

    # Evaluate all schedulers
    comparison_results = evaluate_all_schedulers(model, extreme_tasksets)

    # Save comparison results
    with open("comparison_extended.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved comparison results to comparison_extended.json")

    # Create plots
    print("\nGenerating plots...")
    # Note: We'd need the callback object from training to plot curves
    # For now, focus on comparison plot
    plot_baseline_comparison(comparison_results)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
