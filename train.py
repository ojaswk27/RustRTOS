"""
Training script: trains PPO on the RTOS environment, evaluates against
three classical baseline schedulers, and generates comparison plots.

Baselines:
  - Round Robin: cycles through ready tasks in order
  - Rate Monotonic (RMS): always picks the ready task with the smallest period
  - Earliest Deadline First (EDF): picks the ready task with the nearest deadline

Usage: python train.py
Outputs: ppo_rtos_model/, training_reward.png, comparison.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rtos_env import RTOSEnv, NORMAL_TASKSET, STRESSED_TASKSET, IDLE_ACTION

MODEL_DIR = "ppo_rtos_model"
TRAIN_STEPS = 500_000
EVAL_EPISODES = 100


# ── Baseline schedulers ────────────────────────────────────────────────
# Each takes the env's task list and returns an action index.


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


# ── Evaluation helper ──────────────────────────────────────────────────


def evaluate_baseline(env_cls, taskset, scheduler_fn, episodes, max_ticks=300):
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
            else:
                action = edf(env.tasks, env.tick)
            obs, r, done, _, info = env.step(action)
            total_r += r
        rewards.append(total_r)
        misses.append(info["misses"])
    return np.array(rewards), np.array(misses)


def evaluate_ppo(model, env_cls, taskset, episodes, max_ticks=300):
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


# ── Training callback for logging ─────────────────────────────────────


class RewardLogger(BaseCallback):
    """Logs episode rewards during training for the reward curve plot."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_misses = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 auto-logs episode info when episodes end
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self.episode_rewards.append(ep["r"])
            if "misses" in info and info.get("terminal_observation") is not None:
                self.episode_misses.append(info["misses"])
        return True


# ── Plotting ───────────────────────────────────────────────────────────


def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training(logger, filename="training_reward.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if logger.episode_rewards:
        ax1.plot(smooth(logger.episode_rewards), linewidth=0.8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Training Reward Curve")

    if logger.episode_misses:
        ax2.plot(smooth(logger.episode_misses), linewidth=0.8, color="red")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Deadline Misses")
        ax2.set_title("Deadline Misses Over Training")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_comparison(results, filename="comparison.png"):
    """Bar chart comparing RL vs baselines on both tasksets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (taskset_name, data) in zip(axes, results.items()):
        names = list(data.keys())
        miss_means = [data[n]["misses"] for n in names]
        miss_stds = [data[n]["misses_std"] for n in names]

        x = np.arange(len(names))
        bars = ax.bar(
            x,
            miss_means,
            yerr=miss_stds,
            capsize=4,
            color=["#2196F3", "#FF9800", "#4CAF50", "#F44336"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Avg Deadline Misses / Episode")
        ax.set_title(f"{taskset_name}")

        for bar, v in zip(bars, miss_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    # Train on normal taskset
    print("Creating training environment...")
    env = RTOSEnv(taskset=NORMAL_TASKSET, max_ticks=300)

    print(f"Training PPO for {TRAIN_STEPS} steps...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[32, 32]),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        device="cpu",
        verbose=1,
    )
    logger = RewardLogger()
    model.learn(total_timesteps=TRAIN_STEPS, callback=logger)
    model.save(os.path.join(MODEL_DIR, "ppo_rtos"))
    print(f"Model saved to {MODEL_DIR}/")

    plot_training(logger)

    # Evaluate on both tasksets
    results = {}
    for name, taskset in [
        ("Normal (U≈1.03)", NORMAL_TASKSET),
        ("Stressed (U≈1.15)", STRESSED_TASKSET),
    ]:
        print(f"\nEvaluating on {name}...")
        data = {}
        for label, fn in [
            ("PPO", None),
            ("Round Robin", "rr"),
            ("RMS", "rms"),
            ("EDF", "edf"),
        ]:
            if fn is None:
                rews, miss = evaluate_ppo(model, RTOSEnv, taskset, EVAL_EPISODES)
            else:
                rews, miss = evaluate_baseline(RTOSEnv, taskset, fn, EVAL_EPISODES)
            data[label] = {
                "reward": rews.mean(),
                "reward_std": rews.std(),
                "misses": miss.mean(),
                "misses_std": miss.std(),
            }
            print(
                f"  {label:12s}: reward={rews.mean():.1f}±{rews.std():.1f}  "
                f"misses={miss.mean():.1f}±{miss.std():.1f}"
            )
        results[name] = data

    plot_comparison(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
