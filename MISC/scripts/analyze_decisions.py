#!/usr/bin/env python3
"""
Simplified Decision Analysis: PPO vs Classical Schedulers
==========================================================

Analyzes PPO scheduling decisions and compares them to RMS/EDF.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Suppress warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys

sys.path.insert(0, "/Volumes/Spare/PycharmProjects/OS project/RustRTOS")

from rtos_env import RTOSEnv, NORMAL_TASKSET
from stable_baselines3 import PPO


def load_ppo_model(model_path):
    """Load trained PPO model."""
    print(f"Loading PPO model...")
    model = PPO.load(model_path)
    return model


def run_with_policy(policy, env, taskset, n_episodes=5):
    """Run episodes with the policy and record decisions."""
    env.taskset_cfg = taskset
    env.n_tasks = len(taskset)

    all_decisions = []
    all_rewards = []
    all_misses = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_decisions = []
        episode_reward = 0
        episode_misses = 0

        done = False
        step = 0
        while not done and step < 300:
            action, _ = policy.predict(obs, deterministic=True)
            episode_decisions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_misses += info.get("deadline_miss", 0)
            step += 1

        all_decisions.extend(episode_decisions)
        all_rewards.append(episode_reward)
        all_misses.append(episode_misses)

    return {
        "decisions": np.array(all_decisions),
        "rewards": np.array(all_rewards),
        "misses": np.array(all_misses),
        "mean_reward": np.mean(all_rewards),
        "mean_misses": np.mean(all_misses),
        "std_reward": np.std(all_rewards),
        "std_misses": np.std(all_misses),
    }


def create_extreme_taskset(utilization_target, base_periods=None):
    """Create a taskset with target utilization."""
    if base_periods is None:
        base_periods = [25, 35, 45, 50, 60, 75]

    n = len(base_periods)
    wcet_target = utilization_target / n

    taskset = []
    np.random.seed(42)

    for period in base_periods:
        wcet = int(np.maximum(1, wcet_target * period))
        taskset.append((period, period, wcet))

    return taskset


def analyze_decisions(decisions, taskset, name):
    """Analyze decision patterns."""
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {name}")
    print(f"{'=' * 70}\n")

    periods = np.array([t[0] for t in taskset])
    wcets = np.array([t[2] for t in taskset])

    # Decision frequency
    decision_counts = np.bincount(decisions, minlength=len(taskset))
    decision_freq = decision_counts / len(decisions)

    print("Task Selection Frequency:\n")
    print(f"{'Task':<8} {'Period':<10} {'WCET':<10} {'Freq %':<10} {'RMS?':<8}")
    print("-" * 50)

    rms_task = np.argmin(periods)
    for task_id in range(len(taskset)):
        freq = decision_freq[task_id] * 100
        is_rms = "YES" if task_id == rms_task else "NO"
        print(
            f"{task_id:<8} {periods[task_id]:<10} {wcets[task_id]:<10} {freq:<10.1f} {is_rms:<8}"
        )

    most_selected = np.argmax(decision_counts)

    print(f"\nMost-selected task: {most_selected} (Period={periods[most_selected]})")
    print(f"RMS preferred task: {rms_task} (Period={periods[rms_task]})")

    if most_selected == rms_task:
        print("✓ PPO ALIGNS with RMS on primary choice")
        strategy = "ALIGNED"
    else:
        print("✗ PPO DIVERGES from RMS on primary choice")
        strategy = "DIVERGED"

    return {
        "most_selected": most_selected,
        "rms_preferred": rms_task,
        "decision_freq": decision_freq,
        "strategy": strategy,
    }


def visualize_decisions(results, output_file):
    """Visualize decision patterns across stress levels."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "PPO Decision Patterns Across Stress Levels", fontsize=16, fontweight="bold"
    )

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]

        decisions = data["decisions"]
        taskset = data["taskset"]
        analysis = data["analysis"]

        periods = np.array([t[0] for t in taskset])
        decision_counts = np.bincount(decisions, minlength=len(taskset))
        # Ensure decision_counts has exactly the right length (trim if necessary)
        decision_counts = decision_counts[: len(taskset)]

        # Bar chart
        rms_task = analysis["rms_preferred"]
        colors = ["red" if i == rms_task else "skyblue" for i in range(len(taskset))]

        bars = ax.bar(
            range(len(taskset)),
            decision_counts,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Task ID", fontweight="bold")
        ax.set_ylabel("Selection Count", fontweight="bold")
        ax.set_title(f"{name}\n({analysis['strategy']})", fontweight="bold")
        ax.set_xticks(range(len(taskset)))
        ax.grid(True, alpha=0.3, axis="y")

        # Add period labels
        for i, (count, period) in enumerate(zip(decision_counts, periods)):
            ax.text(
                i, count + 5, f"P={period}", ha="center", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved: {output_file}")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("PHASE 2: DEEP DECISION ANALYSIS - PPO vs RMS")
    print("=" * 80 + "\n")

    # Load model
    model_path = "/Volumes/Spare/PycharmProjects/OS project/RustRTOS/ppo_rtos_extended_model/ppo_rtos_extended"
    ppo_model = load_ppo_model(model_path)

    # Create environment
    env = RTOSEnv(
        max_ticks=300,
        completion_reward=2.0,
        miss_penalty=-3.0,
        context_switch_penalty=0.0,
        urgency_weight=0.5,
        variable_exec=True,
    )

    # Test conditions
    conditions = [
        ("Normal (U≈1.03)", 1.03),
        ("Stressed (U≈1.15)", 1.15),
        ("Extreme U=1.20", 1.20),
        ("Extreme U=1.30", 1.30),
        ("Extreme U=1.40", 1.40),
        ("Extreme U=1.50", 1.50),
    ]

    results = {}

    for name, utilization in conditions:
        print(f"\n{'#' * 80}")
        print(f"# {name}")
        print(f"{'#' * 80}")

        # Create taskset
        taskset = create_extreme_taskset(utilization)

        # Run with PPO
        ppo_data = run_with_policy(ppo_model, env, taskset, n_episodes=3)

        print(f"\nPPO Performance:")
        print(
            f"  Mean reward: {ppo_data['mean_reward']:.2f} ± {ppo_data['std_reward']:.2f}"
        )
        print(
            f"  Mean misses: {ppo_data['mean_misses']:.2f} ± {ppo_data['std_misses']:.2f}"
        )

        # Analyze decisions
        analysis = analyze_decisions(ppo_data["decisions"], taskset, name)

        results[name] = {
            "utilization": utilization,
            "taskset": taskset,
            "decisions": ppo_data["decisions"],
            "ppo_data": ppo_data,
            "analysis": analysis,
        }

    # Visualize
    visualize_decisions(results, "decision_analysis_comparison.png")

    # Summary report
    print("\n\n" + "=" * 80)
    print("SUMMARY: PPO vs RMS STRATEGY ALIGNMENT")
    print("=" * 80 + "\n")

    print(
        f"{'Condition':<25} {'U':<10} {'PPO Primary':<15} {'RMS Primary':<15} {'Align':<10}"
    )
    print("-" * 75)

    for name, data in results.items():
        analysis = data["analysis"]
        taskset = data["taskset"]
        periods = np.array([t[0] for t in taskset])

        ppo_primary = analysis["most_selected"]
        rms_primary = analysis["rms_preferred"]

        align = "YES ✓" if ppo_primary == rms_primary else "NO ✗"

        print(
            f"{name:<25} {data['utilization']:<10.2f} Task {ppo_primary} (P={periods[ppo_primary]:<3}) Task {rms_primary} (P={periods[rms_primary]:<3}) {align:<10}"
        )

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    aligned_count = sum(
        1 for r in results.values() if r["analysis"]["strategy"] == "ALIGNED"
    )
    total_count = len(results)

    print(
        f"\nPPO aligns with RMS on primary choice: {aligned_count}/{total_count} conditions"
    )

    if aligned_count == total_count:
        print("\n✓ STRATEGY INSIGHT: PPO learns RMS's period-based prioritization!")
        print("  If PPO still loses, the problem is in EXECUTION, not STRATEGY.")
        print("  • PPO may struggle with variable execution times")
        print("  • PPO may not have learned optimal prioritization weights")
        print("  • Reward function may need tuning for extreme stress conditions")
    else:
        print("\n✗ STRATEGY INSIGHT: PPO doesn't consistently learn RMS heuristic.")
        print("  The core scheduling strategy needs improvement.")
        print("  • Add explicit reward bonus for selecting short-period tasks")
        print("  • Strengthen urgency weights for deadline-critical situations")
        print("  • Consider curriculum training emphasis on period-based priority")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
