"""
Evaluate Phase 3 model and create comparison with Phase 1 results
"""
import json
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rtos_env import RTOSEnv
from stable_baselines3 import PPO

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

def evaluate_policy(policy, env, taskset, scheduler_name, n_episodes=3):
    """Evaluate a policy on a taskset."""
    env.taskset_cfg = taskset
    env.n_tasks = len(taskset)
    
    all_misses = []
    all_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_misses = 0
        
        done = False
        step = 0
        while not done and step < 300:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_misses += info.get("deadline_miss", 0)
            step += 1
        
        all_misses.append(episode_misses)
        all_rewards.append(episode_reward)
    
    return {
        "misses": np.mean(all_misses),
        "misses_std": np.std(all_misses),
        "reward": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
    }

def main():
    # Load Phase 3 model
    model = PPO.load("/Volumes/Spare/PycharmProjects/OS project/RustRTOS/ppo_rtos_extended_model/ppo_rtos_extended")
    
    # Create environment
    env = RTOSEnv(
        max_ticks=300,
        completion_reward=2.0,
        miss_penalty=-3.0,
        context_switch_penalty=0.0,
        urgency_weight=0.5,
        variable_exec=True,
    )
    
    # Mapping of condition names to JSON keys in phase 1
    conditions = [
        ("Normal (U≈1.03)", 1.03, "Normal (U≈1.03)"),
        ("Stressed (U≈1.15)", 1.15, "Stressed (U≈1.15)"),
        ("Extreme U=1.20", 1.20, "U_1_20"),
        ("Extreme U=1.30", 1.30, "U_1_30"),
        ("Extreme U=1.40", 1.40, "U_1_40"),
        ("Extreme U=1.50", 1.50, "U_1_50"),
    ]
    
    # Load Phase 1 results for comparison
    with open("comparison_extended.json") as f:
        phase1 = json.load(f)
    
    # Create Phase 3 results
    phase3 = {}
    
    for name, utilization, p1_key in conditions:
        print(f"Evaluating {name}...")
        taskset = create_extreme_taskset(utilization)
        
        ppo_result = evaluate_policy(model, env, taskset, "PPO")
        
        phase3[p1_key] = {
            "PPO": ppo_result,
        }
        
        print(f"  PPO: {ppo_result['misses']:.1f} misses, {ppo_result['reward']:.2f} reward")
    
    # Save Phase 3 results
    with open("comparison_phase3.json", "w") as f:
        json.dump(phase3, f, indent=2)
    
    # Create comparison report
    print("\n" + "="*90)
    print("PHASE 3 vs PHASE 1 COMPARISON")
    print("="*90 + "\n")
    
    print(f"{'Condition':<25} {'Phase 1 (PPO)':<20} {'Phase 3 (PPO)':<20} {'Improvement':<20}")
    print("-"*90)
    
    for name, utilization, p1_key in conditions:
        p1_misses = phase1[p1_key]["PPO"]["misses"]
        p3_misses = phase3[p1_key]["PPO"]["misses"]
        
        improvement = p1_misses - p3_misses
        pct_change = (improvement / p1_misses * 100) if p1_misses > 0 else 0
        
        print(f"{name:<25} {p1_misses:>6.1f} misses      {p3_misses:>6.1f} misses      {improvement:>+6.1f} ({pct_change:>+6.1f}%)")
    
    # Summary stats
    print("\n" + "="*90)
    total_p1 = sum(phase1[p1_key]["PPO"]["misses"] for _, _, p1_key in conditions)
    total_p3 = sum(phase3[p1_key]["PPO"]["misses"] for _, _, p1_key in conditions)
    
    print(f"Total misses - Phase 1: {total_p1:.0f}")
    print(f"Total misses - Phase 3: {total_p3:.0f}")
    print(f"Overall improvement: {total_p1 - total_p3:.0f} misses ({(total_p1 - total_p3) / total_p1 * 100:+.1f}%)")

if __name__ == "__main__":
    main()
