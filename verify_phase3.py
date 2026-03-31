"""
Verify Phase 3 results with extended episodes
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

def evaluate_policy(policy, env, taskset, n_episodes=10):
    """Evaluate a policy on a taskset with more detailed output."""
    env.taskset_cfg = taskset
    env.n_tasks = len(taskset)
    
    episode_data = []
    
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
        
        episode_data.append({
            "episode": ep,
            "reward": episode_reward,
            "misses": episode_misses,
            "steps": step
        })
    
    return episode_data

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
    
    # Test extreme stress only
    print("VERIFYING PHASE 3 RESULTS (10 episodes per condition)\n")
    
    conditions = [
        ("Extreme U=1.30", 1.30),
        ("Extreme U=1.50", 1.50),
    ]
    
    for name, utilization in conditions:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        
        taskset = create_extreme_taskset(utilization)
        print(f"Taskset: {taskset}\n")
        
        episodes = evaluate_policy(model, env, taskset, n_episodes=10)
        
        print(f"{'Ep':<5} {'Reward':<12} {'Misses':<10} {'Steps':<10}")
        print("-"*40)
        
        for ep_data in episodes:
            print(f"{ep_data['episode']:<5} {ep_data['reward']:>10.2f} {ep_data['misses']:>9.0f} {ep_data['steps']:>9}")
        
        misses_array = np.array([ep['misses'] for ep in episodes])
        print("\nSummary:")
        print(f"  Mean misses: {np.mean(misses_array):.2f}")
        print(f"  Std misses:  {np.std(misses_array):.2f}")
        print(f"  Min/Max:     {np.min(misses_array):.0f} / {np.max(misses_array):.0f}")

if __name__ == "__main__":
    main()
