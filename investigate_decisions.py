"""
Investigate if PPO's divergent decisions at U=1.15 and U=1.40 are actually beneficial
"""
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

def analyze_with_custom_policy(env, taskset, policy_name, force_task=None, n_episodes=3):
    """Run episodes with a custom policy."""
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
        
        if force_task is not None:
            # Force selecting a specific task every time
            while not done and step < 300:
                action = force_task
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_misses += info.get("deadline_miss", 0)
                step += 1
        
        all_misses.append(episode_misses)
        all_rewards.append(episode_reward)
    
    return {
        "mean_misses": np.mean(all_misses),
        "std_misses": np.std(all_misses),
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
    }

def main():
    # Create environment
    env = RTOSEnv(
        max_ticks=300,
        completion_reward=2.0,
        miss_penalty=-3.0,
        context_switch_penalty=0.0,
        urgency_weight=0.5,
        variable_exec=True,
    )
    
    # Load PPO model
    model = PPO.load("/Volumes/Spare/PycharmProjects/OS project/RustRTOS/ppo_rtos_extended_model/ppo_rtos_extended")
    
    # Test conditions where PPO diverges from RMS
    test_cases = [
        ("U=1.15 (PPO prefers Task 5, RMS prefers Task 0)", 1.15),
        ("U=1.40 (PPO prefers Task 1, RMS prefers Task 0)", 1.40),
    ]
    
    for name, util in test_cases:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}\n")
        
        taskset = create_extreme_taskset(util)
        periods = np.array([t[0] for t in taskset])
        
        # Analyze if forcing RMS choice (Task 0) vs forcing PPO preference helps
        ppo_result = analyze_with_custom_policy(env, taskset, "PPO_choice", force_task=5 if util == 1.15 else 1, n_episodes=3)
        rms_result = analyze_with_custom_policy(env, taskset, "RMS_choice", force_task=0, n_episodes=3)
        
        print(f"{'Policy':<20} {'Mean Misses':<15} {'Mean Reward':<15}")
        print("-"*50)
        print(f"{'Force Task 0 (RMS)':<20} {rms_result['mean_misses']:>13.2f} {rms_result['mean_reward']:>13.2f}")
        if util == 1.15:
            print(f"{'Force Task 5 (PPO)':<20} {ppo_result['mean_misses']:>13.2f} {ppo_result['mean_reward']:>13.2f}")
        else:
            print(f"{'Force Task 1 (PPO)':<20} {ppo_result['mean_misses']:>13.2f} {ppo_result['mean_reward']:>13.2f}")
        
        improvement = rms_result['mean_misses'] - ppo_result['mean_misses']
        print(f"\nPPO's preference is {'BETTER' if improvement > 0 else 'WORSE'} by {abs(improvement):.2f} misses")

if __name__ == "__main__":
    main()
