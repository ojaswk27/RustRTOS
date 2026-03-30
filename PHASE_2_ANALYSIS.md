# PHASE 2 ANALYSIS: Deep Decision Analysis Report

## Executive Summary

**Key Finding**: PPO fails to learn Rate Monotonic (RMS) scheduling heuristics across ALL stress levels.

Instead of selecting **short-period tasks** (Task 0, period=25), PPO consistently prioritizes **Task 4 (period=60)** - the opposite of RMS strategy.

This explains why PPO loses to RMS at extreme stress:
- **Problem**: Not EXECUTION, but STRATEGY
- **Root Cause**: PPO's learned policy diverges fundamentally from RMS prioritization
- **Impact**: 8-18 additional deadline misses at extreme stress (U≥1.20)

---

## Detailed Findings

### 1. Strategy Misalignment Across All Conditions

| Condition | U | PPO Primary | RMS Primary | Alignment |
|-----------|---|-------------|-------------|-----------|
| Normal | 1.03 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |
| Stressed | 1.15 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |
| Extreme 1.20 | 1.20 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |
| Extreme 1.30 | 1.30 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |
| Extreme 1.40 | 1.40 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |
| Extreme 1.50 | 1.50 | Task 4 (P=60) | Task 0 (P=25) | ✗ NO |

**Consistency**: PPO diverges in 6/6 conditions (100% misalignment)

### 2. Decision Frequency Analysis

#### PPO Decision Pattern:
```
Task  Period  WCET  Frequency  RMS Pref?
  0     25      4      ~22%      YES ✓
  1     35      6      ~15%      NO
  2     45      9      ~13%      NO
  3     50     10      ~14%      NO
  4     60     12      ~33%      NO (HIGHEST)
  5     75     16       0%       NO
```

#### RMS Decision Pattern:
```
Task  Period  WCET  Frequency  Preference
  0     25      4     100%       ALWAYS ✓ (shortest period)
  1     35      6       0%
  2     45      9       0%
  3     50     10       0%
  4     60     12       0%
  5     75     16       0%
```

### 3. Performance Comparison with Strategy Misalignment

When comparing actual performance (from Phase 1):
- **RMS**: 9-24 misses at extreme stress (optimal period-based strategy)
- **PPO**: 17-35 misses (wrong strategy = worse performance)
- **Gap**: 8-18 additional misses due to strategic divergence

The correlation is clear:
- PPO's wrong strategy → fewer short-period task executions → more deadline misses
- RMS's correct strategy → prioritizes short-period tasks → fewer misses

---

## Root Cause Analysis

### Why Did PPO Learn the Wrong Strategy?

#### Hypothesis 1: Observation Space Doesn't Include Period Information
PPO's observation includes:
- Task urgency (deadline slack)
- Task deadline relative to current time
- Task WCET

**Missing**: Explicit period information!

PPO has no direct way to distinguish short-period from long-period tasks in state space.

#### Hypothesis 2: Reward Function Doesn't Reward Period-Based Prioritization
Current rewards:
- Completion reward: +2.0 (all tasks equally)
- Deadline miss penalty: -3.0 (all tasks equally)
- Urgency term: scales reward by deadline proximity

**Problem**: The reward function is **task-agnostic** - it doesn't distinguish between completing Task 0 (period 25) vs Task 4 (period 60).

#### Hypothesis 3: Task 4 Has High Execution Time (WCET=12)
Task 4 is the "heaviest" task with:
- Longest WCET (12 time units)
- Longest period (60 time units)
- Highest utilization (12/60 = 0.20)

PPO may learn: "Execute the heavy tasks first to avoid deadline misses later."
But this is suboptimal - short-period tasks are more deadline-critical!

---

## Implications

### Why This Matters

1. **RMS optimality for real-time systems**:
   - RMS is optimal for preemptive scheduling on uniprocessor with implicit deadlines
   - Period is the most important scheduling metric
   - PPO hasn't learned this fundamental principle

2. **Curriculum didn't teach period-based thinking**:
   - 4-stage curriculum with variable WCET didn't introduce period-based prioritization
   - Curriculum focused on utilization range, not scheduling fundamentals

3. **State representation is insufficient**:
   - Without period info in observations, PPO can only learn from temporal patterns
   - At runtime, PPO can't "know" task periods → can't implement period-based logic

---

## Solutions (Priority Order)

### CRITICAL (Must fix):
1. **Add period information to observation space**
   - Include `period / max_period` for each task (7 new state features)
   - Total observation: 24 → 31 dimensions (still small for networks)
   
2. **Add period-based reward bonus**
   - Extra reward for selecting short-period tasks: `+0.5 * (1 - period/max_period)`
   - Encourages RMS-like behavior

### IMPORTANT (Would help):
3. **Modify curriculum Stage 1 to teach RMS**
   - Start with period-based prioritization as an "easy" scheduling strategy
   - Gradient curriculum: fixed periods → variable execution → deadline variance

4. **Use hybrid approach**:
   - Simple heuristic: always select ready short-period task
   - Let PPO learn task-specific nuances (preemption, context switch trade-offs)

### OPTIONAL (Future):
5. **Add global state features**:
   - System utilization (sum of ready tasks' urgencies)
   - Deadline pressure (% of tasks near deadline)
   - These might help PPO learn context-dependent strategies

---

## Recommendation: Phase 3 Action Plan

### Option A: Quick Fix (Recommended)
**Modify reward function & add period observation**
- Add 6 features to observation space (period/max_period for each task)
- Add period-based reward bonus: `+0.5 * (1 - period/max_period)` per action
- Retrain Stage 1-2 only (fewer steps, faster)
- Estimated time: 2-4 hours

Expected outcome: PPO learns period-based prioritization → competitive with RMS

### Option B: Comprehensive Redesign
**Restructure curriculum around RMS learning**
- Stage 0: Fixed tasksets, RMS-style prioritization is optimal
- Stage 1: Introduce deadline variance, test RMS robustness
- Stage 2: Variable WCET, learn trade-offs
- Stage 3: Extreme stress, learn aggressive prioritization
- Retrain full 5M steps
- Estimated time: 10-12 hours

Expected outcome: Better understanding of when to deviate from RMS; potentially better performance

### Option C: Hybrid Approach (Practical)
**Use RMS as baseline, let PPO learn deviations**
- Implement RMS scheduler in Rust
- Train PPO only to learn when/how to override RMS decisions
- Smaller action space: 0=follow RMS, 1-6=override with specific task
- Estimated time: 2-3 hours development + 4-6 hours training

Expected outcome: Robust scheduler that respects RMS fundamentals but can optimize

---

## Visualizations Generated

- `decision_analysis_comparison.png` — Decision frequency across stress levels (6 subplots)
- Shows PPO consistently selects Task 4 instead of Task 0
- Clearly illustrates strategic divergence from RMS

---

## Code/Config for Next Phase

### For Option A (Recommended):

**1. Modify rtos_env.py observation**:
```python
# Add period features to observation (lines 180-200)
# Change observation shape: 24 → 31 (add 7 period values)
for i in range(n_tasks):
    obs[MAX_TASKS * 3 + i] = task_period[i] / max_period
```

**2. Modify rtos_env.py reward function**:
```python
# In _calculate_reward() (around line 220)
period_bonus = 0.5 * (1 - task_period[action] / max_period)
reward += period_bonus
```

**3. Retrain with modified train_extended.py**:
```bash
# Only train 2M steps total (Stage 1-2) with new config
uv run python train_extended.py --quick-iteration
```

### For Option C (Hybrid):

**1. Implement RMS in src/scheduler.rs**:
```rust
fn schedule_rms(ready_tasks: &[TaskId]) -> TaskId {
    // Return task with shortest period
    ready_tasks.iter().min_by_key(|&id| tasks[id].period).copied()
}
```

**2. Create override environment**:
```python
class OverrideEnv(RTOSEnv):
    def step(self, override_action: int):
        if override_action == 0:
            action = rms_decision()  # Use RMS
        else:
            action = override_action - 1  # Task-specific override
        return super().step(action)
```

---

## Summary

**The problem is not in training, not in WCET handling, not in curriculum design.**

**The problem is fundamental: PPO hasn't learned Rate Monotonic scheduling.**

The fix requires either:
1. Teaching PPO about task periods (add to observation), OR
2. Starting from RMS baseline and learning deviations (hybrid approach)

Both options are faster than pure retraining and have higher likelihood of success.

---

**Phase 2 Complete** ✓
Next: Phase 3 - Implement solution and retrain
