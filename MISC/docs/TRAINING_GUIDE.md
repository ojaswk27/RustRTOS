# Extended Training Guide: High-Stress RTOS Scheduler

This guide explains how to run the extended training pipeline that will push your RL scheduler to beat classical schedulers under high stress (U > 1.20).

## Overview

The `train_extended.py` script provides:

1. **4-Stage Curriculum Learning** — Progressive difficulty:
   - Stage 1: Feasible tasksets (U = 0.60-0.95) with fixed WCET
   - Stage 2: Mixed overload (U = 0.85-1.15) with 15% WCET variability
   - Stage 3: Heavy overload (U = 1.10-1.35) with 30% WCET variability
   - Stage 4: Extreme stress (U = 1.25-1.50) with 40% WCET variability

2. **Resumable Checkpointing** — Save/restore at 100k-step intervals
   - Survive power cuts, interruptions, or system restarts
   - Track metrics and performance over time

3. **Periodic Extreme Taskset Evaluation** — Monitor progress on fixed ultra-overloaded tasksets
   - U = 1.20, 1.30, 1.40, 1.50 — benchmarks throughout training

4. **Advanced Baselines** — Compare against stronger schedulers:
   - **Least Slack First (LSF)** — (deadline - remaining) metric
   - **Budget Burn Rate (BBR)** — work-to-deadline urgency ratio
   - Plus classical: Round Robin, RMS, EDF

5. **Enhanced Reward Shaping**:
   - `urgency_weight=0.5` — Prioritize near-deadline tasks more aggressively
   - `completion_reward=2.0` — Reward finishing jobs
   - `miss_penalty=-3.0` — Penalize deadline misses heavily

## Quick Start

### First Run (From Scratch)

```bash
uv run python train_extended.py
```

**Expected output:**
- Progress logs every epoch
- Periodic evaluation on extreme tasksets (logged in checkpoints/)
- Final model: `ppo_rtos_extended_model/ppo_rtos_extended.zip`
- Metrics: `training_extended.json` + `comparison_extended.json`
- Plots: `training_extended.png` + `comparison_extended.png`

**Estimated time:** 8-12 hours wall-clock (5M training steps)

### Resume After Interruption

```bash
uv run python train_extended.py --resume
```

**What happens:**
- Detects latest checkpoint
- Loads model and metrics
- Continues from where it left off
- No data loss, no redundant training

## Directory Structure

```
RustRTOS/
├── train_extended.py           Main training script
├── checkpoints/
│   ├── step_100000/            Checkpoint after 100k steps
│   │   ├── model.zip
│   │   └── metrics.json
│   ├── step_200000/
│   │   ├── model.zip
│   │   └── metrics.json
│   └── ...
├── ppo_rtos_extended_model/
│   └── ppo_rtos_extended.zip   Final trained model
├── training_extended.json      Full training metrics (all stages)
├── comparison_extended.json    Final evaluation results (all schedulers)
├── training_extended.png       Training curves + extreme taskset performance
└── comparison_extended.png     Baseline comparison chart
```

## Interpreting Results

### Training Curves (training_extended.png)

- **Panel 1 (top-left):** Training reward — should increase and plateau
- **Panel 2 (top-right):** Deadline misses during training — should decrease
- **Panel 3 (bottom-left):** Performance on extreme tasksets — tracks U=1.20, 1.30, 1.40, 1.50 over time
- **Panel 4 (bottom-right):** Final distribution of misses per extreme taskset

### Comparison Results (comparison_extended.json)

For each taskset (Normal, Stressed, U_1_20, U_1_30, U_1_40, U_1_50):
- **PPO:** Your RL scheduler
- **Round Robin, RMS, EDF:** Classical baselines
- **LSF, Budget Burn Rate:** Stronger heuristics

Look for:
- PPO misses < EDF/RMS misses on extreme tasksets ✓ (goal!)
- LSF/BBR performance to understand the "smart heuristic ceiling"

### Metrics File (training_extended.json)

```json
{
  "stages": [
    {
      "stage_name": "Feasible",
      "steps_completed": 1000000,
      "episode_rewards": [...],
      "episode_misses": [...],
      "extreme_taskset_results": [
        {
          "step": 100000,
          "U_1_20": {"misses_mean": 8.2, "misses_std": 0.5, ...},
          ...
        },
        ...
      ]
    },
    ...
  ]
}
```

## Advanced: Understanding the Curriculum

### Why 4 Stages?

**Stage 1 (Feasible)**: 
- U < 1.0; all tasks *can* meet deadlines
- Agent learns basic urgency ordering (which task is more time-critical)
- Clean signal: more work = higher priority

**Stage 2 (Mixed)**:
- U crosses 1.0 boundary; some misses are inevitable
- Agent learns tradeoffs (can't do everything; which to prioritize?)
- Small WCET variability (±15%) adds realism

**Stage 3 (Overloaded)**:
- U = 1.10-1.35; heavy overload
- Agent learns aggressive prioritization under pressure
- Moderate WCET variability (±30%)

**Stage 4 (Extreme)**:
- U = 1.25-1.50; extreme stress where classical schedulers degrade
- Agent specializes on handling severe overload
- High WCET variability (±40%) tests robustness

### WCET Variability

Tasks in real systems don't always take their WCET (Worst-Case Execution Time). The `variable_exec=True` flag makes each task release have a random actual execution time in [BCET, WCET]:
- BCET = max(1, WCET // 2)
- Actual execution = uniform random in [BCET, WCET]

This forces the agent to learn *adaptive* scheduling, not just static priorities.

## Monitoring Training

### Check Training Progress in Real-Time

```bash
# Watch the latest checkpoint
ls -lh checkpoints/step_*/metrics.json | tail -5

# Or inspect JSON
cat checkpoints/step_500000/metrics.json | jq '.stages[-1].extreme_taskset_results[-1]'
```

### Key Metrics to Track

As training progresses, look for:

1. **Reward increasing** — agent learning to maximize reward
2. **Deadline misses decreasing** — especially on extreme tasksets
3. **Performance on U_1_50 improving** — the hardest benchmark
4. **Converging by Stage 4** — by end of training, should stabilize

## Next Steps After Training

1. **Phase 2: Analysis** — Reverse-engineer learned heuristics
   - Which tasksets cause most misses? Why?
   - Does PPO learn task importance?
   - How sensitive to stress level (U)?

2. **Phase 3: Refinement** (if needed)
   - Tweak reward function based on failure modes
   - Add global state features if useful

3. **Phase 4: Validation**
   - Export weights to Rust (Q10 fixed-point)
   - Run extended simulation (3000+ ticks)
   - Test on real hardware

## Troubleshooting

### Script won't start?

```bash
# Ensure uv environment is fresh
uv sync --refresh

# Verify dependencies
uv run python -c "from stable_baselines3 import PPO; import gymnasium; import numpy"
```

### Running out of memory?

Reduce `batch_size` in `ppo_kwargs` (currently 64). Try 32:
```python
ppo_kwargs = dict(batch_size=32, ...)
```

### Too slow?

- Training with CPU is correct for this small network (32-32)
- If using GPU, ensure CUDA/PyTorch is properly configured
- Can parallelize by running multiple seeds on separate hardware

### Checkpointing not working?

```bash
# Verify checkpoint directory exists
mkdir -p checkpoints
ls -la checkpoints/
```

If corrupted, delete bad checkpoint and resume from previous:
```bash
rm -rf checkpoints/step_500000/  # Delete bad checkpoint
uv run python train_extended.py --resume  # Will load step_400000
```

## Questions?

If something unexpected happens:
1. Check the console output for error messages
2. Inspect the latest metrics file: `checkpoints/step_*/metrics.json`
3. Verify taskset utilization: `compute_utilization(taskset)` should match label

---

**You're ready!** Start training with:
```bash
uv run python train_extended.py
```

It will run for ~8-12 hours and automatically checkpointevery 100k steps. You can pause/resume freely.
