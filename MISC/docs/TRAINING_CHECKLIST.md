# Training Progress Checklist

Use this checklist to track your training journey from Phase 1 → Phase 4.

## Phase 1: Extended Training (8-12 hours)

### Pre-Training
- [ ] Read `START_HERE.txt` for overview
- [ ] Read `PHASE_0_1_READY.md` for quick start
- [ ] Read `TRAINING_GUIDE.md` for detailed guide
- [ ] Verify disk space: Need ~2-3 GB for checkpoints + model
- [ ] Ensure internet stable (no interruptions expected, but resumable if needed)

### During Training
- [ ] Training starts: `uv run python train_extended.py`
- [ ] Stage 1 starts (1M steps, U=0.60-0.95)
  - [ ] Checkpoint at step_100000 (~2.5 hours)
  - [ ] Checkpoint at step_200000 (~5 hours)
  - [ ] Checkpoint at step_300000 (~7.5 hours)
  - [ ] ... through step_1000000 (~25 hours cumulative)
- [ ] Stage 2 starts (1.5M steps, U=0.85-1.15, 15% WCET variability)
  - [ ] Checkpoints through step_2500000
- [ ] Stage 3 starts (1.5M steps, U=1.10-1.35, 30% WCET variability)
  - [ ] Checkpoints through step_4000000
- [ ] Stage 4 starts (1M steps, U=1.25-1.50, 40% WCET variability) — **THE CRITICAL STAGE**
  - [ ] Checkpoints through step_5000000

### After Training
- [ ] Training completes: Final evaluation on all tasksets
- [ ] Check outputs created:
  - [ ] `ppo_rtos_extended_model/ppo_rtos_extended.zip` exists
  - [ ] `training_extended.json` exists
  - [ ] `comparison_extended.json` exists
  - [ ] `comparison_extended.png` exists
- [ ] No errors in final output

### Quick Quality Check
```bash
# Verify training completed
ls -lh ppo_rtos_extended_model/ppo_rtos_extended.zip

# Check final comparison results
cat comparison_extended.json | jq '.' | head -50

# View final stats
cat training_extended.json | jq '.stages[-1]' | head -30
```

---

## Phase 2: Analysis (2-3 hours)

### Understanding Results
- [ ] Read `comparison_extended.json` carefully
- [ ] Check PPO performance:
  - [ ] Normal (U≈1.03): _____ misses (baseline: 4.0)
  - [ ] Stressed (U≈1.15): _____ misses (baseline: 7.0)
  - [ ] U=1.20: _____ misses (target: < 6.0)
  - [ ] U=1.30: _____ misses (target: < 8.0)
  - [ ] U=1.40: _____ misses (target: < 10.0)
  - [ ] U=1.50: _____ misses (target: < 12.0)

### Compare Against Baselines
For U=1.40 and U=1.50:
- [ ] PPO vs Round Robin: PPO better? YES / NO
- [ ] PPO vs RMS: PPO better? YES / NO
- [ ] PPO vs EDF: PPO better/equal? YES / NO
- [ ] PPO vs LSF: Gap? _____ misses
- [ ] PPO vs BBR: Gap? _____ misses

### Analysis Tasks
- [ ] Create decision boundary visualization
  - [ ] Script created: `analyze_decisions.py`
  - [ ] Heatmap generated: `policy_decisions.png`
  
- [ ] Task-level miss analysis
  - [ ] Which task misses most? Task _____
  - [ ] Which task never misses? Task _____
  - [ ] Fair distribution? YES / NO
  
- [ ] Identify failure modes
  - [ ] Biggest weakness: _____________________
  - [ ] Strongest point: _____________________
  
- [ ] Sensitivity analysis
  - [ ] Performance variance across seeds: HIGH / MEDIUM / LOW
  - [ ] Robustness to WCET variability: GOOD / FAIR / POOR

### Key Questions Answered
- [ ] Why does PPO perform better on high-stress?
- [ ] What learned heuristics matter most?
- [ ] Where does PPO differ from classical schedulers?
- [ ] Are there obvious failure modes to fix?

---

## Phase 3: Refinement (0-4 hours, conditional)

**Only do this if Phase 2 reveals actionable improvements.**

### Option A: Reward Function Tweaking
- [ ] Identified sub-optimal reward signals? YES / NO
- [ ] New reward configuration designed
- [ ] Retrain Stage 3-4 with new rewards
  - [ ] Training started: `uv run python train_extended.py --resume`
  - [ ] Reached new checkpoint: step_5000000

### Option B: State Representation Enhancement
- [ ] Need global context features? YES / NO
- [ ] New features identified: _____________________
- [ ] State vector expanded from 24 to _____ dimensions
- [ ] Retrain full pipeline (5M steps)

### Option C: No Changes Needed
- [ ] Phase 2 analysis shows good performance
- [ ] Skip refinement, proceed to Phase 4

### Decision: Which option?
**Choice: A / B / C**

Rationale: _________________________________________________________________

---

## Phase 4: Validation (1-2 hours)

### Export to Rust
- [ ] Generate Q10 fixed-point weights
  - [ ] Script: `uv run python export_weights.py`
  - [ ] File created: `policy_weights.json`
  
- [ ] Embed in Rust binary
  - [ ] Copy weights to `src/policy.rs`
  - [ ] Verify no overflow/saturation issues
  
- [ ] Build Rust binary
  - [ ] `cargo build --release` succeeds
  - [ ] Binary size: _____ KB

### Extended Rust Simulation
- [ ] Run 3000-tick simulation (10 hyperperiods)
  - [ ] Modify `main.rs`: `sched.run(3000)`
  - [ ] `cargo run --release` completes
  - [ ] Output shows reasonable metrics

- [ ] Verify numerical precision
  - [ ] No saturation warnings
  - [ ] Results match Python (within rounding)

### Power/Energy Analysis
- [ ] Measure inference time
  - [ ] Per-tick NN execution: _____ μs
  - [ ] Percentage of 1ms tick: _____ %
  
- [ ] Compare to context switch cost
  - [ ] Switch cost vs NN cost ratio: _____
  - [ ] Reasonable tradeoff? YES / NO

### Real Hardware (Optional)
- [ ] Test on actual Cortex-M4 board
  - [ ] Hardware: _____________________
  - [ ] Performance matches simulation: YES / NO
  - [ ] Any timing surprises? _____________________

---

## Success Metrics

### Minimum (Phase 1-2)
- [ ] Model trains to completion
- [ ] PPO beats RMS on U ≥ 1.20
- [ ] PPO approaches EDF on U ≥ 1.40

### Target (Phase 1-3)
- [ ] PPO beats EDF on U = 1.40
- [ ] PPO matches or beats EDF on U = 1.50
- [ ] Performance stable across checkpoints

### Excellent (Phase 1-4)
- [ ] All above
- [ ] Validated on Rust (QEMU)
- [ ] Analyzed and understood learned heuristics
- [ ] Ready for RTOS kernel integration

---

## Problem Solving

### Training takes too long?
- [ ] Reduce `n_epochs` from 10 to 5 in train_extended.py
- [ ] Reduce `batch_size` from 64 to 32
- [ ] Use GPU if available: `device="cuda"`

### Memory issues?
- [ ] Reduce `batch_size` to 32
- [ ] Reduce `n_steps` from 2048 to 1024

### Checkpoints slow to save?
- [ ] Reduce checkpoint frequency (every 250k steps instead of 100k)
- [ ] Compress JSON output

### Results worse than expected?
- [ ] Check WCET variability is enabled in rtos_env.py
- [ ] Verify extreme tasksets have correct utilization
- [ ] Try increasing `urgency_weight` to 0.7 or 0.8
- [ ] Try increasing `completion_reward` to 3.0

### Can't beat EDF on U=1.50?
- [ ] This is very hard; EDF is optimal for implicit-deadline tasks
- [ ] Consider this a success if within 1-2 misses of EDF
- [ ] Add state features for better stress detection
- [ ] Increase training duration (10M steps)

---

## Final Checklist

### Phase 1 Complete?
- [ ] Training finished
- [ ] All 5 output files exist
- [ ] No errors in logs

### Phase 2 Complete?
- [ ] Analysis completed
- [ ] Insights documented
- [ ] Decision for Phase 3 made

### Phase 3 (If Needed)
- [ ] Refinement done OR skipped
- [ ] New model trained (if done)

### Phase 4 Complete?
- [ ] Weights exported to Rust
- [ ] Extended simulation validates
- [ ] Power analysis done
- [ ] Ready for RTOS kernel

---

## Notes Section

Use this space to document findings and decisions:

```
Phase 1 Results:
[Your notes here]

Phase 2 Analysis:
[Your notes here]

Phase 3 Decisions:
[Your notes here]

Phase 4 Validation:
[Your notes here]
```

---

## Timeline Tracker

**Start Date:** _____________

**Phase 1 Start:** _____________
**Phase 1 Complete:** _____________ (Duration: _____ hours)

**Phase 2 Start:** _____________
**Phase 2 Complete:** _____________ (Duration: _____ hours)

**Phase 3 Start:** _____________ (If needed)
**Phase 3 Complete:** _____________ (Duration: _____ hours)

**Phase 4 Start:** _____________
**Phase 4 Complete:** _____________ (Duration: _____ hours)

**Total Time:** _____ hours (expected: 12-18 hours total)

---

Good luck! Track your progress and celebrate milestones! 🚀
