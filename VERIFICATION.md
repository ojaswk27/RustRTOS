# Phase 3 Verification Guide - Quick Start

**Time to verify**: ~10 minutes  
**Purpose**: Confirm Phase 3 results are reproducible and correct  
**Status**: All steps automated and documented

---

## ✅ Quick Verification (5 minutes)

Run these three commands to verify all Phase 3 results:

```bash
cd /Volumes/Spare/PycharmProjects/OS\ project/RustRTOS

# 1. Verify decision patterns (4/6 RMS alignment)
uv run python analyze_decisions.py 2>&1 | grep "PPO aligns"

# Expected output: PPO aligns with RMS on primary choice: 4/6 conditions

# 2. Verify performance (0 misses across all conditions)
uv run python eval_phase3.py 2>&1 | tail -5

# Expected output: Total misses - Phase 3: 0

# 3. Verify extended robustness (10 episodes, U=1.50)
uv run python verify_phase3.py 2>&1 | grep "Mean misses"

# Expected output: Mean misses: 0.00
```

---

## 📊 Detailed Verification Checklist

### 1. Environment Setup ✓
```bash
uv sync
# Should complete without errors
# All dependencies in pyproject.toml should install
```

### 2. Model Exists ✓
```bash
ls -la ppo_rtos_extended_model/ppo_rtos_extended.zip
# Should show file size ~40KB
```

### 3. Decision Analysis ✓
```bash
uv run python analyze_decisions.py
```

**Verify output contains**:
- [ ] "Loading PPO model..." (without errors)
- [ ] All 6 conditions evaluated (Normal, Stressed, Extreme U=1.20, U=1.30, U=1.40, U=1.50)
- [ ] Decision frequency tables for each condition
- [ ] "PPO aligns with RMS on primary choice: 4/6 conditions"
- [ ] File saved: decision_analysis_comparison.png

**Expected alignment results**:
```
Normal (U≈1.03)       Task 0    Task 0   YES ✓
Stressed (U≈1.15)     Task 5    Task 0   NO ✗
Extreme U=1.20        Task 0    Task 0   YES ✓
Extreme U=1.30        Task 0    Task 0   YES ✓
Extreme U=1.40        Task 1    Task 0   NO ✗
Extreme U=1.50        Task 0    Task 0   YES ✓
```

### 4. Performance Evaluation ✓
```bash
uv run python eval_phase3.py
```

**Verify output contains**:
- [ ] Evaluation running for all 6 conditions
- [ ] 0.0 misses for every condition
- [ ] Table showing Phase 1 vs Phase 3 comparison
- [ ] "Overall improvement: 96 misses (+100.0%)"

**Expected metrics**:
```
Normal (U≈1.03)        5.0 → 0.0 misses
Stressed (U≈1.15)      8.0 → 0.0 misses
Extreme U=1.20        12.0 → 0.0 misses
Extreme U=1.30        20.0 → 0.0 misses
Extreme U=1.40        21.0 → 0.0 misses
Extreme U=1.50        30.0 → 0.0 misses
```

### 5. Extended Robustness ✓
```bash
uv run python verify_phase3.py
```

**Verify output contains**:
- [ ] Two test cases: U=1.30 and U=1.50
- [ ] 10 episodes for each condition
- [ ] All episodes show 0 misses
- [ ] "Mean misses: 0.00" with "Std misses: 0.00"

**Example (U=1.50, 10 episodes)**:
```
Mean:    99.14    0       300
Std:     12.82    0       0
```

---

## 🔍 Document Review Checklist

- [ ] **PHASE_3_ANALYSIS.md** - Read executive summary (5 min)
  - Confirms 100% performance improvement
  - Explains WCET penalty mechanism
  - Documents 4-stage curriculum

- [ ] **PROJECT_SUMMARY.md** - Quick orientation (10 min)
  - Overview of all phases
  - File structure and locations
  - Key discoveries and limitations

- [ ] **phase3_analysis_fixed.log** - Raw output verification (5 min)
  - Console output from analyze_decisions.py
  - Decision frequency tables
  - PPO vs RMS comparison

- [ ] **comparison_phase3.json** - Raw metrics (2 min)
  - Performance data for all 6 conditions
  - 3 episodes per condition
  - Confirms 0 misses

---

## 🔧 Code Review Checklist

### WCET Penalty Implementation
```bash
grep -n "wcet_penalty" rtos_env.py
```

**Expected**:
- [ ] Line ~145: `wcet_penalty = 0.1 * (t.wcet / self.max_wcet)`
- [ ] Line ~146: `reward -= wcet_penalty`
- [ ] Line ~80: `self.max_wcet` calculation in __init__

### Curriculum Design
```bash
grep -n "TRAINING_STAGES\|Stage" train_extended.py | head -20
```

**Expected**:
- [ ] 4 stages defined (Light-Task, Deadline Pressure, Robust Prioritization, Extreme Stress)
- [ ] Stage 1: 1.0M steps, U=0.70-0.90, 0% variance
- [ ] Stage 4: 1.0M steps, U=1.30-1.50, 30% variance
- [ ] Total: 5.0M steps

---

## 🚀 What to Do After Verification

### If All Checks Pass ✅
→ Proceed to Phase 4 (Rust Integration)
- Create `export_weights.py` for Q10 conversion
- Modify `src/scheduler.rs` for fixed-point inference
- Test on Cortex-M4 simulator

### If Any Check Fails ❌
→ Debug following this order:

1. **Model not found**
   ```bash
   ls -la ppo_rtos_extended_model/
   # If empty, retrain: uv run python train_extended.py --train
   ```

2. **Import errors**
   ```bash
   uv sync --refresh
   # Reinstall all dependencies
   ```

3. **Results don't match**
   ```bash
   git log --oneline | head -5
   git show b97aa6c  # Check Phase 3 final commit
   # Verify no uncommitted changes to rtos_env.py or train_extended.py
   ```

4. **Visualization errors**
   - Already fixed in commit b97aa6c
   - Check analyze_decisions.py line 157: decision_counts trimming

---

## 📈 Expected vs Actual Comparison

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Total misses (Phase 1 vs 3) | 96 → 0 | TBD | |
| RMS alignment | 4/6 | TBD | |
| Analysis output | 4/6 alignment | TBD | |
| Performance eval | 0 misses | TBD | |
| Robustness (10 ep) | 0 misses | TBD | |
| Visualization | PNG created | TBD | |

---

## 🎓 Key Learnings for Next Engineer

1. **WCET Penalty Works**: Simple mechanism (0.1 coefficient) creates effective implicit RMS bias
2. **Curriculum is Critical**: 4-stage progression with increasing stress essential for success
3. **Perfect Results May Indicate Over-fit**: 0 misses in evaluation is suspiciously perfect; validate on real hardware
4. **Divergence ≠ Failure**: 4/6 alignment doesn't mean failure at U=1.15, U=1.40 (still achieves 0 misses)
5. **PPO Learns Beyond RMS**: Outperforms RMS at extreme stress (0 vs 14 misses at U=1.30)

---

## ⏱️ Timing Notes

- **analyze_decisions.py**: ~30 seconds (3 episodes × 6 conditions)
- **eval_phase3.py**: ~30 seconds (3 episodes × 6 conditions)
- **verify_phase3.py**: ~1 minute (10 episodes × 2 conditions)
- **Total verification time**: ~2 minutes

---

## 📞 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'gymnasium'"**
```bash
uv sync --refresh
```

**Q: "Model not found or corrupted"**
```bash
# Check integrity
file ppo_rtos_extended_model/ppo_rtos_extended.zip
# Should be valid ZIP file
```

**Q: "Results differ from expected"**
```bash
# Check git status
git status
# Verify no uncommitted changes to core files
# Check Python version: python3 --version (should be 3.11+)
```

**Q: "Visualization crashes"**
- Already fixed in commit b97aa6c
- Check line 157 in analyze_decisions.py: `decision_counts = decision_counts[:len(taskset)]`

---

**Generated**: March 31, 2026  
**Duration**: 5-15 minutes to complete all checks  
**Confidence Level**: High (all automated, reproducible)
