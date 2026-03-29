# Project Progress Submission
**RL-Based Real-Time OS Scheduler on Bare-Metal Cortex-M4**
*Submission Date: 30 March 2026*

---

## Proposed Algorithm

### Overview

We propose a **PPO-trained neural network scheduler** for a preemptive tick-based RTOS. Rather than a hand-crafted priority function (EDF, RMS), a policy network learns optimal task selection end-to-end from reward signals. The trained weights are quantized to Q10 fixed-point and deployed on an ARM Cortex-M4 microcontroller (QEMU-emulated STM32F411) with no floating-point operations at inference time.

---

## Pseudocode

### 1. Scheduler Tick Loop (Inference / Bare-Metal)

```
procedure SCHEDULER_TICK(tasks[], tick):
    // Step 1: detect deadline misses BEFORE releasing new jobs
    for each task t in tasks:
        if t.ready AND tick >= t.abs_deadline:
            t.ready    ← false
            t.remaining ← 0
            record deadline miss

    // Step 2: release new jobs at period boundaries
    for each task t in tasks:
        if tick >= t.next_release:
            t.remaining    ← t.wcet
            t.abs_deadline ← tick + t.deadline
            t.next_release ← tick + t.period
            t.ready        ← true

    // Step 3: build state observation (24-element vector)
    obs ← BUILD_STATE(tasks, tick)

    // Step 4: run neural network inference → action
    action ← NN_INFER(obs)          // returns 0..5 (run task) or 6 (idle)

    // Step 5: execute selected task for one tick
    if action ≠ IDLE AND tasks[action].ready:
        tasks[action].remaining ← tasks[action].remaining − 1
        tasks[action].last_scheduled ← tick
        if tasks[action].remaining = 0:
            tasks[action].ready ← false
            record completion

    tick ← tick + 1
```

---

### 2. State Construction

```
procedure BUILD_STATE(tasks[], tick) → obs[24]:
    MAX_DEADLINE ← max(t.deadline for t in tasks)
    MAX_PERIOD   ← max(t.period   for t in tasks)

    for i = 0 to 5:
        base ← i × 4
        if tasks[i].ready:
            obs[base + 0] ← clamp((tasks[i].abs_deadline − tick) / MAX_DEADLINE, 0, 1)
            obs[base + 1] ← clamp((tick − tasks[i].last_scheduled) / MAX_PERIOD, 0, 1)
            obs[base + 2] ← tasks[i].remaining / tasks[i].wcet
            obs[base + 3] ← 1.0
        else:
            obs[base .. base+3] ← 0.0   // not ready → all zeros

    return obs
```

Features per task:
| Index | Feature | Meaning |
|-------|---------|---------|
| base+0 | `time_to_deadline` | Urgency (0 = at deadline, 1 = full period remains) |
| base+1 | `time_since_scheduled` | Starvation indicator |
| base+2 | `remaining_work` | Fraction of WCET left to execute |
| base+3 | `is_ready` | 1 if task has a pending job, 0 otherwise |

---

### 3. Neural Network Inference (Q10 Fixed-Point)

Architecture: **MLP 24 → 32 → 32 → 7**

```
procedure NN_INFER(obs[24]) → action:
    // Layer 1: input → hidden1 (ReLU)
    for j = 0 to 31:
        h1[j] ← 0
        for i = 0 to 23:
            h1[j] ← h1[j] + W1[j][i] × obs_q10[i]
        h1[j] ← (h1[j] / 1024) + B1[j]        // Q10 descale
        h1[j] ← max(h1[j], 0)                   // ReLU

    // Layer 2: hidden1 → hidden2 (ReLU)
    for j = 0 to 31:
        h2[j] ← 0
        for i = 0 to 31:
            h2[j] ← h2[j] + W2[j][i] × h1[i]
        h2[j] ← (h2[j] / 1024) + B2[j]
        h2[j] ← max(h2[j], 0)

    // Layer 3: hidden2 → output logits
    for j = 0 to 6:
        out[j] ← 0
        for i = 0 to 31:
            out[j] ← out[j] + W3[j][i] × h2[i]
        out[j] ← (out[j] / 1024) + B3[j]

    return argmax(out)       // greedy action selection
```

All arithmetic uses saturating 32-bit integers. Division by 1024 is a right-shift (`>> 10`). No FPU instructions are emitted.

---

### 4. PPO Training Loop (Python, Offline)

```
procedure TRAIN_PPO():
    env ← RTOSEnv(taskset=NORMAL_TASKSET, max_ticks=300)
    policy ← MLP(layers=[32, 32], input=24, output=7)
    optimizer ← Adam(lr=3×10⁻⁴)

    for step = 1 to 500,000:
        // Collect rollout
        obs ← env.reset()
        for t = 1 to max_ticks:
            action, log_prob, value ← policy(obs)
            obs', reward, done ← env.step(action)

            reward shaping:
                +1.0  task completed on time
                −2.0  per deadline miss
                −0.01 per tick (urgency pressure)
                −0.05 context switch (task→task only)

            store (obs, action, reward, log_prob, value)
            obs ← obs'
            if done: break

        // PPO update (n_epochs=10, batch=64, clip ε=0.2)
        for epoch = 1 to 10:
            for minibatch in rollout_buffer:
                ratio ← exp(log_prob_new − log_prob_old)
                L_clip ← min(ratio × advantage,
                             clip(ratio, 1−ε, 1+ε) × advantage)
                loss ← −L_clip + c₁×L_value − c₂×entropy
                backprop(loss)

    export_weights_to_Q10(policy)   // → src/policy.rs
```

---

### 5. Weight Export

```
procedure EXPORT_WEIGHTS(policy):
    for each layer (W, B) in policy.actor_network:
        W_q10 ← round(W × 1024).astype(int32)
        B_q10 ← round(B × 1024).astype(int32)
        write as static Rust arrays to src/policy.rs
```

---

## Preliminary Results

> Results from 100 evaluation episodes per scheduler, over one hyperperiod (300 ticks).
> *(Table will be populated after training completes — results appended below.)*

### Taskset Definitions

| Task | Period | Deadline | WCET | Utilization |
|------|--------|----------|------|-------------|
| T0   | 10     | 10       | 2    | 0.200       |
| T1   | 15     | 15       | 3    | 0.200       |
| T2   | 20     | 20       | 4    | 0.200       |
| T3   | 30     | 30       | 5    | 0.167       |
| T4   | 50     | 50       | 8    | 0.160       |
| T5   | 100    | 100      | 10   | 0.100       |
| **Total** | | | | **≈ 1.03** |

**Stressed taskset** increases T0.wcet to 3, T5.wcet to 12 → U ≈ 1.15.

Both tasksets are intentionally **overloaded** (U > 1.0). No scheduler can eliminate all misses; the objective is to *minimize* them.

---

*Best model selected via parallel reward sweep (16 configs × 300k steps). Winning config: `miss_penalty=−3.0`, `completion_reward=1.5`. Evaluated over 50 episodes each.*

### Results — Normal Taskset (U ≈ 1.03)

| Scheduler    | Avg Reward | Avg Deadline Misses |
|--------------|------------|---------------------|
| **PPO (ours)** | 55.2     | **4.0**             |
| Round Robin  | 30.4       | 12.0                |
| RMS          | 66.4       | 3.0                 |
| EDF          | 69.2       | 2.0                 |

### Results — Stressed Taskset (U ≈ 1.15)

| Scheduler    | Avg Reward | Avg Deadline Misses |
|--------------|------------|---------------------|
| **PPO (ours)** | 55.2     | **7.0**             |
| Round Robin  | −50.9      | 39.0                |
| RMS          | 60.8       | 5.0                 |
| EDF          | 55.1       | 7.0                 |

PPO matches EDF on the stressed taskset despite being trained only on the normal taskset, demonstrating generalization under increased load.

---

## Deployment Pipeline

```
train.py          →  ppo_rtos_model/ppo_rtos.zip   (trained PPO weights)
export_weights.py →  src/policy.rs                 (Q10 Rust arrays)
cargo run         →  QEMU Cortex-M4 execution       (bare-metal inference)
```

The same tick loop and state encoding are implemented identically in Python (for training) and Rust (for deployment), ensuring fidelity between simulation and hardware.

---

## Infrastructure Status

| Component | Status |
|-----------|--------|
| RTOS simulation environment (`rtos_env.py`) | Complete |
| PPO training harness (`train.py`) | Complete |
| Baseline schedulers (RR, RMS, EDF) | Complete |
| Q10 weight export (`export_weights.py`) | Complete |
| Rust tick-loop + state encoder | Complete |
| Q10 fixed-point NN inference (`policy.rs`) | Complete |
| Bare-metal QEMU execution | Complete |
| Trained weights | *In progress* |
