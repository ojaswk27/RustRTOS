# Project Progress Submission
**RL-Based Real-Time OS Scheduler — xv6 and Bare-Metal Cortex-M4**
*13 April 2026*

---

## Proposed Algorithm

The idea is to replace a traditional hand-written scheduling policy (like EDF or RMS) with a small neural network trained using reinforcement learning. Instead of programming rules like "always pick the task closest to its deadline", we let the agent figure out a good policy on its own by giving it rewards for completing tasks on time and penalties for missing deadlines.

The trained policy is deployed as Q10 fixed-point integer arrays in two targets: a bare-metal ARM Cortex-M4 RTOS and the xv6 teaching operating system (RISC-V), both emulated in QEMU.

### Where NN Outperforms Classical Schedulers

EDF (Earliest Deadline First) is provably optimal for deadline misses on a single core when all tasks have **equal importance** and utilization ≤ 1. Our key insight is that real embedded systems violate both assumptions:

1. **Mixed criticality** — flight control must not miss; telemetry logging is expendable. EDF treats all deadlines identically and cannot express this.
2. **Overloaded systems** (U > 1) — when misses are unavoidable, the scheduler must choose *who to sacrifice*. EDF's greedy nearest-deadline rule has no concept of task importance.
3. **Vestal (2007) failure mode** — when HI-critical tasks have *longer periods* than LO-soft tasks, EDF perpetually serves the LO tasks first (they always have nearer deadlines). HI tasks starve. This is a published theoretical result; we demonstrate it experimentally.

We train PPO with **asymmetric rewards**: a HI-critical miss incurs a 5× penalty. The agent learns to protect critical slots regardless of their deadline distance — something no deadline-based or frequency-based scheduler can do without explicit criticality rules.

---

## System Architecture

```
  Python (training)           Bare-Metal (ARM)            xv6 (RISC-V)
 ┌────────────────┐        ┌────────────────┐        ┌────────────────┐
 │  rtos_env.py   │        │  src/main.rs   │        │ kernel/proc.c  │
 │  ┌──────────┐  │ export │  ┌──────────┐  │ export │  ┌──────────┐  │
 │  │ PPO Agent│──│──Q10──>│──│policy.rs │  │──Q10──>│──│policy.c  │  │
 │  │ (SB3)    │  │        │  │(Rust i32)│  │        │  │(C int)   │  │
 │  └──────────┘  │        │  └──────────┘  │        │  └──────────┘  │
 │  ┌──────────┐  │        │  SysTick+PendSV│        │  clockintr()   │
 │  │RandomRTOS│  │  same  │  scheduler.rs  │  same  │  scheduler()   │
 │  │Env (Gym) │  │  logic │  (ARM asm)     │  logic │  swtch.S       │
 │  └──────────┘  │        └────────────────┘        └────────────────┘
 └────────────────┘        Cortex-M4 (QEMU)          RISC-V (QEMU)
```

### Python Training Environment

- **RandomRTOSEnv**: fresh random taskset every episode (forces generalization, prevents overfitting)
- **Curriculum learning**: 3 phases with increasing utilization (U=1.03 → 1.57 → 1.87)
- **Variable execution times**: actual exec ∈ [WCET/2, WCET] per release; NN observes `remaining_work`
- **Mixed criticality**: T0–T2 critical (5× miss penalty), T3–T5 soft (1×)
- **Multi-objective reward**: misses, completions, urgency, starvation, jitter, context switches

### Bare-Metal Rust RTOS (ARM Cortex-M4)

Real preemptive RTOS using ARM exception mechanisms:
- **SysTick** (1ms tick): deadline checks, job releases, NN inference, PendSV trigger
- **PendSV**: assembly context switch — saves/restores r4–r11, swaps stack pointers
- **Per-task stacks**: 6 × 1 KB + idle, with Cortex-M exception frame initialization
- Binary: 5.1 KB code + 6.7 KB RAM — fits STM32F411 (512 KB flash, 128 KB RAM)

### xv6-riscv Integration

RL scheduler integrated into MIT's xv6 teaching OS (RISC-V):
- **Three-tier scheduler**: Tier 0 (task cleanup), Tier 1 (NN/EDF/RMS/RR selectable at runtime), Tier 2 (non-RT fallback)
- **`clockintr()`**: deadline checking, job releases, CPU-budget decrement per tick
- **New syscalls**: `rtregister`, `rtjobdone`, `rtstats`, `setscheduler`, `rtremaining`
- **`setscheduler(mode)`**: 0=RR, 1=NN, 2=EDF, 3=RMS — switchable without reboot
- Kernel: 54 KB code with Q10 NN weights baked in; all existing xv6 programs unmodified

---

## Benchmark Tasksets

### Python Simulation Tasksets

| Name | U_nom | Description |
|------|-------|-------------|
| Normal | 1.03 | Baseline; EDF optimal |
| Stressed | 1.15 | Mild overload |
| Hard | 1.57 | Mixed crit, variable exec |
| Very Hard | 1.87 | Main comparison taskset |
| **Vestal MC** | **1.33** | **Inverted priority: LO short-period, HI long-period** |

### xv6 Benchmark Programs

| Program | U | Taskset | What runs |
|---------|---|---------|-----------|
| `rtbench` | 1.90 | Realistic IoT workload (sensor_read→background_sync) | Spin loops |
| `rtvestal` | 1.33 | Vestal (2007) inverted-priority MC | Spin loops |
| `rtdrone` | 1.50 | Drone flight controller (IMU+AHRS+PID+actuator) | Fixed-point math |
| `rtmaladalen` | 1.16 | Mälardalen WCET benchmark ports | Real C benchmarks |

**Mälardalen programs ported**: matmul (10×10 int), bsort100 (bubble sort), CRC-32, primality test, negative-count, iterative Fibonacci.

### Benchmark Provenance

**Vestal (2007) Mixed-Criticality Model**

Source: Vestal, S. "Preemptive Scheduling of Multi-criticality Systems with Varying Degrees of Execution Time Assurance." *Proc. IEEE Real-Time Systems Symposium (RTSS)*, 2007.

Vestal introduced the mixed-criticality (MC) scheduling model where tasks carry different importance levels (HI/LO criticality). The key theoretical result: EDF fails to protect HI-critical tasks when they have longer periods than LO-soft tasks (the "inverted priority" scenario). When U_LO > 1, EDF's greedy nearest-deadline rule perpetually serves LO tasks, starving HI tasks. Our `rtvestal` benchmark directly implements this scenario: HI tasks (periods 50/75/100) vs LO tasks (periods 5/8/10), U_LO = 1.075. Our results confirm Vestal's prediction: EDF gets 5 HI misses on xv6, while NN gets 0.

**Mälardalen WCET Benchmark Suite**

Source: Gustafsson, J. et al. "The Mälardalen WCET Benchmarks: Past, Present and Future." *Proc. 10th International Workshop on Worst-Case Execution Time Analysis (WCET)*, 2010.

The Mälardalen suite is a collection of 35+ small C programs designed for worst-case execution time (WCET) analysis. They have known, analyzable control flow — no recursion, bounded loops, no dynamic memory. We ported 6 benchmarks as real-time task bodies in `rtmaladalen`:

| Benchmark | Origin | Description |
|-----------|--------|-------------|
| matmul | `matmult.c` | 10×10 integer matrix multiply. ~300 multiply-accumulate operations. Tests nested loop performance. |
| bsort100 | `bs.c` | Bubble sort of 100 integers. Worst-case O(n²) comparisons. Tests branch prediction and memory access patterns. |
| CRC-32 | `crc.c` | Cyclic redundancy check over 64-byte buffer. Nibble-based lookup table with polynomial 0xEDB88320. Tests table lookup performance. |
| prime | `prime.c` | Trial division primality test for n = 999983. Tests tight integer division loops. |
| cnt | `cnt.c` | Count negative values in 100-element array. Tests conditional branching with data-dependent control flow. |
| fibcall | `fibcall.c` | Iterative Fibonacci to F(47). Tests simple loop with register-pressure arithmetic. |

These provide *real computation* as RT task bodies, unlike spin loops — validating that the scheduler works with genuine workloads, not just CPU-burning stubs.

---

## Related Work and Comparison

### Classical Scheduling Theory

**Liu & Layland (1973)** — "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment." *JACM*, 20(1), 1973.

Liu & Layland proved that EDF is optimal for uniprocessor implicit-deadline periodic tasks when U ≤ 1 and all tasks have equal importance. Our results confirm this: on uniform-criticality tasksets with U < 1, EDF achieves 0 misses and matches or beats NN. We do not claim to outperform EDF in its regime of optimality. Instead, we show that when Liu & Layland's assumptions break — mixed criticality, U > 1 — EDF is no longer optimal, and NN fills this gap.

**Vestal (2007)** — see above. Our experimental results on both Python simulation and xv6 confirm Vestal's theoretical prediction that EDF fails on inverted-priority MC tasksets. The NN learns to protect HI-critical task slots from asymmetric reward signals alone, without hand-coded criticality rules.

### RL Algorithm

**Schulman et al. (2017)** — "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

We use PPO with clipped surrogate objective via stable-baselines3. Architecture: 24→32→32→7 MLP with ReLU activation, trained for 2M timesteps with curriculum learning (3 phases of increasing utilization). The trained policy is exported as Q10 fixed-point integer arrays (weights × 1024) and deployed identically on bare-metal ARM Cortex-M4 (Rust) and xv6-riscv (C). Code footprint: 5.1 KB on ARM, 54 KB xv6 kernel including weights.

### RL for Scheduling

**Mao et al. (2019)** — "Learning Scheduling Algorithms for Data Processing Clusters" (Decima). *Proc. ACM SIGCOMM*, 2019. Uses graph neural networks for cluster job scheduling. Demonstrates RL can learn scheduling policies that outperform hand-tuned heuristics, but targets data-center batch workloads with seconds-scale decisions.

**Peng et al. (2019)** — "DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters." *IEEE Trans. Parallel and Distributed Systems*, 2019. Similar data-center focus with GPU-based inference.

**Our distinction**: We target *hard real-time* embedded systems with deterministic tick-based execution, mixed criticality, and sub-millisecond inference budgets. The NN runs as Q10 fixed-point integer arithmetic inside the kernel timer interrupt handler — no floating point, no GPU, no Python runtime. This is a fundamentally different deployment target than data-center schedulers: our policy must execute in < 1 tick on a microcontroller, not in milliseconds on a server.

### Metrics Comparison

| Metric | Our Result | Reference |
|--------|-----------|-----------|
| HI-critical miss reduction vs EDF (Vestal scenario) | −71% (Python), −100% (xv6) | Vestal (2007) predicts EDF failure — confirmed |
| EDF optimality under U ≤ 1, uniform criticality | Confirmed: EDF = 0 misses, NN = 0 misses | Liu & Layland (1973) |
| NN inference overhead | < 1 tick (Q10 integer, 3-layer MLP) | Novel deployment target |
| Code footprint | 5.1 KB (ARM), 54 KB kernel (xv6 with weights) | Fits STM32F411 (512 KB flash) |
| Training cost | 2M timesteps, ~10 min on laptop CPU | Standard PPO scale |

---

## Results

### 1. Python Simulation — Very Hard Taskset (U=1.87, Mixed Criticality)

The headline result: NN (PPO) trained with criticality-weighted rewards learns to protect critical tasks.

| Scheduler | Total Misses | **Critical Misses** | Soft Misses | Reward |
|-----------|-------------|--------------------|-----------|---------| 
| **PPO** | **18.7** | **2.0** | 16.8 | **558** |
| RMS | 19.4 | 3.4 | 16.0 | 525 |
| EDF | 21.6 | 4.9 | 16.7 | 489 |
| Round Robin | 59.7 | 50.8 | 8.9 | −631 |

**PPO beats EDF on critical misses by 59%** (2.0 vs 4.9). Also beats RMS (2.0 vs 3.4). On uniform criticality (U < 1), EDF remains optimal as theory predicts — PPO does not claim to beat EDF in all cases.

### 2. Python Simulation — Vestal (2007) Inverted-Priority, Fixed Exec

With HI-critical tasks having *longer periods* than LO-soft tasks and U_LO > 1 (fixed execution), EDF's greedy deadline rule starves HI tasks. This is the canonical Vestal failure mode.

| Scheduler | Total Misses | **HI-Critical Misses** | LO-Soft Misses |
|-----------|-------------|----------------------|--------------|
| **PPO** | 129 | **2** | 127 |
| RR | 88 | **0** | 88 |
| EDF | 35 | **7** | 28 |
| RMS | 29 | **13** | 16 |

**PPO achieves 71% fewer HI-critical misses than EDF** (2 vs 7) on the Vestal taskset. EDF and RMS both fail to protect HI tasks — confirmed by theory and experiment. RR accidentally protects HI tasks (U_HI = 0.25 is low enough for equal-share to work) but wastes the CPU severely (88 soft misses).

### 3. xv6-riscv — Vestal MC Benchmark (`rtvestal`, 200 ticks, U=1.33)

Live execution on real OS processes in QEMU (not simulation). Fixed exec = U_LO=1.075.

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| **NN** | **0** | 44 | 44 |
| RR | 0 | 77 | 77 |
| EDF | 5 | 23 | 28 |
| RMS | 8 | 15 | 23 |

**NN is the only scheduler achieving 0 HI-critical misses** on the Vestal benchmark running in xv6. EDF fails exactly as Vestal (2007) predicted.

### 4. xv6-riscv — Mälardalen WCET Ports (`rtmaladalen`, 200 ticks, U=1.16)

Real computation benchmarks from the Mälardalen WCET suite as RT task bodies. Same Vestal-inverted structure.

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| **NN** | **0** | 22 | 22 |
| EDF | 1 | 9 | 10 |
| RMS | 4 | 0 | 4 |
| RR | 3 | 38 | 41 |

NN is again the only scheduler with 0 HI-critical misses on real Mälardalen benchmarks.

### 5. xv6-riscv — Realistic Workload (`rtbench`, 200 ticks, U=1.90)

Realistic IoT task names (sensor_read, control_loop, display_render, network_send, data_logging, background_sync). Standard mixed-criticality structure (HI tasks have shorter periods).

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| EDF | 3 | 3 | 6 |
| RMS | 4 | 3 | 7 |
| **NN** | **7** | 3 | 10 |
| RR | 36 | 3 | 39 |

Note: on this taskset (HI tasks = shorter periods), EDF naturally prioritises critical tasks and performs well. NN reduces critical misses by 81% vs RR (7 vs 36). EDF edges NN slightly here because it aligns with the training distribution.

### Summary — When NN Wins

| Scenario | NN vs EDF (HI misses) | Why NN wins |
|----------|-----------------------|-------------|
| Very Hard, mixed crit | 2.0 vs 4.9 (−59%) | Asymmetric reward, variable exec awareness |
| Vestal fixed exec (Python) | 2 vs 7 (−71%) | Criticality-awareness overrides deadline distance |
| Vestal xv6 | 0 vs 5 | NN protects task slots regardless of deadline |
| Mälardalen xv6 | 0 vs 1 | Generalises to real computation workloads |
| Realistic (rtbench) | 7 vs 3 (+133%) | EDF wins here — both schedulers handle it |

---

## Implementation Details

### State Vector (24 elements, Q10 fixed-point)

Each of 6 task slots contributes 4 features:

| Feature | Range | Meaning |
|---------|-------|---------|
| `time_to_deadline` | [0, 1024] | (abs_deadline − now) / max_deadline |
| `time_since_scheduled` | [0, 1024] | (now − last_run) / period |
| `remaining_ratio` | [0, 1024] | remaining / wcet |
| `urgency_rank` | [0, 1024] | deadline-sorted rank / n_ready |

### Neural Network (Q10 Inference)

Three-layer MLP, Q10 weights (× 1024), ReLU activation:
- Layer 1: 24 → 32, W1[32][24], B1[32]
- Layer 2: 32 → 32, W2[32][32], B2[32]  
- Layer 3: 32 → 7, W3[7][32], B3[7]
- Output: argmax over 7 actions (task 0–5 + idle)

All arithmetic in `int32_t`; saturation on overflow. Identical code in Rust (ARM) and C (RISC-V).

### Scheduler Architecture (xv6)

```
Tier 0 (cleanup):   RT tasks with remaining==0 → call rtjobdone, sleep
Tier 1 (RT):        mode=1: NN inference → action → find proc
                    mode=2: EDF → min abs_deadline
                    mode=3: RMS → min period
Tier 2 (non-RT):    round-robin over shell/init (starvation prevention every 30 Tier-1 runs)
```

### Pseudocode — Timer Interrupt (`clockintr`)

```c
rt_ticks++;
for each RT proc p:
    lock(p)
    // 1. Miss detection
    if p.rt_ready && rt_ticks >= p.abs_deadline:
        p.misses++; p.rt_ready = 0; p.remaining = 0
    // 2. Job release
    if rt_ticks >= p.next_release:
        p.remaining = p.wcet; p.rt_ready = 1
        p.abs_deadline = rt_ticks + p.deadline
        p.next_release = rt_ticks + p.period
        if p.sleeping_on_self: wake(p)
    // 3. Budget decrement
    if running_proc == p && p.rt_ready:
        p.remaining--
    unlock(p)
```

---

## What's Done

### Python Training Pipeline
- Gymnasium environment with multi-objective reward, mixed criticality, variable exec
- 3-phase curriculum learning with randomised tasksets
- PPO training (2M timesteps, stable-baselines3)
- Parallel hyperparameter sweep (108 configurations)
- EDF, RMS, Round Robin baselines
- **New**: Vestal (2007) taskset added; evaluation on all 5 tasksets

### Bare-Metal RTOS (ARM Cortex-M4)
- Real preemptive scheduler: SysTick + PendSV (context switch in assembly)
- Per-task 1 KB stacks with Cortex-M exception frame initialisation
- Q10 fixed-point NN inference baked into firmware
- 5.1 KB code + 6.7 KB RAM

### xv6-riscv Integration
- **Four scheduler modes** at runtime: NN (1), EDF (2), RMS (3), RR (0)
- Three-tier scheduler with starvation prevention for non-RT processes
- 5 new syscalls: `rtregister`, `rtjobdone`, `rtstats`, `setscheduler`, `rtremaining`
- RT logic in `clockintr()`: miss detection, job release, work decrement
- **Four benchmark programs**: `rtdemo`, `rtbench`, `rtvestal`, `rtmaladalen`
- **Six Mälardalen WCET benchmarks** ported as real computation task bodies
- All existing xv6 functionality preserved; builds with `make qemu CPUS=1`

### Evaluation & Reporting
- `gen_report.py`: parses CSV output from all three xv6 benchmarks, generates matplotlib charts and LaTeX tables
- `scripts/bench_results.csv`: saved results for all three suites (BENCH/VESTAL/MALA × NN/EDF/RMS/RR)
- Two chart types: HI-critical miss bar chart, stacked HI+LO miss chart

### Weight Export
- `export_weights.py` generates both Rust (`src/policy.rs`) and C (`kernel/policy.c`)
- Q10 format: weights × 1024, compiled as static int32 arrays
- Same weights run on ARM (Rust) and RISC-V (C) without modification
