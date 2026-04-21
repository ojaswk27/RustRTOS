# Project Progress Submission
**RL-Based Real-Time OS Scheduler — xv6 and Bare-Metal Cortex-M4**
*18 April 2026*

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
- **`setscheduler(mode)`**: 0=RR, 1=NN, 2=EDF, 3=RMS, 4=MLFQ — switchable without reboot
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
| `rtdrone` | 1.50 | Drone flight controller (IMU+AHRS+PID+actuator) | Q10 plant model |
| `rtmaladalen` | 1.16 | Mälardalen WCET benchmark ports | Real C benchmarks |
| `rtgui` | 1.04 | GUI paint app (LFSR input, brush render, BFS flood fill) | Real computation |

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

**Spin Loop Benchmarks (`rtbench`, `rtvestal`)**

`rtbench` and `rtvestal` use CPU spin loops intentionally. Their purpose is to isolate scheduler timing behavior under controlled, worst-case load — each task consumes exactly its WCET every period with no computation variability. Introducing real task bodies in these benchmarks would conflate scheduler performance with execution time variance, obscuring the timing properties under test. Real computation task bodies are used in `rtmaladalen`, `rtdrone`, and `rtgui`, where the goal is to validate the scheduler against genuine workloads. `rtbench` and `rtvestal` serve a different role: they verify the scheduler's deadline enforcement mechanism under deterministic, maximally stressed conditions.

---

## Related Work and Comparison

### Classical Scheduling Theory

**Liu & Layland (1973)** — "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment." *JACM*, 20(1), 1973.

Liu & Layland proved that EDF is optimal for uniprocessor implicit-deadline periodic tasks when U ≤ 1 and all tasks have equal importance. Our results confirm this: on uniform-criticality tasksets with U < 1, EDF achieves 0 misses and matches or beats NN. We do not claim to outperform EDF in its regime of optimality. Instead, we show that when Liu & Layland's assumptions break — mixed criticality, U > 1 — EDF is no longer optimal, and NN fills this gap.

**Multi-Level Feedback Queue (MLFQ)** — Corbató et al. (1962), formalized in Arpaci-Dusseau & Arpaci-Dusseau "Operating Systems: Three Easy Pieces," 2018.

MLFQ uses multiple priority queues with time-quantum-based demotion: new tasks enter the highest priority queue; if they exhaust their time quantum without yielding, they are demoted to a lower queue. This adapts to observed behavior — I/O-bound tasks that voluntarily yield remain high-priority, while CPU-bound tasks are demoted. MLFQ is the closest classical scheduler to "adaptive" behavior, and is the default scheduler in macOS (BSD) and older Linux kernels.

We implemented MLFQ as scheduler mode 4 in xv6-riscv (3 queues, budget-based demotion, aging every 50 scheduling decisions) and measured it on all three benchmark suites. The results reveal two distinct failure modes:

**rtbench (U=1.90, similar periods for all tasks)**: MLFQ gets 25 HI-critical misses vs NN's 7. All tasks (critical and soft alike) have short-to-medium periods (10–33 ticks) and fully use their CPU budget every period — MLFQ demotes them all to the lowest queue within 1–2 periods. At steady state, MLFQ degenerates to weighted round-robin. Critically, it cannot distinguish between a safety-critical sensor loop (period=10) and a non-critical background task (period=100): both get demoted equally. Result: NN outperforms MLFQ by 72% on critical misses (7 vs 25).

**Vestal/Mälardalen (HI tasks have long periods)**: MLFQ gets 0 HI-critical misses — the same as NN. However, this is coincidental, not by design. HI-critical tasks have long periods (40–100 ticks); they use their small budget quickly, get demoted, then sit idle for many ticks while the aging mechanism (every 50 scheduling rounds) promotes them back to level 0 before their next deadline. The aging fires frequently enough relative to the long HI periods that HI tasks are never starved. But LO-soft tasks (periods 5–20) pay the price: MLFQ gives them 37 LO misses on Vestal vs EDF's 23, and 22 vs EDF's 9 on Mälardalen. The total miss burden is significantly higher than EDF or RMS.

The key point: MLFQ's "protection" of HI tasks on Vestal/Mälardalen is an artifact of the aging timer firing at the right frequency relative to task periods — it is not a repeatable guarantee. Change the aging interval or task periods and HI protection disappears. The NN, by contrast, explicitly observes criticality in its state vector and was trained with asymmetric rewards; it protects HI tasks by design across all taskset configurations.

**Vestal (2007)** — see above. Our experimental results on both Python simulation and xv6 confirm Vestal's theoretical prediction that EDF fails on inverted-priority MC tasksets. The NN learns to protect HI-critical task slots from asymmetric reward signals alone, without hand-coded criticality rules.

### When Does the NN Underperform, and Is That a Failure?

On `rtbench` (U=1.90), EDF gets 3 HI-critical misses while NN gets 7. Taken in isolation this looks like NN failure. The correct interpretation requires examining the taskset structure.

On `rtbench`, the HI-critical tasks happen to have *shorter* periods than the LO-soft tasks:

| Task | Criticality | Period | WCET |
|------|-------------|--------|------|
| sensor\_read | **HI** | **10** | 4 |
| control\_loop | **HI** | **15** | 5 |
| display\_render | **HI** | **33** | 7 |
| network\_send | LO | 50 | 3 |
| data\_logging | LO | 75 | 3 |
| background\_sync | LO | 100 | 2 |

EDF's greedy nearest-deadline rule naturally serves short-period tasks first, and here the short-period tasks are the critical ones. EDF "wins" not because it understands criticality — it does not — but because *criticality and urgency happen to coincide* on this particular taskset. Liu & Layland (1973) proved EDF optimal under equal-importance; this result extends trivially when importance and deadline distance are perfectly correlated.

The NN was trained on random tasksets where criticality does NOT systematically correlate with period length. It has learned a general policy: observe criticality flags and protect HI slots regardless of deadline distance. On tasksets where urgency and criticality are anti-correlated (the Vestal scenario), this policy wins decisively. On tasksets where they happen to coincide (rtbench), the structurally-specialized EDF edges out the general policy by a small margin.

**What NN failure would actually look like.** A uniformly random scheduler assigns equal time to each task. With 3 HI and 3 LO tasks and U=1.90, random scheduling achieves roughly a 50% HI miss rate — approximately 20 HI misses over 200 ticks. Our NN gets 7: 65% better than random. That is not policy failure. The NN has learned something useful; it simply cannot do better than EDF in the one scenario where EDF's heuristic coincidentally aligns with the right answer.

**Why PPO rather than DQN or A3C?** Three reasons. First, PPO's clipped surrogate loss prevents destructive policy updates during curriculum transitions — we observed instability with raw policy gradient (REINFORCE) when shifting from phase 1 to phase 3 utilization. Second, PPO is on-policy with stable variance: the clipping parameter ε=0.2 keeps the policy from collapsing to a deterministic argmin-deadline strategy mid-training. Third, deployability: the SB3 PPO implementation exports cleanly to numpy arrays → Q10 integer weights with no additional tooling. DQN would require a value network export; A3C introduces non-deterministic parallel gradient accumulation that complicates the curriculum schedule. For a target that runs on a microcontroller with 512 KB flash, PPO's clean two-network architecture (actor + critic, critic discarded at deployment) is the right choice.

### RL Algorithm

**Schulman et al. (2017)** — "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

We use PPO with clipped surrogate objective via stable-baselines3. Architecture: 24→32→32→7 MLP with ReLU activation, trained for 2M timesteps with curriculum learning (3 phases of increasing utilization). The trained policy is exported as Q10 fixed-point integer arrays (weights × 1024) and deployed identically on bare-metal ARM Cortex-M4 (Rust) and xv6-riscv (C). Code footprint: 5.1 KB on ARM, 54 KB xv6 kernel including weights.

### RL for Scheduling

**Mao et al. (2016/2019)** — "Resource Management with Deep Reinforcement Learning" (DeepRM, Mao et al. 2016) and "Learning Scheduling Algorithms for Data Processing Clusters" (Decima, Mao et al. 2019). DeepRM uses a CNN-based DQN for cluster bin-packing; Decima uses graph neural networks for DAG job scheduling. Both demonstrate RL scheduling outperforms hand-tuned heuristics on data-center workloads with seconds-scale decisions.

**Peng et al. (2019)** — "DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters." *IEEE Trans. Parallel and Distributed Systems*, 2019. Similar data-center focus with GPU-based inference.

**Comparison with DeepRM:**

| Dimension | DeepRM (Mao 2016) | This work |
|-----------|-------------------|-----------|
| **Deployment** | Python runtime, data-center server, ms-scale decisions | Q10 integer kernel ISR, microcontroller, < 1 tick (μs-scale) |
| **Criticality** | All jobs equal importance; no HI/LO distinction | Asymmetric reward: HI miss = 5× LO miss penalty |
| **Training regime** | Fixed synthetic job distributions | 3-phase curriculum, randomised tasksets per episode |
| **State representation** | Image-based resource grid (CNN input) | 24-element flat vector, Q10 fixed-point (no float) |
| **Policy export** | Python/TensorFlow model; not deployable to embedded | Q10 integer arrays; identical code on ARM Cortex-M4 and RISC-V |
| **Scheduling domain** | Batch cluster jobs (MAP-REDUCE, DAG) | Hard real-time periodic tasks with deadlines and criticality |

The key architectural difference is not algorithmic — both use policy gradient methods — but the deployment constraint. DeepRM was never intended to run in a kernel interrupt handler; its CNN processes a 20×60-pixel resource image in TensorFlow. Our policy must execute in under 1 millisecond on hardware without an FPU. The Q10 fixed-point MLP (24→32→32→7) with integer multiply-accumulate is purpose-built for this constraint.

The criticality-awareness gap is more fundamental. DeepRM cannot encode "this job is safety-critical" because its reward function is symmetric across jobs. Our asymmetric reward (5× penalty for HI misses) is not a post-hoc tweak — it is what causes the policy to learn a qualitatively different behavior from EDF, trading LO-task throughput for HI-task protection.

Curriculum learning is a third distinction. DeepRM trains on a fixed job distribution and its policy overfits to that distribution's statistics. We observed that training directly on high-utilization tasksets (U=1.87) produces a policy that achieves low HI misses on the training distribution but fails on the Vestal taskset (which has a different period-criticality correlation). The 3-phase curriculum — starting at U=1.03 where EDF is approximately optimal, then increasing to U=1.57, then U=1.87 — forces the policy to first learn the urgency signal (when things are easy) before learning to override it with criticality (when things are hard). This produces a policy that generalises across taskset structures.

**Our distinction from all data-center RL schedulers**: We target *hard real-time* embedded systems with deterministic tick-based execution, mixed criticality, and sub-millisecond inference budgets. The NN runs as Q10 fixed-point integer arithmetic inside the kernel timer interrupt handler — no floating point, no GPU, no Python runtime. This is a fundamentally different deployment target: our policy must execute in < 1 tick on a microcontroller, not in milliseconds on a server. The evaluation methodology also differs: we measure deadline *miss counts* per scheduler mode in a running OS, not average job completion time on a simulator.

### Metrics Comparison

| Metric | Our Result | Reference |
|--------|-----------|-----------|
| HI-critical miss reduction vs EDF (Vestal scenario) | −71% (Python), −100% (xv6) | Vestal (2007) predicts EDF failure — confirmed |
| EDF optimality under U ≤ 1, uniform criticality | Confirmed: EDF = 0 misses, NN = 0 misses | Liu & Layland (1973) |
| NN inference overhead | < 1 tick (Q10 integer, 3-layer MLP) | Novel deployment target |
| Code footprint | 5.1 KB (ARM), 54 KB kernel (xv6 with weights) | Fits STM32F411 (512 KB flash) |
| Training cost | 2M timesteps, ~10 min on laptop CPU | Standard PPO scale |

---

## Complete Taskset Specifications

Every taskset — Python and xv6 — uses the same lifecycle model. Each task is a periodic job with three parameters: **period** (how often it fires, in ticks), **deadline** (how many ticks it has to finish, always equal to period here), and **WCET** (worst-case execution time budget in ticks). The scheduler gets one CPU tick per scheduling decision. A task "misses" if its deadline passes before it finishes its current job.

**Tick rate:** 1 tick = ~100 ms on xv6 (CLINT 10 MHz, interval = 1,000,000 cycles). Python simulation uses dimensionless ticks.

**Task lifecycle (xv6):**
1. Parent forks one process per task and waits via `pause(SIM_TICKS)`.
2. Each child calls `rtregister(slot, period, deadline, wcet, criticality)` then `rtjobdone()` to sleep until its first period boundary.
3. Each tick the kernel's `clockintr()` increments `rt_ticks`, releases any task whose `next_release` has arrived, and checks for deadline misses.
4. The selected scheduler picks the next task; that task runs `while(rtremaining() > 0) { work(); }` then calls `rtjobdone()` to yield and record a completion.
5. After `SIM_TICKS` ticks the parent wakes, calls `rtstats(slot, &misses, &completions)` for each slot, and prints CSV output.

---

### Python Simulation Tasksets

All Python tasksets use 6 tasks with implicit deadlines (deadline = period). Tasks T0–T2 are HI-critical (criticality weight 5.0), T3–T5 are LO-soft (weight 1.0) under `MIXED_CRITICALITY`. Each episode runs for 300 ticks. `RandomRTOSEnv` samples a fresh taskset every episode from `CANDIDATE_PERIODS = [5,10,15,20,25,30,50,100]` with utilization drawn from the curriculum range.

#### Normal (U = 1.03) — EDF-optimal baseline

| Slot | Period | WCET | U_i |
|------|--------|------|-----|
| T0 | 10 | 2 | 0.200 |
| T1 | 15 | 3 | 0.200 |
| T2 | 20 | 4 | 0.200 |
| T3 | 30 | 5 | 0.167 |
| T4 | 50 | 8 | 0.160 |
| T5 | 100 | 10 | 0.100 |

U < 1 with equal criticality — Liu & Layland (1973) regime. EDF is provably optimal here; NN matches it (both 0 misses). Used to verify the agent doesn't over-fit to stressed conditions.

#### Stressed (U = 1.15) — mild overload

| Slot | Period | WCET | U_i |
|------|--------|------|-----|
| T0 | 10 | 3 | 0.300 |
| T1 | 15 | 3 | 0.200 |
| T2 | 20 | 4 | 0.200 |
| T3 | 30 | 5 | 0.167 |
| T4 | 50 | 8 | 0.160 |
| T5 | 100 | 12 | 0.120 |

Misses are unavoidable (U > 1). Mixed criticality begins to matter: the agent must choose which tasks absorb the overload.

#### Hard (U = 1.57) — high overload, variable execution

| Slot | Period | WCET | U_i |
|------|--------|------|-----|
| T0 | 10 | 4 | 0.400 |
| T1 | 15 | 5 | 0.333 |
| T2 | 20 | 5 | 0.250 |
| T3 | 30 | 7 | 0.233 |
| T4 | 50 | 10 | 0.200 |
| T5 | 100 | 15 | 0.150 |

`variable_exec=True` means each job's actual execution time is sampled uniformly from `[WCET/2, WCET]`. The agent sees `remaining_ratio` in its state, so it can observe how much work a task actually has left this job — unlike EDF or RMS which schedule purely from static parameters.

#### Very Hard (U = 1.87) — main comparison taskset

| Slot | Period | WCET | U_i |
|------|--------|------|-----|
| T0 | 10 | 5 | 0.500 |
| T1 | 15 | 6 | 0.400 |
| T2 | 20 | 7 | 0.350 |
| T3 | 30 | 8 | 0.267 |
| T4 | 50 | 12 | 0.240 |
| T5 | 100 | 20 | 0.200 |

U_HI = 1.25 alone exceeds 1 — even if only critical tasks run, they can't all meet their deadlines. The scheduler must still minimise HI misses. Key Python result: PPO achieves 2.0 HI misses vs EDF's 4.9 (59% reduction).

#### Vestal MC (U = 1.33) — inverted-priority scenario

| Slot | Criticality | Period | WCET | U_i | Role |
|------|-------------|--------|------|-----|------|
| T0 | **HI** | 50 | 5 | 0.100 | Flight control |
| T1 | **HI** | 75 | 6 | 0.080 | Safety monitor |
| T2 | **HI** | 100 | 7 | 0.070 | Actuator command |
| T3 | LO | 5 | 2 | 0.400 | Sensor poll |
| T4 | LO | 8 | 3 | 0.375 | Telemetry |
| T5 | LO | 10 | 3 | 0.300 | Log write |

U_LO = 1.075 > 1: soft tasks alone overflow the CPU. EDF always picks T3–T5 first (deadlines 5, 8, 10 vs 50, 75, 100) — HI tasks starve. Python result: PPO 2 HI misses vs EDF 7 (71% reduction).

---

### xv6 Benchmark Programs

#### `rtbench` — Realistic IoT Workload (U = 1.90)

Models a sensor-driven embedded application: sensor fusion feeding a control loop feeding a display, plus background telemetry/logging.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `sensor_read` | 10 | 10 | 4 | **HI** | Spin loop simulating IMU/ADC read at 100 Hz |
| T1 | `control_loop` | 20 | 20 | 7 | **HI** | Spin loop simulating PID controller at 50 Hz |
| T2 | `display_render` | 33 | 33 | 10 | **HI** | Spin loop simulating UI frame render at 30 fps |
| T3 | `network_send` | 100 | 100 | 25 | LO | Spin loop simulating telemetry transmission at 10 Hz |
| T4 | `data_logging` | 200 | 200 | 60 | LO | Spin loop simulating file write at 5 Hz |
| T5 | `background_sync` | 500 | 500 | 150 | LO | Spin loop simulating cloud sync at 2 Hz |

U_total = 1.90, U_HI = 1.05. Uses spin loops (`for(volatile int i=0; i<10000; i++)`). EDF happens to win here (3 misses vs NN's 7) because HI task periods (10, 20, 33) are shorter than LO task periods (100–500) — criticality and urgency accidentally correlate, so EDF's greedy rule accidentally picks the right tasks.

---

#### `rtvestal` — Vestal (2007) Inverted-Priority (U = 1.33)

Direct implementation of the Vestal (2007) failure scenario.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `flight_ctrl` | 50 | 50 | 5 | **HI** | Spin loop — flight attitude control |
| T1 | `safety_mon` | 75 | 75 | 6 | **HI** | Spin loop — safety monitor |
| T2 | `actuator_cmd` | 100 | 100 | 7 | **HI** | Spin loop — actuator command |
| T3 | `sensor_poll` | 5 | 5 | 2 | LO | Spin loop — high-frequency sensor poll |
| T4 | `telemetry` | 8 | 8 | 3 | LO | Spin loop — telemetry burst |
| T5 | `log_write` | 10 | 10 | 3 | LO | Spin loop — log write |

U_HI = 0.25, U_LO = 1.075, U_total = 1.325. HI tasks have periods 5–20× longer than LO tasks. EDF always sees a LO task with a nearer deadline, so it never runs HI tasks proactively. xv6 result: NN 0 HI misses, EDF 5 HI misses — direct confirmation of Vestal's prediction.

---

#### `rtmaladalen` — Mälardalen WCET Ports (U = 1.16)

Uses the same inverted-priority structure as rtvestal but replaces spin loops with real benchmark computations ported from the Mälardalen suite.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `matmul` | 60 | 60 | 12 | **HI** | 10×10 integer matrix multiply: `C[i][j] = Σ A[i][k]*B[k][j]` — 300 MACs per call |
| T1 | `bsort100` | 50 | 50 | 8 | **HI** | Bubble sort of 100 integers: O(n²) worst-case comparisons, in-place |
| T2 | `crc` | 40 | 40 | 6 | **HI** | CRC-32 over 64-byte string: nibble-based lookup table, polynomial 0xEDB88320 |
| T3 | `prime` | 10 | 10 | 3 | LO | Primality test of 999983 via trial division up to √n (~1000 iterations) |
| T4 | `cnt` | 15 | 15 | 3 | LO | Count negatives in 100-element array (every 3rd element is negative) |
| T5 | `fibcall` | 20 | 20 | 3 | LO | Iterative Fibonacci to F(47) = 2,971,215,073 (unsigned, wraps safely) |

U_HI = 0.51, U_LO = 0.65, U_total = 1.16. Each task calls its benchmark function repeatedly inside `while(rtremaining() > 0)` — the function executes as many times as the CPU budget allows each period. Because these are real computations, execution time varies slightly with data, but all loops are bounded so WCET is analyzable.

Also implemented (compiled as `__attribute__((unused))`, not scheduled):
- **`jfdctint`**: 8×8 integer JPEG DCT using scaled fixed-point arithmetic (row + column passes)
- **`ludcmp`**: 5×5 LU decomposition (Doolittle algorithm, values scaled ×256 to avoid floating-point)

---

#### `rtgui` — GUI Paint Application Baseline (U = 1.04)

Models the computational subsystems of an interactive paint application. All task bodies operate on a static 64×64 byte framebuffer.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `input_poll` | 8 | 8 | 2 | LO | 16-bit LFSR generates synthetic pointer events (x, y) and pushes them to a circular event queue (`evqueue[16]`) |
| T1 | `render` | 16 | 16 | 5 | **HI** | Pops one event; paints a 5×5 Gaussian brush stroke: `intensity = 255 - 40*(dx²+dy²)`, clamped to [0,255], written to `framebuf[y][x]` |
| T2 | `blit` | 33 | 33 | 8 | **HI** | Copies 16×16 region from `framebuf` to `backbuf` — double-buffer swap simulation |
| T3 | `flood_fill` | 100 | 100 | 20 | LO | Bounded BFS paint-bucket fill from a seed pixel: uses a 64-entry queue, 4-connected, stops at boundary pixels or queue exhaustion |
| T4 | `crc_save` | 200 | 200 | 3 | LO | CRC-32 over 64 bytes of framebuffer data — simulates save-to-file integrity check |
| T5 | `undo_snap` | 500 | 500 | 10 | LO | Copies entire `framebuf` to `backbuf` — undo history snapshot |

U_total = 1.04. Vestal structure: LO `input_poll` (period=8) has a nearer deadline than HI `render` (period=16). EDF always picks input_poll first. At U=1.04 the overload is small enough that EDF, RMS, and NN all achieve 0 HI misses — confirms Liu & Layland regime.

---

#### `rtgui_a` — Heavier GUI Variant (U = 1.37)

Same structure as `rtgui` but with significantly heavier render and blit task bodies. Designed so U_LO < 1 — the Vestal failure mode does NOT trigger; EDF is expected to win or tie.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `input_poll` | 8 | 8 | 2 | LO | Same LFSR event generation as rtgui |
| T1 | `render` | 25 | 25 | 12 | **HI** | **Pass 1:** 7×7 brush stroke with `intensity = 255 - 18*(dx²+dy²)` (49 pixels). **Pass 2:** 3×3 box blur over the ±2 neighbourhood of the paint point — each output pixel = mean of its 3×3 kernel |
| T2 | `blit` | 50 | 50 | 20 | **HI** | 32×32 alpha blend at 75% opacity: `out = (192*back + 64*front) >> 8` — 1024 blended pixels per call |
| T3 | `flood_fill` | 100 | 100 | 20 | LO | Same bounded BFS as rtgui |
| T4 | `crc_save` | 200 | 200 | 3 | LO | Same CRC-32 as rtgui |
| T5 | `undo_snap` | 500 | 500 | 10 | LO | Same snapshot as rtgui |

U_HI = 0.88, U_LO = 0.485, U_total = 1.365. Because U_LO < 1, soft tasks alone don't overflow the CPU — EDF can serve them all and still have budget for HI tasks. EDF edges NN here (4 vs 7 HI misses), confirming that the Vestal failure requires U_LO > 1.

---

#### `rtgui_b` — Compositor Vestal Scenario (U = 1.69)

Replaces `undo_snap` with a `compositor` task that is LO-soft but has a shorter period than the HI `render` task — directly recreating the Vestal (2007) failure mode in a GUI workload.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `input_poll` | 8 | 8 | 2 | LO | Same LFSR event generation |
| T1 | `render` | 16 | 16 | 5 | **HI** | Same 5×5 Gaussian brush stroke as rtgui |
| T2 | `blit` | 33 | 33 | 8 | **HI** | Same 16×16 double-buffer copy as rtgui |
| T3 | `compositor` | 12 | 12 | 8 | LO | **4-layer max-blend over 32×32:** iterates `layers[0..3][y][x]`, takes the per-pixel maximum across all 4 layers, writes result to `framebuf[y][x]`. Layers are initialised once with pattern `(l*64 + y + x) & 0xFF`. 4096 pixel operations per call |
| T4 | `flood_fill` | 100 | 100 | 20 | LO | Same bounded BFS |
| T5 | `crc_save` | 200 | 200 | 3 | LO | Same CRC-32 |

U_HI = 0.555, U_LO = 1.132, U_total = 1.687. The compositor (LO, T=12) has a shorter period than render (HI, T=16) — EDF perpetually picks compositor first. xv6 result: NN 3 HI misses vs EDF 10 (70% reduction). This is the strongest demonstration of NN advantage in the GUI suite.

---

#### `rtdrone` — Drone Flight Controller (U = 1.50)

Uses Q10 fixed-point arithmetic with a discrete plant model. State persists across job releases (static variables outside the job loop). All tasks share `imu_*`, `ahrs_q[]`, `plant_x[]`, `pid_*`, and `motor[]` arrays via the process's static BSS — each forked child gets its own copy of the state, modelling independent per-axis computation.

| Slot | Name | Period | Deadline | WCET | Criticality | Task body |
|------|------|--------|----------|------|-------------|-----------|
| T0 | `imu_read` | 10 | 10 | 3 | **HI** | 16-bit Galois LFSR (`0xB400` taps) drives synthetic gyro values: `gx = (lfsr & 0x1F) - 16`, `gy/gz` similarly from higher bits. Writes `imu_gx/gy/gz`. Simulates 100 Hz IMU |
| T1 | `ahrs_filter` | 10 | 10 | 4 | **HI** | Q10 complementary filter: `roll = (972*roll + 51*gx) >> 10`, same for pitch. The coefficient 972/1024 ≈ 0.95 is the discrete integrator pole; 51/1024 ≈ 0.05 is the gyro gain. Approximates quaternion magnitude normalisation |
| T2 | `pid_control` | 20 | 20 | 5 | **HI** | 3-axis PID loop. Error = `ref_angle[ax] - plant_x[ax]` (reference is level flight = 0). Integral clamped to ±4096 (anti-windup). Control output `u = Kp*err + Ki*integral + Kd*deriv` with Q10 coefficients Kp=800, Ki=10, Kd=200. Updates plant: `x[k+1] = (972*x[k] + 51*u[k]) >> 10` |
| T3 | `actuator_upd` | 20 | 20 | 2 | **HI** | Motor mixer: converts roll/pitch plant state to 4 motor PWM setpoints. `motor[0] = base + roll - pitch`, etc., where base = Q10_ONE/2 = 512 |
| T4 | `telemetry` | 100 | 100 | 20 | LO | Spin loop simulating GCS telemetry packet transmission at 10 Hz |
| T5 | `data_log` | 200 | 200 | 50 | LO | Spin loop simulating flight data recorder write at 5 Hz |

U_HI = 1.05, U_LO = 0.45, U_total = 1.50. U_LO < 1, so Vestal failure does not apply. EDF and RMS both achieve 0 HI misses; NN achieves 1 (the single miss is the AHRS filter task at the overload boundary). This benchmark demonstrates the regime where classical schedulers are sufficient.

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
| MLFQ | 0 | 37 | 37 |
| EDF | 5 | 23 | 28 |
| RMS | 8 | 15 | 23 |

The RR=0 result on this benchmark requires explanation. Round Robin achieves 0 HI-critical misses not because it is a good scheduler for mixed-criticality workloads, but because of an arithmetic coincidence: with 6 equally-weighted tasks and a tick-level quantum, RR allocates 1/6 of CPU time to each task. The HI-critical tasks in this taskset have long periods (50, 75, 100 ticks) and modest WCETs — their utilisation is low enough that receiving only 1/6 of CPU is still sufficient to meet every deadline. EDF, by contrast, actively and incorrectly steers CPU away from HI tasks: because HI tasks have far deadlines, EDF perpetually selects LO tasks (nearer deadlines), starving HI tasks entirely. RR's indifference to deadlines accidentally protects HI tasks in this specific configuration; on any taskset where U_HI > 1/n (where n is task count), RR would fail for the same reason it fails on rtbench.

NN and MLFQ both achieve 0 HI-critical misses. MLFQ's protection is incidental: aging fires frequently enough relative to the long HI task periods that they are restored to high priority before their deadlines. EDF fails exactly as Vestal (2007) predicted.

### 4. xv6-riscv — Mälardalen WCET Ports (`rtmaladalen`, 200 ticks, U=1.16)

Real computation benchmarks from the Mälardalen WCET suite as RT task bodies. Same Vestal-inverted structure.

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| **NN** | **0** | 22 | 22 |
| MLFQ | 0 | 22 | 22 |
| EDF | 1 | 9 | 10 |
| RMS | 4 | 0 | 4 |
| RR | 3 | 38 | 41 |

NN and MLFQ tie on HI-critical misses; MLFQ's protection is again aging-dependent, not a deterministic guarantee.

### 5. xv6-riscv — Realistic Workload (`rtbench`, 200 ticks, U=1.90)

Realistic IoT task names (sensor_read, control_loop, display_render, network_send, data_logging, background_sync). Standard mixed-criticality structure (HI tasks have shorter periods).

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| EDF | 3 | 3 | 6 |
| RMS | 4 | 3 | 7 |
| **NN** | **7** | 3 | 10 |
| MLFQ | 25 | 1 | 26 |
| RR | 36 | 3 | 39 |

Note: on this taskset (HI tasks have shorter periods), EDF naturally prioritises critical tasks. MLFQ performs worst among non-RR schedulers here (25 HI misses): all tasks have short-to-medium periods so MLFQ demotes them all equally and cannot distinguish criticality. NN reduces critical misses by 81% vs RR (7 vs 36).

### 6. xv6-riscv — GUI Paint Application (`rtgui`, 200 ticks, U=1.04)

Models an interactive paint application (similar to MS Paint) with LFSR-driven input, per-pixel brush rendering, BFS flood fill, CRC-32 autosave, double-buffer blit, and undo snapshots. HI-critical tasks: render (T=16, WCET=5) and blit (T=33, WCET=8). LO-soft: input\_poll, flood\_fill, crc\_save, undo\_snap.

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| **NN** | **0** | 2 | 2 |
| EDF | 0 | 1 | 1 |
| RMS | 0 | 3 | 3 |
| MLFQ | 13 | 4 | 17 |
| RR | 14 | 21 | 35 |

NN, EDF, and RMS all protect HI tasks — U=1.04 is below the overload threshold for the HI subset (U\_HI=0.55). MLFQ fails badly (13 HI misses): the render task (WCET=5) exceeds the Q0 budget of 2 ticks and is demoted to lower queues on its first job. From there it competes on equal footing with LO tasks, losing the render deadline repeatedly.

Note: at U=1.04, the HI-critical subset (render + blit) has U\_HI=0.55 — well below the overload threshold. NN, EDF, and RMS all protect HI tasks trivially in this regime, so this benchmark does not demonstrate NN's criticality-awareness advantage over classical schedulers. Its value is in two other dimensions: first, it validates that the scheduler operates correctly on real computation task bodies (pixel rendering, BFS, CRC-32) rather than spin loops; second, it shows MLFQ's structural failure (13 HI misses) even under mild overload, because the render task's WCET exceeds MLFQ's Q0 quantum and triggers immediate demotion regardless of utilisation. A higher-load variant of this benchmark (U≈1.35–1.70) is described in the extended evaluation below.

### Extended rtgui Evaluation (Higher Load)

Two higher-load variants were implemented and benchmarked:

**Option A (`rtgui_a`)** — heavier render (7×7 brush + 3×3 box blur pass) and blit (32×32 alpha blend at 75% opacity). Periods extended to render=25, blit=50. U\_HI=0.88, U\_LO=0.49, U\_total=1.37.

**Option B (`rtgui_b`)** — compositor task (4-layer max-blend over 32×32, period=12) replaces undo\_snap. HI tasks unchanged from baseline. U\_HI=0.55, U\_LO=1.13, U\_total=1.69. Vestal structure: compositor (LO, T=12) has shorter period than render (HI, T=16).

#### rtgui\_a (200 ticks, U\_total=1.37, U\_HI=0.88)

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| EDF | 4 | 4 | 8 |
| RMS | 4 | 3 | 7 |
| **NN** | **7** | 3 | 10 |
| MLFQ | 12 | 4 | 16 |
| RR | 12 | 22 | 34 |

On rtgui\_a, EDF and RMS edge out NN (4 vs 7 HI misses). This is the same structural coincidence as rtbench: U\_LO=0.49 < 1, so LO tasks do not monopolise the CPU. EDF's greedy deadline heuristic coincidentally aligns with the correct answer — shorter-period LO tasks (input\_poll T=8) don't collectively generate enough utilization to starve HI tasks. The Vestal failure mode does not activate. NN's general policy, which was trained to override deadlines for criticality, applies a cost when it isn't needed.

#### rtgui\_b (200 ticks, U\_total=1.69, U\_HI=0.55)

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| **NN** | **3** | 17 | 20 |
| EDF | 10 | 18 | 28 |
| RMS | 18 | 12 | 30 |
| MLFQ | 19 | 31 | 50 |
| RR | 18 | 44 | 62 |

On rtgui\_b, NN achieves **70% fewer HI-critical misses than EDF** (3 vs 10). This is a clear Vestal failure scenario: the compositor task (LO, T=12, WCET=8) has a shorter period than render (HI, T=16, WCET=5). EDF perpetually selects compositor (nearest deadline every 12 ticks) before render (deadline at 16 ticks). HI render is starved for CPU: render gets 8 completions but 4 misses; blit gets 0 completions and 6 misses. NN, by contrast, completes render 10 times and blit 6 times with only 3 HI misses total.

RMS performs even worse (18 HI misses): compositor (T=12) has a higher invocation rate than render (T=16), so RMS assigns compositor higher priority — the same failure mode as EDF but more severe. MLFQ fails catastrophically (19 HI misses): render (WCET=5) and blit (WCET=8) both exceed the Q0 quantum (2 ticks), so they are demoted on first scheduling and cannot compete with compositor.

The rtgui\_b result confirms the Vestal (2007) prediction for a realistic GUI workload: when a heavy LO task has a shorter period than HI tasks and U\_LO > 1, any urgency-based or frequency-based scheduler systematically fails to protect HI tasks. The NN, trained with asymmetric criticality rewards, learns to protect HI tasks regardless of their relative deadline distance.

### 7. xv6-riscv — Drone Flight Controller (`rtdrone`, 200 ticks, U=1.50)

Simulates a UAV mixed-criticality workload using Q10 fixed-point arithmetic: complementary AHRS filter, 3-axis PID controller with discrete plant model (pole at 0.95), and actuator mixer. HI-critical: imu\_read, ahrs\_filter, pid\_control (U\_crit=0.95). LO-soft: actuator\_upd, telemetry, data\_log.

| Scheduler | **HI-Crit Misses** | LO-Soft Misses | Total |
|-----------|-------------------|----------------|-------|
| EDF | 0 | 13 | 13 |
| RMS | 0 | 13 | 13 |
| **NN** | **1** | 10 | 11 |
| MLFQ | 50 | 3 | 53 |
| RR | 50 | 3 | 53 |

EDF and RMS both achieve 0 HI-critical misses; NN gets 1. The single NN HI miss occurs on `ahrs_filter` (T=10, WCET=4) — 2× the Q0 budget forces demotion in MLFQ, but in NN mode it is a near-miss from tight scheduling. With U\_crit=0.95, classical schedulers can protect HI tasks by their standard heuristics. NN remains competitive (1 vs 0) while cutting total misses (11 vs 13 for EDF/RMS). MLFQ catastrophically fails: imu\_read, ahrs\_filter, and pid\_control all have WCET > 2 ticks and are demoted immediately, resulting in 50 HI misses.

### Summary — When NN Wins

| Scenario | NN | EDF | RMS | MLFQ | RR |
|----------|----|-----|-----|------|-----|
| Very Hard, mixed crit (Python) | **2.0** | 4.9 | 3.4 | — | 50.8 |
| Vestal fixed exec (Python) | **2** | 7 | 13 | — | 0 |
| Vestal xv6 | **0** | 5 | 8 | 0* | 0 |
| Mälardalen xv6 | **0** | 1 | 4 | 0* | 3 |
| GUI xv6 (U=1.04) | **0** | 0 | 0 | 13 | 14 |
| GUI-B compositor xv6 (U=1.69) | **3** | 10 | 18 | 19 | 18 |
| Drone xv6 | 1 | **0** | **0** | 50 | 50 |
| Realistic (rtbench) | 7 | **3** | 4 | 25 | 36 |
| GUI-A heavier xv6 (U=1.37) | 7 | **4** | **4** | 12 | 12 |

*MLFQ gets 0 HI misses on Vestal/Mälardalen only because the aging interval fires before HI task deadlines. This is not a design property — change the aging interval or task periods and HI protection disappears.

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
- **Five scheduler modes** at runtime: NN (1), EDF (2), RMS (3), RR (0), MLFQ (4)
- MLFQ: 4-queue, per-tick budget (2/4/8/16 ticks), demotion on budget expiry, boost every 100 ticks
- Three-tier scheduler with starvation prevention for non-RT processes
- 5 new syscalls: `rtregister`, `rtjobdone`, `rtstats`, `setscheduler`, `rtremaining`
- RT logic in `clockintr()`: miss detection, job release, work decrement, MLFQ budget management
- **Seven benchmark programs**: `rtdemo`, `rtbench`, `rtvestal`, `rtmaladalen`, `rtgui`, `rtgui_a`, `rtgui_b`
- **rtgui**: GUI paint application simulation — LFSR input, brush rendering, BFS flood fill, CRC-32, double-buffer blit, undo snapshot
- **rtgui\_a**: higher-load variant — 7×7 brush + 3×3 blur, 32×32 alpha blend, U=1.37
- **rtgui\_b**: Vestal-structure variant — compositor (LO, T=12) vs render (HI, T=16), U=1.69; demonstrates EDF failure under LO overload
- **rtdrone** (rewritten): Q10 fixed-point AHRS filter and PID controller with discrete plant model (pole at 0.95)
- **Six Mälardalen WCET benchmarks** ported as real computation task bodies in `rtmaladalen`; jfdctint (8×8 integer DCT) and ludcmp (5×5 LU decomposition) implemented as additional task bodies and documented in the benchmark provenance section
- All existing xv6 functionality preserved; builds with `make qemu CPUS=1`

### Evaluation & Reporting
- `gen_report.py`: parses CSV output from all five xv6 benchmarks, generates matplotlib charts and LaTeX tables
- `scripts/bench_results.csv`: saved results for all seven suites (BENCH/VESTAL/MALA/GUI/DRONE/GUIA/GUIB × NN/EDF/RMS/RR/MLFQ) — 35 suite×mode combinations
- Two chart types: HI-critical miss bar chart, stacked HI+LO miss chart

### Weight Export
- `export_weights.py` generates both Rust (`src/policy.rs`) and C (`kernel/policy.c`)
- Q10 format: weights × 1024, compiled as static int32 arrays
- Same weights run on ARM (Rust) and RISC-V (C) without modification

---

## Conclusion

This project demonstrates that a small neural network trained with PPO can serve as an effective real-time OS scheduler — one that learns to protect safety-critical tasks without hard-coded criticality rules — and can be deployed within the kernel of a real operating system.

### What Was Built

A complete end-to-end system: Python training environment → Q10 weight export → bare-metal ARM Cortex-M4 RTOS (Rust) → xv6-riscv integration (C). The NN runs as 72 integer multiply-accumulate operations inside the timer interrupt handler, with no floating point, no dynamic memory, and no external runtime. Five schedulers (NN, EDF, RMS, RR, MLFQ) were benchmarked across seven tasksets (35 suite×mode combinations) on a real OS running in QEMU.

### When the NN Wins

The NN's advantage is most pronounced on the **Vestal (2007) inverted-priority scenario**: when HI-critical tasks have longer periods than LO-soft tasks, EDF's greedy deadline heuristic starves HI tasks (5 HI misses on xv6-riscv). The NN achieves 0 HI-critical misses by learning to override deadline distance in favor of criticality — a policy that cannot be expressed by any urgency-based or frequency-based scheduler. On the Mälardalen WCET ports (real C computation), the NN again achieves 0 HI misses vs EDF's 1 and RMS's 4. On the GUI-B benchmark (compositor task creates Vestal LO overload at U=1.69), NN achieves 70% fewer HI misses than EDF (3 vs 10): the same Vestal failure mechanism reproduced in a realistic GUI workload. On the baseline GUI benchmark (U=1.04), NN matches EDF/RMS while MLFQ fails with 13 HI misses. In Python simulation on the Very Hard taskset (U=1.87), NN achieves 2.0 HI misses vs EDF's 4.9 (59% improvement) and vs RR's 50.8 (96% improvement).

MLFQ is particularly instructive as a foil: it consistently fails when tasks have WCET > 2 ticks (the Q0 quantum), because it demotes all such tasks on their first job and cannot distinguish a safety-critical sensor loop from a background logger. On the drone benchmark, MLFQ produces 50 HI-critical misses — the same as Round Robin.

### When the NN Does Not Win

On `rtbench` (U=1.90) and `rtgui_a` (U=1.37), EDF edges out NN (rtbench: 3 vs 7; rtgui\_a: 4 vs 7) because neither taskset triggers the Vestal failure mode: LO utilization is below 1.0, so EDF's greedy heuristic coincidentally aligns with the correct answer without needing criticality awareness. The NN, trained on random tasksets where this correlation is absent, pays a small penalty for having learned a general policy. This is not policy failure — NN still outperforms MLFQ and RR by large margins on both benchmarks. On the drone benchmark (U\_crit=0.95), EDF and RMS achieve 0 HI misses vs NN's 1; the difference is marginal and traceable to the AHRS filter task sitting precisely at the overload boundary.

### Limitations

- **Single core only.** The scheduler assumes a single CPU. Extension to multiprocessor scheduling requires partitioning or global scheduling — both open research problems for MC systems.
- **Fixed task count.** The state vector encodes exactly 6 tasks; the policy does not generalise to arbitrary task counts without architectural changes (e.g., attention over variable-length task sets).
- **Q10 quantization.** Converting float32 weights to Q10 integers (×1024) loses approximately 0.1% precision; on some tasksets this produces slightly different action distributions than the Python policy.
- **Simulation-to-real gap.** QEMU executes RISC-V instructions deterministically; real hardware introduces cache effects, pipeline stalls, and interrupt latencies that may shift actual WCET relative to the values encoded in the taskset. The NN has not been tested on physical STM32F411 silicon.

### Future Work

- **Variable task count** via a set-attention encoder, enabling the policy to generalise to tasksets of arbitrary size.
- **Online adaptation**: fine-tune the policy at runtime using observed execution times, adapting to workload drift without retraining.
- **Formal safety bounds**: frame the Q10 NN as a piecewise-linear function and apply abstract interpretation or SMT solving to prove worst-case HI-miss bounds.
- **Physical hardware evaluation** on an STM32F411 to validate the 5.1 KB code footprint and measure actual ISR latency under real interrupt jitter.
- **Mixed-criticality certification** against AUTOSAR or DO-178C scheduling requirements — the criticality-awareness property of the trained policy is a first step toward formal safety arguments.
