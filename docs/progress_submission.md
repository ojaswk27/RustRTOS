# Project Progress Submission
**RL-Based Real-Time OS Scheduler — xv6 and Bare-Metal Cortex-M4**
*8 April 2026*

---

## Proposed Algorithm

The idea is to replace a traditional hand-written scheduling policy (like EDF or RMS) with a small neural network trained using reinforcement learning. Instead of programming rules like "always pick the task closest to its deadline", we let the agent figure out a good policy on its own by giving it rewards for completing tasks on time and penalties for missing deadlines.

The trained policy is deployed as Q10 fixed-point integer arrays in two targets: a bare-metal ARM Cortex-M4 RTOS and the xv6 teaching operating system (RISC-V), both emulated in QEMU.

### Where PPO Outperforms Classical Schedulers

EDF (Earliest Deadline First) is provably optimal for deadline misses on single-core periodic task systems when all tasks have equal importance. Our key insight is that real embedded systems have **mixed criticality** — some tasks are safety-critical (sensor reads, control loops) and others are soft (logging, telemetry). EDF treats all deadlines equally and cannot distinguish between them.

We train PPO with **asymmetric rewards**: missing a critical task's deadline incurs a 5x penalty compared to a soft task. The agent learns to **selectively sacrifice soft tasks to protect critical ones** under overload — something EDF fundamentally cannot do.

Additionally, with **variable execution times** (actual exec sampled from [WCET/2, WCET] each job release), PPO has a genuine information advantage: it observes the `remaining_work` feature and can reason about how much CPU a task actually needs. EDF only looks at deadlines and has no concept of remaining work.

---

## System Architecture

The project has three layers:

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
 │  │Env (Gym) │  │  logic │  switch.rs     │  logic │  swtch.S       │
 │  └──────────┘  │        │  (ARM asm)     │        │  (RISC-V asm)  │
 └────────────────┘        └────────────────┘        └────────────────┘
                           Cortex-M4 (QEMU)          RISC-V (QEMU)
```

### Python Training Environment

- **RandomRTOSEnv**: generates a fresh random taskset every episode to force generalization
- **Curriculum learning**: 3 training phases with increasing utilization
- **Variable execution times**: actual exec sampled from [WCET/2, WCET] each release
- **Mixed criticality**: T0-T2 are critical (5x miss penalty), T3-T5 are soft (1x)
- **Multi-objective reward**: deadline misses, completions, urgency, starvation, jitter, context switches

### Bare-Metal Rust RTOS (ARM Cortex-M4)

A real preemptive RTOS using ARM exception mechanisms:
- **SysTick**: fires every 1ms, runs scheduler (deadline checks, releases, NN inference, PendSV trigger)
- **PendSV**: assembly-level context switch (saves/restores r4-r11, swaps stack pointers)
- **Per-task stacks**: 6 x 1KB + idle stack, with Cortex-M exception frame initialization
- Binary: 5.1KB code + 6.7KB RAM

### xv6-riscv Integration

The RL scheduler is integrated into MIT's xv6 teaching OS:
- **Two-tier scheduler**: NN governs 6 RT tasks; non-RT processes (shell, init) use round-robin fallback
- **Timer interrupt**: `clockintr()` handles deadline checking, job releases, and work decrement
- **New syscalls**: `rtregister`, `rtjobdone`, `rtstats`, `setscheduler`
- **Context switching**: uses xv6's existing `swtch.S` (RISC-V callee-saved register save/restore)
- **User program**: `rtdemo` launches 6 periodic tasks, runs for 300+ ticks, prints stats
- Kernel: 54KB code + 107KB BSS (with NN weights baked in)
- Switchable between NN and round-robin at runtime for A/B comparison

---

## Pseudocode

### Timer Interrupt (clockintr / SysTick)

Every tick, the timer interrupt runs the RT scheduling logic. The ordering matters — deadline checks happen before releases so that a task expiring right at its period boundary is counted as a miss rather than quietly overwritten by the new job.

```
clockintr():
    tick ← tick + 1

    for each RT task t:
        // 1. Check deadline miss (before release)
        if t.rt_ready AND tick >= t.abs_deadline:
            t.rt_ready  ← false
            t.remaining ← 0
            t.misses    ← t.misses + 1

        // 2. Release new job at period boundary
        if tick >= t.next_release:
            t.remaining    ← t.wcet
            t.abs_deadline ← tick + t.deadline
            t.next_release ← tick + t.period
            t.rt_ready     ← true
            if t.state = SLEEPING: t.state ← RUNNABLE

        // 3. Decrement currently-running RT task's work
        if current_cpu.proc = t AND t.rt_ready AND t.remaining > 0:
            t.last_scheduled ← tick
            t.remaining      ← t.remaining - 1
            if t.remaining = 0:
                t.completions ← t.completions + 1
                t.rt_ready    ← false
```

### Two-Tier Scheduler

```
scheduler():
    loop forever:
        // Tier 1: NN scheduling for RT tasks
        if any RT task is RUNNABLE:
            state ← BUILD_STATE(proc_table, tick)
            action ← NN_INFER(state)       // Q10 fixed-point, ~2000 MACs
            if action ≠ IDLE:
                switch to proc with rt_id = action

        // Tier 2: Round-robin for non-RT processes (shell, init, etc.)
        for each proc p:
            if p.state = RUNNABLE:
                switch to p
```

### State Encoding (24 floats)

```
BUILD_STATE(tasks[], tick) → obs[24]:
    ready_sorted ← sort(ready tasks, by abs_deadline ascending)
    n_ready      ← count of ready tasks

    for i = 0 to 5:
        base ← i × 4
        if tasks[i] is ready:
            obs[base+0] ← (abs_deadline − tick) / MAX_DEADLINE   // time_to_deadline
            obs[base+1] ← (tick − last_scheduled) / period       // time_since_scheduled
            obs[base+2] ← remaining / wcet                       // remaining_work
            obs[base+3] ← (n_ready − rank) / n_ready             // urgency_rank
        else:
            obs[base .. base+3] ← 0.0
```

| Feature | What it tells the network |
|---------|--------------------------|
| `time_to_deadline` | How urgent the task is. 1.0 = lots of time, 0.0 = at the deadline. |
| `time_since_scheduled` | Whether the task is being starved. Normalized by own period. |
| `remaining_work` | How much CPU work is left. With variable exec, reflects actual remaining. |
| `urgency_rank` | Relative urgency among ready tasks. 1.0 = most urgent (nearest deadline). |

### Neural Network Inference (Q10 Fixed-Point)

```
NN_INFER(obs[24]) → action:
    // Layer 1: 24 → 32, ReLU
    for j = 0 to 31:
        h1[j] ← B1[j] + Σ(W1[j][i] × obs_q10[i]) >> 10
        h1[j] ← max(h1[j], 0)

    // Layer 2: 32 → 32, ReLU
    for j = 0 to 31:
        h2[j] ← B2[j] + Σ(W2[j][i] × h1[i]) >> 10
        h2[j] ← max(h2[j], 0)

    // Output: 32 → 7, argmax
    return argmax(B3[j] + Σ(W3[j][i] × h2[i]) >> 10)
```

### PPO Training (3-Phase Curriculum with Mixed Criticality)

```
TRAIN():
    criticality ← [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]  // T0-T2 critical, T3-T5 soft

    // Phase 1: Easy (U = 0.60-0.95)
    train PPO for 666,667 steps on RandomRTOSEnv(U=0.60-0.95)

    // Phase 2: Medium (U = 0.85-1.10)
    train PPO for 666,667 steps on RandomRTOSEnv(U=0.85-1.10)

    // Phase 3: Hard (U = 0.95-1.20)
    train PPO for 666,667 steps on RandomRTOSEnv(U=0.95-1.20)

    // Reward (per tick, criticality-weighted):
    reward += -3.0 × criticality[i] × misses   // critical miss = -15.0, soft = -3.0
    reward += +2.0 × criticality[i] × completions
    reward += +0.1 × urgency
    reward += -0.02 × context_switch
    reward += -0.05 × starvation
    reward += -0.02 × jitter
    reward += -0.01  // tick cost
```

---

## Tasksets

### Standard Tasksets (for baseline comparison)

| Task | Period | Deadline | WCET (normal) | WCET (stressed) | U (normal) |
|------|--------|----------|--------------|-----------------|------------|
| T0   | 10     | 10       | 2            | 3               | 0.20       |
| T1   | 15     | 15       | 3            | 3               | 0.20       |
| T2   | 20     | 20       | 4            | 4               | 0.20       |
| T3   | 30     | 30       | 5            | 5               | 0.17       |
| T4   | 50     | 50       | 8            | 8               | 0.16       |
| T5   | 100    | 100      | 10           | 12              | 0.10       |
| **Total** | | | | | **~1.03** |

### Very Hard Taskset (where PPO wins)

Used for the mixed-criticality evaluation. High enough utilization that misses are unavoidable even with variable execution, forcing the scheduler to choose which tasks to sacrifice.

| Task | Period | Deadline | WCET | Criticality | U |
|------|--------|----------|------|-------------|---|
| T0   | 10     | 10       | 5    | Critical    | 0.50 |
| T1   | 15     | 15       | 6    | Critical    | 0.40 |
| T2   | 20     | 20       | 7    | Critical    | 0.35 |
| T3   | 30     | 30       | 8    | Soft        | 0.27 |
| T4   | 50     | 50       | 12   | Soft        | 0.24 |
| T5   | 100    | 100      | 20   | Soft        | 0.20 |
| **Total** | | | | | **~1.87** |

---

## Results

### Mixed Criticality — Very Hard Taskset (U_nom ~1.87, Variable Exec)

This is the key result. Under heavy overload with mixed criticality, PPO learns to protect critical tasks by selectively sacrificing soft ones. EDF treats all deadlines equally and cannot make this distinction.

| Scheduler | Total Misses | Critical Misses | Soft Misses | Reward |
|-----------|-------------|----------------|-------------|--------|
| **PPO (ours)** | **18.7** | **2.0** | 16.7 | **555.1** |
| RMS | 19.3 | 3.2 | 16.1 | 530.0 |
| EDF | 21.4 | 4.9 | 16.5 | 491.4 |
| Round Robin | 59.8 | 50.8 | 9.0 | -630.3 |

**PPO beats EDF on critical task miss rate by 59%** (2.0 vs 4.9 critical misses). It also beats RMS (2.0 vs 3.2). PPO achieves the fewest total misses (18.7 vs 21.4 EDF) and the highest reward (555.1 vs 491.4 EDF).

### Uniform Criticality — Standard Tasksets (Variable Exec)

On easier tasksets where all tasks have equal weight and misses are rare, EDF remains optimal as theory predicts.

| Scheduler | Misses (Normal) | Misses (Stressed) |
|-----------|----------------|-------------------|
| EDF | 0.0 | 0.0 |
| RMS | 0.0 | 0.0 |
| PPO | 7.1 | 7.1 |
| Round Robin | 0.1 | 3.0 |

This is expected — EDF's optimality holds when U < 1.0 and all tasks are equally important. The value of PPO emerges only when these assumptions are violated (overload + mixed criticality).

### Why These Results Make Sense

EDF's optimality is proven under specific conditions: single core, preemptive, periodic tasks, equal importance, known WCET, U <= 1.0. Our PPO breaks three of these:

1. **U > 1.0** — overloaded, misses are unavoidable, scheduler must choose who to sacrifice
2. **Mixed criticality** — tasks have different importance levels, EDF cannot distinguish
3. **Variable execution** — PPO observes remaining work, EDF only sees deadlines

Under these violations, EDF's greedy "nearest deadline first" strategy becomes suboptimal because it wastes CPU protecting soft tasks that should be sacrificed.

---

## What's Done

### Python Training Pipeline
- Gymnasium environment with multi-objective reward, mixed criticality, variable exec
- 3-phase curriculum learning with randomized tasksets for generalization
- PPO training (2M timesteps, stable-baselines3, ReLU activation)
- Parallel hyperparameter sweep (108 configurations)
- EDF, RMS, Round Robin baselines for comparison
- Evaluation on 4 tasksets with critical/soft miss breakdown

### Bare-Metal RTOS (ARM Cortex-M4)
- Real preemptive scheduler: SysTick (1ms tick) + PendSV (context switch in assembly)
- Per-task 1KB stacks with Cortex-M exception frame initialization
- Q10 fixed-point NN inference baked into firmware
- 5.1KB code + 6.7KB RAM — fits in STM32F411 (512KB FLASH, 128KB RAM)

### xv6-riscv Integration
- RL scheduler integrated into MIT's xv6 teaching OS
- Two-tier scheduler: NN for RT tasks, round-robin fallback for shell/init
- 4 new syscalls: `rtregister`, `rtjobdone`, `rtstats`, `setscheduler`
- RT logic in `clockintr()`: deadline checks, job releases, work decrement
- `rtdemo` user program: launches 6 tasks, switchable NN/RR mode
- 54KB kernel with NN weights, builds clean with `make qemu CPUS=1`
- All existing xv6 functionality preserved (shell, ls, cat, etc.)

### Weight Export
- `export_weights.py` generates both Rust (`src/policy.rs`) and C (`kernel/policy.c`)
- Q10 format: weights × 1024, rounded to int, compiled as static arrays
- Architecture-independent: same weights run on ARM (Rust) and RISC-V (C)
