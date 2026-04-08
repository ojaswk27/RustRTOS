# RL-Based Adaptive RTOS Scheduler

Trains a reinforcement learning agent (PPO) to schedule periodic real-time tasks with **mixed criticality**, then deploys the learned policy as a fixed-point neural network in two targets:

1. **xv6-riscv** — MIT's teaching OS, with NN scheduler replacing round-robin
2. **Bare-metal ARM Cortex-M4** — real preemptive RTOS with SysTick/PendSV

Under heavy overload with mixed-criticality tasks, **PPO beats EDF on critical task miss rate by 59%** (2.0 vs 4.9 critical misses) by learning to selectively sacrifice soft tasks to protect critical ones.

## Key Result

Very Hard Taskset (U ~1.87, variable exec, mixed criticality):

| Scheduler | Total Misses | Critical Misses | Soft Misses |
|-----------|-------------|----------------|-------------|
| **PPO** | **18.7** | **2.0** | 16.7 |
| RMS | 19.3 | 3.2 | 16.1 |
| EDF | 21.4 | 4.9 | 16.5 |
| Round Robin | 59.8 | 50.8 | 9.0 |

## Structure

```
Python (training):
  rtos_env.py           Gymnasium env — tick-based RTOS with mixed criticality
  train.py              PPO training with 3-phase curriculum
  eval.py               Evaluation vs EDF/RMS/Round Robin (critical + soft misses)
  sweep.py              Parallel hyperparameter sweep
  export_weights.py     Export weights to Rust + C (Q10 fixed-point)

xv6-riscv (RISC-V deployment):
  xv6-riscv/kernel/proc.c       Two-tier scheduler (NN + round-robin fallback)
  xv6-riscv/kernel/trap.c       RT logic in clockintr() (deadlines, releases, work)
  xv6-riscv/kernel/policy.c     Generated Q10 NN inference (C)
  xv6-riscv/kernel/policy.h     NN constants
  xv6-riscv/kernel/sysproc.c    rtregister, rtjobdone, rtstats, setscheduler syscalls
  xv6-riscv/user/rtdemo.c       Demo: 6 RT tasks, prints miss/completion stats

Rust bare-metal (ARM deployment):
  src/main.rs           Entry point — SysTick/PendSV init
  src/scheduler.rs      SysTick handler — RL policy + scheduling
  src/switch.rs         PendSV context switch (inline ARM assembly)
  src/policy.rs         Generated Q10 NN inference (Rust)
  src/task.rs           Task Control Block + stack frame init
  src/stacks.rs         Static per-task stack allocations
  src/tasks.rs          Task entry-point functions
```

## Prerequisites

```bash
# Python
uv sync

# xv6 (RISC-V)
sudo pacman -S riscv64-linux-gnu-gcc qemu-system-riscv  # Arch
# sudo apt install gcc-riscv64-linux-gnu qemu-system-misc  # Ubuntu

# Bare-metal (ARM)
rustup target add thumbv7em-none-eabihf
sudo pacman -S qemu-system-arm
```

## Quick Start

### 1. Train

```bash
uv run python train.py    # 2M steps, ~10-30 min
```

### 2. Evaluate

```bash
uv run python eval.py     # PPO vs EDF vs RMS vs RR, critical/soft miss breakdown
```

### 3. Export weights

```bash
uv run python export_weights.py   # generates src/policy.rs + xv6-riscv/kernel/policy.c
```

### 4. Run on xv6

```bash
cd xv6-riscv
make qemu CPUS=1
# At the shell:
$ rtdemo        # NN scheduler
$ rtdemo rr     # round-robin for comparison
```

### 5. Run on bare-metal ARM

```bash
cargo run --release   # builds for Cortex-M4, launches QEMU
```

## How It Works

### Training

PPO is trained with **mixed criticality**: T0-T2 are critical (5x miss penalty), T3-T5 are soft (1x). The agent learns that protecting critical tasks is worth sacrificing soft ones under overload. Training uses random tasksets and a 3-phase curriculum (easy → medium → hard utilization).

### Deployment

The actor network (24→32→32→7, ReLU) is exported as Q10 fixed-point integer arrays. On target hardware, inference is ~2000 integer multiply-accumulate operations — runs in microseconds.

**xv6**: The scheduler checks for RUNNABLE RT tasks, builds the state vector, runs NN inference, and context-switches to the chosen task. Non-RT processes (shell, init) fall through to round-robin. All existing xv6 functionality is preserved.

**ARM**: SysTick fires every 1ms, runs the NN in the ISR, triggers PendSV for the actual context switch. Each task has its own 1KB stack.

### Why PPO Wins

EDF is provably optimal under: single core, periodic tasks, equal importance, known WCET, U <= 1.0. We break three assumptions:

- **U > 1.0** — overloaded, must choose who to sacrifice
- **Mixed criticality** — EDF treats all deadlines equally, PPO doesn't
- **Variable execution** — PPO observes remaining work, EDF is blind to it
