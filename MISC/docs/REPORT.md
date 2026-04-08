# RL-Based Adaptive RTOS Scheduler

An Operating Systems course project that trains a reinforcement learning agent to schedule periodic real-time tasks, then deploys the learned policy on a bare-metal ARM Cortex-M4 RTOS running in QEMU.

## Project Overview

The project has two components that share the same task model and scheduling interface:

1. **Python simulation** — a Gymnasium environment where a PPO agent learns to schedule 6 periodic tasks, evaluated against three classical baselines.
2. **Rust bare-metal RTOS** — a `no_std` binary targeting `thumbv7em-none-eabihf` (STM32F411) that embeds the trained neural network as fixed-point weights and runs the same scheduling loop on QEMU.

The core idea: instead of hard-coding a scheduling algorithm (Round Robin, Rate Monotonic, EDF), we let the agent discover its own policy by trial and error, then freeze that policy into a lookup table (neural network) small enough for a microcontroller.

## File Map

```
os_project/
├── rtos_env.py          Python — Gymnasium RTOS environment
├── train.py             Python — PPO training, baselines, plots
├── export_weights.py    Python — extract NN weights to JSON
├── Cargo.toml           Rust   — crate manifest
├── memory.x             Rust   — STM32F411 linker script
├── .cargo/config.toml   Rust   — cross-compile + QEMU runner config
└── src/
    ├── main.rs          Rust   — entry point, task definitions
    ├── task.rs          Rust   — Task struct and state machine
    ├── scheduler.rs     Rust   — tick-based preemptive scheduler
    └── policy.rs        Rust   — fixed-point NN inference
```

## Component 1 — Python Simulation and Training

### rtos_env.py

A custom `gymnasium.Env` that simulates a preemptive tick-based RTOS.

**Task model.** Each task is periodic: it releases a new job every `period` ticks, and that job must complete within `deadline` ticks of release. Each job requires `wcet` (worst-case execution time) ticks of CPU to finish. On each tick the agent picks one task to run; that task's remaining execution decrements by one. If a task reaches its deadline without completing, the job is abandoned and a deadline miss is recorded.

**State space.** A 24-element float32 vector (6 tasks x 4 features):

| Feature | Meaning | Range |
|---|---|---|
| `time_to_deadline` | Ticks until this task's deadline, normalized by max deadline | 0–1 |
| `time_since_scheduled` | Ticks since this task last got CPU, normalized by max period | 0–1 |
| `remaining / wcet` | Fraction of work left for this job | 0–1 |
| `is_ready` | Whether the task has a pending job | 0 or 1 |

Non-ready tasks emit all zeros, so the agent implicitly learns which tasks need attention.

**Action space.** Discrete(7) — actions 0–5 select a task, action 6 idles. If the agent picks a non-ready task, the tick is effectively wasted (treated as idle).

**Reward function.**

| Signal | Value | Purpose |
|---|---|---|
| Task completion | +1.0 | Reward finishing jobs |
| Deadline miss | -2.0 | Penalize lateness heavily |
| Per-tick cost | -0.01 | Encourage urgency, don't dawdle |
| Context switch | -0.05 | Discourage thrashing between tasks |

**Episode.** One hyperperiod of 300 ticks (LCM of all task periods). This is the shortest interval where every task has released exactly a whole number of jobs, giving a clean evaluation boundary.

**Tasksets defined.**

- `NORMAL_TASKSET` — total utilization ~1.03 (slightly overloaded; some misses are inevitable).
- `STRESSED_TASKSET` — total utilization ~1.15 (heavily overloaded; classical schedulers degrade).

### train.py

Trains a PPO agent and evaluates it against three classical real-time scheduling algorithms.

**PPO configuration.**

| Parameter | Value | Rationale |
|---|---|---|
| `net_arch` | [32, 32] | Matches the Rust inference network exactly |
| `n_steps` | 2048 | ~6.8 episodes per rollout buffer |
| `batch_size` | 64 | Standard mini-batch size for PPO |
| `n_epochs` | 10 | Multiple passes over rollout data |
| `learning_rate` | 3e-4 | Default PPO learning rate |
| `total_timesteps` | 500,000 | ~1,667 episodes of training |
| `device` | cpu | Small network, CPU is faster than GPU overhead |

**Baseline schedulers.**

- **Round Robin** — cycles through ready tasks in index order. Simple, fair, but oblivious to deadlines.
- **Rate Monotonic (RMS)** — always picks the ready task with the smallest period. Optimal among static-priority algorithms for implicit-deadline tasksets (Liu & Layland 1973).
- **Earliest Deadline First (EDF)** — always picks the ready task with the nearest absolute deadline. Optimal among dynamic-priority algorithms for uniprocessor systems.

Each baseline is a pure function that takes the task list and returns an action, run through the same environment.

**Evaluation.** After training, each scheduler (PPO + 3 baselines) is evaluated for 100 episodes on both tasksets. Metrics: average reward, average deadline misses per episode.

**Outputs.**

| File | Contents |
|---|---|
| `ppo_rtos_model/ppo_rtos.zip` | Saved PPO model (can be reloaded) |
| `training_reward.png` | Two-panel plot: reward curve and deadline miss rate over training |
| `comparison.png` | Bar chart comparing PPO vs RR vs RMS vs EDF on both tasksets |

### export_weights.py

Extracts the trained actor network weights from the saved PPO model and writes them to `policy_weights.json`.

The JSON contains both float and Q10 fixed-point (scaled by 1024, rounded to integer) representations of each layer. The Q10 values can be directly pasted into the Rust `policy.rs` arrays.

**JSON structure:**

```json
{
  "q10_scale": 1024,
  "layers": [
    {
      "weight_shape": [32, 24],
      "weights": [[...floats...]],
      "biases": [...floats...],
      "weights_q10": [[...ints...]],
      "biases_q10": [...ints...]
    },
    { "weight_shape": [32, 32], ... },
    { "weight_shape": [7, 32], ... }
  ]
}
```

## Component 2 — Bare-Metal Rust RTOS

A minimal `no_std` RTOS that runs on QEMU-emulated ARM Cortex-M4. No operating system, no allocator, no standard library — just the scheduler loop running directly on the hardware abstraction.

### Cargo.toml

Crate manifest targeting `thumbv7em-none-eabihf`. Dependencies:

| Crate | Purpose |
|---|---|
| `cortex-m` | Low-level ARM Cortex-M primitives |
| `cortex-m-rt` | Runtime startup (vector table, memory init) |
| `panic-halt` | Panic handler that halts the CPU |
| `cortex-m-semihosting` | Printf-style output through QEMU's semihosting |

Release profile uses `opt-level = "z"` (size optimization) and LTO for a minimal binary.

### memory.x

Linker script defining the STM32F411 memory layout:

- **FLASH**: 512 KB at `0x0800_0000` — where the program code and const data (including NN weights) live.
- **RAM**: 128 KB at `0x2000_0000` — stack, static variables, mutable data.

### .cargo/config.toml

- Sets default build target to `thumbv7em-none-eabihf`.
- Configures `cargo run` to launch QEMU with the right machine, CPU, and semihosting flags.
- Passes `-Tmemory.x` to the linker so it picks up our memory layout.

### src/task.rs

Defines the `Task` struct and `TaskState` enum.

**TaskState**: `Ready` (has a pending job), `Running` (currently on CPU), `Blocked` (waiting on a resource — included for completeness), `Completed` (job finished for this period).

**Task fields:**

| Field | Type | Meaning |
|---|---|---|
| `id` | usize | Task index (0–5) |
| `period` | u32 | Release interval in ticks |
| `deadline` | u32 | Relative deadline in ticks |
| `wcet` | u32 | Worst-case execution time in ticks |
| `remaining` | u32 | Ticks of work left for current job |
| `next_release` | u32 | Tick when next job releases |
| `abs_deadline` | u32 | Absolute deadline of current job |
| `state` | TaskState | Current state |
| `deadline_misses` | u32 | Cumulative miss count |

**Key methods:**

- `release(tick)` — starts a new job: sets remaining to WCET, computes absolute deadline.
- `tick_execute()` — runs one tick of work; returns true if the job just completed.
- `check_deadline(tick)` — if the task is ready and past its deadline, records a miss and abandons the job.

### src/scheduler.rs

The tick-based preemptive scheduler. Each tick follows this sequence:

1. **Release** — check each task; if `tick >= next_release`, release a new job.
2. **Deadline check** — if any ready task has passed its absolute deadline, record a miss and abandon it.
3. **Build state** — construct the 24-element Q10 fixed-point state vector matching the Python environment's observation.
4. **Infer** — call `policy::infer(&state)` to get the action (0–6).
5. **Execute** — if the action selects a ready task, decrement its remaining time. Track context switches.
6. **Advance tick.**

Logs progress every 50 ticks and prints final statistics (completions, misses, switches, per-task miss counts) via semihosting.

**State vector construction** uses Q10 fixed-point (multiply by 1024) for the same four features as Python: time-to-deadline, time-since-scheduled, remaining/WCET, is-ready. This ensures the Rust scheduler feeds the neural network the same numerical representation it was trained on.

### src/policy.rs

A 3-layer feedforward neural network implemented entirely in `i32` fixed-point arithmetic.

**Architecture**: 24 inputs -> 32 neurons (ReLU) -> 32 neurons (ReLU) -> 7 outputs (linear).

**Q10 format**: all weights and activations are integers representing `value * 1024`. After each layer's multiply-accumulate, the result is divided by 1024 to maintain scale. This avoids all floating-point operations, which matters for:
- Deterministic timing (no FPU pipeline stalls)
- Portability (works on Cortex-M variants without FPU)
- Predictable worst-case execution time (important in RTOS contexts)

**Overflow safety**: uses `saturating_add` and `saturating_mul` to prevent integer overflow from crashing the system. With 24 inputs and weight magnitudes typically under 1024, the accumulator stays well within `i32` range (~2.1 billion).

**Weight arrays** are currently placeholders (all zeros). After training, paste the `weights_q10` and `biases_q10` values from `policy_weights.json` into the `static` arrays `W1`, `B1`, `W2`, `B2`, `W3`, `B3`.

**Inference cost**: 24*32 + 32*32 + 32*7 = 2,016 multiply-accumulate operations per tick. At 100 MHz (typical Cortex-M4 clock), this completes in ~20 microseconds — negligible compared to a 1 ms tick period.

### src/main.rs

The `#[entry]` function — program starts here after the cortex-m-rt runtime initializes the stack and static memory.

1. Prints a startup banner via semihosting.
2. Creates the 6-task array with the same parameters as the Python training taskset.
3. Constructs a `Scheduler` and runs it for 300 ticks (one hyperperiod).
4. Calls `debug::exit(EXIT_SUCCESS)` to cleanly terminate the QEMU session.

## How to Use

### Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Rust via [rustup](https://rustup.rs/) with the `thumbv7em-none-eabihf` target
- `qemu-system-arm` (for running the Rust binary)

The Python deps and Rust target are already installed in this project. If setting up fresh:

```bash
# Python
uv sync

# Rust
rustup default stable
rustup target add thumbv7em-none-eabihf

# QEMU (Arch)
sudo pacman -S qemu-system-arm
# QEMU (Ubuntu/Debian)
sudo apt install qemu-system-arm
```

### Step 1 — Train the RL agent

```bash
uv run python train.py
```

This takes a few minutes. It will:
- Train PPO for 500k timesteps (~1,667 episodes)
- Save the model to `ppo_rtos_model/`
- Evaluate against Round Robin, RMS, and EDF on both tasksets
- Save `training_reward.png` and `comparison.png`

### Step 2 — Export weights

```bash
uv run python export_weights.py
```

Produces `policy_weights.json` with float and Q10 integer weight arrays.

### Step 3 — Paste weights into Rust

Open `policy_weights.json` and copy the `weights_q10` and `biases_q10` arrays for each layer into `src/policy.rs`, replacing the placeholder zero arrays:

- Layer 0 (shape 32x24) -> `W1` and `B1`
- Layer 1 (shape 32x32) -> `W2` and `B2`
- Layer 2 (shape 7x32) -> `W3` and `B3`

### Step 4 — Build and run on QEMU

```bash
cargo build --release
cargo run --release
```

`cargo run` invokes QEMU automatically (configured in `.cargo/config.toml`). You'll see semihosting output:

```
========================================
  RL-RTOS Scheduler — Cortex-M4 Demo
========================================

Scheduler starting for 300 ticks
tick=50 misses=0 completions=12 switches=8
tick=100 misses=1 completions=27 switches=15
...
=== Final Stats ===
Total ticks:     300
Completions:     78
Deadline misses: 3
Context switches:42
  Task 0: misses=0
  Task 1: misses=1
  ...
Scheduler finished. Halting.
```

## OS Concepts Demonstrated

| Concept | Where |
|---|---|
| Periodic task model | `task.rs` — period, deadline, WCET, release/completion cycle |
| Preemptive scheduling | `scheduler.rs` — any task can be preempted each tick |
| Deadline miss handling | `task.rs:check_deadline` — job abandonment on miss |
| Rate Monotonic scheduling | `train.py:rate_monotonic` — static priority by period |
| Earliest Deadline First | `train.py:edf` — dynamic priority by absolute deadline |
| Round Robin | `train.py:round_robin` — cyclic task selection |
| Context switching | `scheduler.rs` — tracked when the running task changes |
| CPU utilization / overload | Both tasksets have U > 1.0, forcing schedulability tradeoffs |
| Fixed-point arithmetic | `policy.rs` — Q10 format avoids FPU for deterministic timing |
| Bare-metal execution | `main.rs` — `no_std`, `no_main`, runs directly on hardware |
| Linker scripts | `memory.x` — defines FLASH/RAM regions for the target MCU |
| Semihosting I/O | All Rust output uses ARM semihosting (debug channel to host) |

## Design Decisions

**Why utilization > 1.0?** A system where all deadlines are trivially meetable doesn't differentiate schedulers. With U > 1.0, the scheduler must make tradeoffs about which tasks to prioritize, and the RL agent can potentially learn smarter tradeoffs than fixed heuristics.

**Why Q10 fixed-point?** The Cortex-M4 has a hardware FPU, but using fixed-point gives us bit-exact reproducibility across builds and targets, predictable execution time (no FPU pipeline variability), and it works on Cortex-M0/M3 variants without FPU.

**Why PPO?** PPO is the standard baseline for discrete-action RL. It's stable, sample-efficient, and well-supported by Stable-Baselines3. The small network (32,32) converges quickly.

**Why 300-tick episodes?** 300 = LCM(10, 15, 20, 30, 50, 100). One hyperperiod is the minimal interval where the schedule pattern fully repeats, making it the natural evaluation unit.
