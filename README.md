# RL-Based Adaptive RTOS Scheduler

Trains a reinforcement learning agent to schedule periodic real-time tasks in simulation, then deploys the learned policy as a fixed-point neural network on a bare-metal ARM Cortex-M4 RTOS running in QEMU.

## Structure

```
rtos_env.py           Gymnasium environment simulating a tick-based RTOS
train.py              PPO training + baseline comparisons + plots
export_weights.py     Exports trained weights to JSON for Rust embedding
src/main.rs           Bare-metal entry point and task definitions
src/task.rs           Task struct and state machine
src/scheduler.rs      Tick-based preemptive scheduler
src/policy.rs         Fixed-point (Q10) neural network inference
memory.x              STM32F411 linker script
```

## Prerequisites

- Python 3.11+ with uv
- Rust with the `thumbv7em-none-eabihf` target
- qemu-system-arm

```
uv sync
rustup target add thumbv7em-none-eabihf
```

## Usage

Train the agent:

```
uv run python train.py
```

Export weights to JSON:

```
uv run python export_weights.py
```

Copy the `weights_q10` and `biases_q10` arrays from `policy_weights.json` into `src/policy.rs`, replacing the placeholder zero arrays. Layer 0 maps to W1/B1, layer 1 to W2/B2, layer 2 to W3/B3.

Build and run on QEMU:

```
cargo run --release
```

## How It Works

The Python environment simulates 6 periodic tasks with deadlines. Each tick, the agent picks which task to run. It gets +1 for completing a task on time and -2 for a deadline miss. PPO learns a policy over 500k steps, then the actor network (24 -> 32 -> 32 -> 7) is exported as Q10 fixed-point integers and hardcoded into the Rust binary.

The Rust side is a `no_std` bare-metal program targeting Cortex-M4. It runs the same tick loop -- release tasks, check deadlines, build a state vector, run the neural network, execute the chosen task -- and prints results via semihosting.

Training compares the learned policy against Round Robin, Rate Monotonic, and Earliest Deadline First on both a normal (U ~ 1.03) and stressed (U ~ 1.15) taskset. Both are intentionally overloaded so that no scheduler can meet all deadlines, making the comparison meaningful.

See `REPORT.md` for detailed documentation of each component.
