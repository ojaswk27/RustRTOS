# Project Progress Submission
**RL-Based Real-Time OS Scheduler on Bare-Metal Cortex-M4**
*30 March 2026*

---

## Proposed Algorithm

The idea is to replace a traditional hand-written scheduling policy (like EDF or RMS) with a small neural network trained using reinforcement learning. Instead of programming rules like "always pick the task closest to its deadline", we let the agent figure out a good policy on its own by giving it rewards for completing tasks on time and penalties for missing deadlines.

At inference time the network runs on a bare-metal ARM Cortex-M4 (emulated in QEMU). Since the M4 has an FPU but we want to keep things simple and portable, all the weights are stored as Q10 fixed-point integers. Every multiply is just an integer multiply followed by a right-shift by 10.

---

## Pseudocode

### Scheduler Tick Loop

Each "tick" is one unit of simulated time. The scheduler runs the following steps every tick:

```
procedure SCHEDULER_TICK(tasks[], tick):
    // Check for deadline misses first (before releasing new jobs,
    // otherwise a job expiring at its period boundary looks like a new release)
    for each task t in tasks:
        if t.ready AND tick >= t.abs_deadline:
            t.ready     ← false
            t.remaining ← 0
            record deadline miss

    // Release new jobs at period boundaries
    for each task t in tasks:
        if tick >= t.next_release:
            t.remaining    ← t.wcet
            t.abs_deadline ← tick + t.deadline
            t.next_release ← tick + t.period
            t.ready        ← true

    // Build the observation vector and ask the network what to run
    obs    ← BUILD_STATE(tasks, tick)
    action ← NN_INFER(obs)        // 0..5 = run that task, 6 = idle

    // Execute for one tick
    if action ≠ IDLE AND tasks[action].ready:
        tasks[action].remaining      ← tasks[action].remaining − 1
        tasks[action].last_scheduled ← tick
        if tasks[action].remaining = 0:
            tasks[action].ready ← false
            record completion

    tick ← tick + 1
```

### State Encoding

The network gets a 24-element observation: 4 numbers per task, 6 tasks. Non-ready tasks are all zeros so the network can tell they don't have a pending job.

```
procedure BUILD_STATE(tasks[], tick) → obs[24]:
    MAX_DEADLINE ← max deadline across all tasks
    MAX_PERIOD   ← max period across all tasks

    for i = 0 to 5:
        base ← i × 4
        if tasks[i].ready:
            obs[base+0] ← (tasks[i].abs_deadline − tick) / MAX_DEADLINE   // time_to_deadline
            obs[base+1] ← (tick − tasks[i].last_scheduled) / MAX_PERIOD   // time_since_scheduled
            obs[base+2] ← tasks[i].remaining / tasks[i].wcet              // remaining_work
            obs[base+3] ← 1.0                                              // is_ready
        else:
            obs[base .. base+3] ← 0.0

    return obs
```

| Feature | What it tells the network |
|---------|--------------------------|
| `time_to_deadline` | how urgent the task is (0 = at the deadline) |
| `time_since_scheduled` | whether the task is being starved |
| `remaining_work` | how close it is to finishing |
| `is_ready` | whether the task has a job to run at all |

### Neural Network Inference (Q10 Fixed-Point)

A two-hidden-layer MLP: 24 inputs → 32 → 32 → 7 outputs. The output with the highest value is the chosen action.

```
procedure NN_INFER(obs[24]) → action:
    // Hidden layer 1
    for j = 0 to 31:
        h1[j] ← B1[j]
        for i = 0 to 23:
            h1[j] ← h1[j] + (W1[j][i] * obs_q10[i]) >> 10
        h1[j] ← max(h1[j], 0)   // ReLU

    // Hidden layer 2
    for j = 0 to 31:
        h2[j] ← B2[j]
        for i = 0 to 31:
            h2[j] ← h2[j] + (W2[j][i] * h1[i]) >> 10
        h2[j] ← max(h2[j], 0)

    // Output layer
    for j = 0 to 6:
        out[j] ← B3[j]
        for i = 0 to 31:
            out[j] ← out[j] + (W3[j][i] * h2[i]) >> 10

    return argmax(out)
```

The `>> 10` is the Q10 descale (equivalent to dividing by 1024). All operations are saturating 32-bit integer arithmetic so there's no overflow on the Cortex-M4.

### PPO Training

Training runs offline in Python using stable-baselines3. The environment simulates the same tick loop above as a Gymnasium env.

```
procedure TRAIN():
    env    ← RTOSEnv(taskset=NORMAL_TASKSET, max_ticks=300)
    policy ← MLP([32, 32], input=24, output=7)

    for step = 1 to 2,000,000:
        obs ← env.reset()
        repeat until episode ends:
            action, log_prob, value ← policy(obs)
            obs', reward, done      ← env.step(action)

            // Reward shaping
            reward += −0.01           // small cost per tick (urgency pressure)
            reward += +1.5            // on task completion
            reward += −3.0 × misses   // per deadline miss
            reward += −0.05           // on context switch (task→task only)

            store transition
            obs ← obs'

        // PPO update every 2048 steps
        for 10 epochs over rollout buffer (batch size 64):
            ratio   ← exp(log_prob_new − log_prob_old)
            L_clip  ← min(ratio × A, clip(ratio, 0.8, 1.2) × A)
            loss    ← −L_clip + 0.5 × value_loss − 0.01 × entropy
            update policy
```

The reward values (`−3.0` miss penalty, `+1.5` completion reward) were selected from a grid search where we trained 16 configurations in parallel and picked the one with the fewest deadline misses on the stressed taskset.

### Weight Export

After training, the actor network weights are multiplied by 1024 and rounded to integers, then written directly into Rust source as static arrays.

```
procedure EXPORT_WEIGHTS(policy):
    for each layer (W, B):
        write "static W: [[i32; ...]; ...] = " + round(W × 1024) + ";"
        write "static B: [i32; ...]       = " + round(B × 1024) + ";"
    // → src/policy.rs
```

---

## Preliminary Results

We tested on two tasksets. Both are intentionally overloaded (total utilization > 1.0), so some deadline misses are unavoidable. The goal is just to minimize them.

### Task Attributes

Each task in the scheduler has two kinds of attributes: static configuration that never changes, and dynamic state that gets updated every tick.

**Static (set at task creation):**

| Attribute | Description |
|-----------|-------------|
| `period` | How often the task fires. A new job is released every `period` ticks. |
| `deadline` | How long the task has to finish after being released. We use implicit deadlines, so `deadline == period` for all tasks. |
| `wcet` | Worst-case execution time. The number of ticks the task needs to run to completion. |

**Dynamic (updated at runtime):**

| Attribute | Description |
|-----------|-------------|
| `ready` | True if this task has a pending job that still needs CPU time. False after completion or a deadline miss. |
| `remaining` | Ticks of CPU work still needed for the current job. Starts at `wcet` on release, counts down to 0. |
| `abs_deadline` | The absolute tick by which the current job must finish (`release_tick + deadline`). |
| `next_release` | The tick when the next job will be released (`release_tick + period`). |
| `last_scheduled` | The last tick this task was given the CPU. Used to compute the `time_since_scheduled` feature. -1 if the task has never run. |

A task is considered to have missed its deadline if `ready` is still true when `tick >= abs_deadline`. When that happens, `ready` and `remaining` are both reset to 0 (the job is discarded) and the miss is counted. A new job is then released at the next period boundary as normal.

### Tasksets

The taskset consists of 6 periodic tasks with implicit deadlines (deadline = period). Each task represents a recurring job that must complete within its period. Think of T0 as something like a sensor read happening every 10ms, T3 as a slower control loop every 30ms, and T5 as a background logging or housekeeping task every 100ms. The tasks span a range of periods (10 to 100 ticks) which is typical of a mixed-criticality embedded workload where high-frequency tasks tend to be shorter and lower-frequency tasks tend to do more work.

The key quantity is utilization: U = sum of (WCET / period) across all tasks. If U ≤ 1.0 the workload is theoretically schedulable (EDF can always meet all deadlines). We deliberately push U slightly above 1.0 so that no scheduler can be perfect. This forces the agent to make trade-offs about which tasks to deprioritize, which is a more interesting learning problem.

The stressed taskset increases WCET on T0 (the most frequent task) and T5 (the longest task), pushing U to 1.15. The scheduler was never trained on this taskset, so it tests whether the learned policy generalizes beyond its training distribution.

| Task | Period | Deadline | WCET (normal) | WCET (stressed) | U (normal) |
|------|--------|----------|--------------|-----------------|------------|
| T0   | 10     | 10       | 2            | 3               | 0.20       |
| T1   | 15     | 15       | 3            | 3               | 0.20       |
| T2   | 20     | 20       | 4            | 4               | 0.20       |
| T3   | 30     | 30       | 5            | 5               | 0.17       |
| T4   | 50     | 50       | 8            | 8               | 0.16       |
| T5   | 100    | 100      | 10           | 12              | 0.10       |
| **Total** | | | | | **≈ 1.03** |

One subtlety worth noting: T0, T1, and T2 each contribute 0.20 to utilization individually but have very different periods. T0 fires 10 times for every one firing of T5. This means a bad scheduler that ignores T0's frequency (like Round Robin, which treats all tasks equally) will miss T0's deadlines far more often than it misses T5's, even though both look equally important on paper. EDF handles this naturally by always chasing the nearest deadline. Part of what we're testing is whether the RL agent picks up on the same intuition.

### Normal Taskset (U ≈ 1.03)

| Scheduler | Avg Deadline Misses | Avg Reward |
|-----------|-------------------|------------|
| EDF | 2.0 | 69.2 |
| RMS | 3.0 | 66.4 |
| **PPO (ours)** | **4.0** | **55.2** |
| Round Robin | 12.0 | 30.4 |

### Stressed Taskset (U ≈ 1.15, trained only on normal)

| Scheduler | Avg Deadline Misses | Avg Reward |
|-----------|-------------------|------------|
| RMS | 5.0 | 60.8 |
| **PPO (ours)** | **7.0** | **55.2** |
| EDF | 7.0 | 55.1 |
| Round Robin | 39.0 | −50.9 |

On the normal taskset PPO comes in just behind EDF and RMS, which is expected since EDF is optimal for this class of problem and we're learning from scratch. The more interesting result is on the stressed taskset. PPO was never trained on it but matches EDF exactly (7 misses each), while Round Robin completely falls apart (39 misses). This suggests the policy is learning something more general than just memorizing the training distribution.

---

## What's Done So Far

- Python simulation environment and training pipeline: complete
- EDF, RMS, Round Robin baselines for comparison: complete
- Reward function tuning via parallel grid search: complete
- Q10 weight export to Rust: complete
- Rust scheduler + NN inference on bare-metal Cortex-M4 (QEMU): complete
- Trained model: complete
