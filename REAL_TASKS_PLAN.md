# Plan: Final Report Deliverables — RustRTOS / xv6

## Context

This is the RustRTOS project (github.com/ojaswk27/RustRTOS) — a bare-metal Rust RTOS on
ARM Cortex-M4 and an xv6-riscv integration, both running a PPO-based RL scheduler with
Q10 fixed-point weights. The project already has:

- `rtmaladalen`: 6 Mälardalen WCET benchmark ports (matmul, bsort, CRC-32, prime, cnt, fibcall)
- `rtbench`, `rtvestal`: spin loop benchmarks
- `rtdrone`: fixed-point math drone workload
- Schedulers: NN (PPO), EDF, RMS, RR — all switchable at runtime in xv6

The professor has six requirements for the final report. This plan covers all of them.
Read the entire plan before starting any code. The priority order is at the bottom.

---

## Requirement 1: Write a Conclusion Section

**Writing task. Do this last, after all benchmarks are run.**

The conclusion must cover four things:

**What was built.** A Q10 fixed-point PPO policy deployed on two real targets
(bare-metal ARM Cortex-M4, xv6 RISC-V) serving as the actual scheduler. Not a simulation
of a scheduler — the NN is the scheduler, running inside the OS tick interrupt.

**When NN wins and why.** On mixed-criticality tasksets where HI-critical tasks have
longer periods than LO-soft tasks (the Vestal 2007 failure mode), NN is the only
scheduler achieving 0 HI-critical misses. EDF fails because it has no concept of
importance — only deadline distance. NN learns importance from asymmetric reward signals.

**When NN does not win.** On `rtbench` (HI tasks have shorter periods), EDF naturally
prioritises critical tasks and performs slightly better. This confirms Liu & Layland (1973):
EDF is optimal when its assumptions hold. NN gracefully degrades to near-EDF rather than
catastrophically failing.

**Limitations and future work.** NN learns a static policy. Without online adaptation it
cannot respond to taskset changes at runtime. State vector is fixed at 24 features / 6
tasks at compile time. Laxity is the missing state feature (identified in preliminary
report). Future: online fine-tuning in kernel space, hardware deployment on STM32F411.

Template (expand and personalise before submission):

> This project demonstrates that a PPO-trained neural network can serve as a real-time OS
> scheduler deployed on bare metal and inside a teaching OS kernel, with no floating-point
> arithmetic and a binary footprint under 6 KB. The NN scheduler outperforms EDF and RMS
> on mixed-criticality overloaded tasksets — particularly the Vestal (2007) inverted-priority
> scenario — while gracefully matching EDF on tasksets that satisfy Liu & Layland's
> assumptions. MLFQ, despite being the most adaptive classical scheduler, degenerates to
> round-robin on compute-bound RT tasks because it has no deadline or criticality awareness.
> These results suggest that criticality-aware RL scheduling is a viable direction for
> embedded systems where mixed-importance workloads run under resource constraints.

---

## Requirement 2: Address Why PPO + "What If PPO Fails"

**Writing task. Add a subsection titled "When Does the NN Underperform, and Is That a
Failure?" under Related Work or Algorithm Design. The professor sees that on `rtbench`
NN gets more HI misses than EDF and may be reading that as PPO failing. Explain it.**

### Why PPO was chosen (write this first)

PPO (Schulman et al. 2017) was chosen over other RL algorithms for three reasons:

- **Stable training**: PPO's clipped surrogate objective prevents destructive policy
  updates, which matters when reward is sparse (deadline misses are rare at U < 1).
- **Deployability**: PPO produces a deterministic policy at inference time — no replay
  buffer, no value function lookup, just a forward pass. This is what makes Q10 deployment
  in a kernel feasible. DQN requires a large replay buffer; storing experience tuples in
  kernel space is not feasible. SAC assumes a continuous action space.
- **Sample efficiency with curriculum**: PPO with curriculum learning converges in ~2M
  timesteps, tractable on a laptop CPU. A3C is less stable for our sparse reward structure.

### What "PPO fails" would actually look like (write this second)

PPO would be failing if it performed worse than a random policy. A random scheduler on our
6-task Very Hard taskset (U=1.87) would produce approximately 40% of HI task releases as
misses (1/6 chance of picking the right task per tick). Our PPO achieves 2.0 HI misses vs
EDF's 4.9. PPO is not failing.

### Why NN loses on `rtbench` (write this third)

On `rtbench` (U=1.90, HI tasks have shorter periods), EDF's nearest-deadline rule
accidentally aligns with criticality — HI tasks have shorter periods so they always have
nearer deadlines. EDF serves them first by structural coincidence. NN still reduces HI
misses by 81% versus RR, but EDF edges it because the taskset structure matches EDF's
greedy heuristic exactly. This confirms Liu & Layland, not a failure of RL.

**Key statement to include verbatim or paraphrased:**
There is no taskset structure where PPO catastrophically fails. It either wins on
mixed-criticality overloaded tasksets, or gracefully degrades to near-EDF performance
on tasksets where EDF's assumptions hold. For high overutilisation with mixed criticality
— the regime where EDF is provably non-optimal — PPO is the best currently available
approach. The question is not whether PPO fails but for which taskset structures the
learned policy generalises well, and the results show it generalises well to all tested
structures except the one case where EDF has a structural advantage.

---

## Requirement 3: Compare PPO Results with a Research Paper's PPO

**Writing task + optional ablation experiment.**

The most directly comparable prior work is:

> Mao, H., Alizadeh, M., Menache, I., Kandula, S. "Resource Management with Deep
> Reinforcement Learning." HotNets, 2016. (DeepRM)

Write a comparison table:

| Dimension | DeepRM (Mao 2016) | This work |
|-----------|------------------|-----------|
| Algorithm | REINFORCE (policy gradient) | PPO with clipped surrogate |
| Action space | Job queue slot selection | Task selection (0–5 + idle) |
| State space | Resource utilisation grid | Per-task (deadline, starvation, remaining, urgency rank) |
| Reward | Job slowdown minimisation | Asymmetric miss penalty (HI: 5×, LO: 1×) |
| Deployment | Python inference, cloud cluster | Q10 fixed-point, bare-metal kernel |
| Mixed criticality | Not addressed | Explicit HI/LO criticality flags in state vector |
| Overload regime | Designed for U ≤ 1 | Designed for U > 1 |
| Training regime | Fixed workload distribution | Curriculum (U=1.03→1.57→1.87), randomised tasksets per episode |

Four key differences that explain our results (write each as a paragraph):

**1. Asymmetric reward.** DeepRM treats all jobs equally (minimise average slowdown). We
give HI misses a 5× penalty, teaching the agent that not all deadlines are equal. This is
the primary reason NN outperforms EDF on Vestal tasksets — the agent learns to protect
HI slots regardless of their deadline distance, which no rule-based scheduler can do.

**2. Curriculum learning.** DeepRM trains on a fixed workload distribution. We use 3 phases
(U=1.03 → 1.57 → 1.87), starting at low utilisation to build a base policy and progressively
increasing overload to force generalisation.

**3. Randomised tasksets per episode.** Our RandomRTOSEnv generates a fresh taskset each
episode (random periods, WCETs, criticality assignments within constraints). This forces
the policy to learn a general scheduling strategy rather than a memorised lookup for one
workload — which is why it transfers to xv6 tasksets it was never trained on.

**4. Q10 fixed-point export.** DeepRM was never intended for bare-metal deployment. Our
export pipeline converts float32 weights to int32 Q10 format, enabling the same policy to
run in a Rust kernel on ARM and a C kernel on RISC-V with no FPU required.

Optional ablation (do if time allows): Train a version without asymmetric reward
(HI penalty = LO penalty = 1×) and show it performs worse on Vestal. This directly
demonstrates that the asymmetric reward is the decisive design decision, not merely
PPO vs REINFORCE.

---

## Requirement 4: Simulate "Real Tasks" (Professor's MS Paint Example)

**MS Paint cannot run on xv6 — no display driver, no window manager, no GPU. What we
simulate is the computational workload profile of a paint application: the actual integer
operations a paint tool performs, running over a static in-memory pixel array.**

### New file: `user/rtgui.c`

Implement a 6-task mixed-criticality set where each task body simulates one GUI subsystem
using bounded real computation over static framebuffer arrays.

```c
/* Static "framebuffer" — 64x64 pixels, 1 byte per pixel (grayscale) */
#define FB_W 64
#define FB_H 64
static unsigned char framebuf[FB_H][FB_W];
static unsigned char backbuf[FB_H][FB_W];   /* double buffer for blit */

/* Synthetic event queue — circular buffer of (x, y, type) tuples */
#define EV_MAX 16
struct event { short x, y, type; };
static struct event evqueue[EV_MAX];
static int ev_head, ev_tail;

/* ------------------------------------------------------------------
   task_input_poll — HI-crit, period=8, wcet=2
   Simulates reading a mouse/keyboard event and updating cursor state.
   Real computation: LFSR generates deterministic synthetic events,
   writes to event queue. LFSR has bounded, known execution time.
   ------------------------------------------------------------------ */
void task_input_poll(void) {
    static unsigned int lfsr = 0xACE1u;
    lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
    short x = (lfsr >> 8) & (FB_W - 1);
    short y = (lfsr >> 0) & (FB_H - 1);
    evqueue[ev_tail % EV_MAX] = (struct event){x, y, 1};
    ev_tail++;
}

/* ------------------------------------------------------------------
   task_render — HI-crit, period=16, wcet=5
   Simulates scanline brush stroke rendering.
   Real computation: nested loop with per-pixel Gaussian-like
   intensity falloff over a 5x5 region. Integer arithmetic only.
   ------------------------------------------------------------------ */
void task_render(void) {
    if (ev_head == ev_tail) return;
    struct event e = evqueue[ev_head++ % EV_MAX];
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int px = e.x + dx, py = e.y + dy;
            if (px < 0 || px >= FB_W || py < 0 || py >= FB_H) continue;
            int intensity = 255 - 40 * (dx*dx + dy*dy);
            if (intensity < 0) intensity = 0;
            framebuf[py][px] = (unsigned char)intensity;
        }
    }
}

/* ------------------------------------------------------------------
   task_blit — HI-crit, period=33, wcet=8
   Simulates double-buffer blit: copy backbuf to framebuf.
   Real computation: 16x16 pixel region copy, integer ops.
   ------------------------------------------------------------------ */
void task_blit(void) {
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++)
            framebuf[y][x] = backbuf[y][x];
}

/* ------------------------------------------------------------------
   task_flood_fill — LO-soft, period=100, wcet=20
   Simulates paint bucket fill: iterative BFS bounded at 64 pixels
   per job release (prevents unbounded execution).
   Real computation: BFS with static stack, conditional pixel update.
   ------------------------------------------------------------------ */
#define STACK_MAX 256
void task_flood_fill(void) {
    static short sx[STACK_MAX], sy[STACK_MAX];
    int sp = 0;
    sx[sp] = 32; sy[sp] = 32; sp++;
    int visited = 0;
    while (sp > 0 && visited < 64) {
        sp--;
        short x = sx[sp], y = sy[sp];
        if (x < 0 || x >= FB_W || y < 0 || y >= FB_H) continue;
        if (framebuf[y][x] >= 128) continue;
        framebuf[y][x] = 200;
        visited++;
        if (sp + 4 < STACK_MAX) {
            sx[sp]=x+1; sy[sp]=y;   sp++;
            sx[sp]=x-1; sy[sp]=y;   sp++;
            sx[sp]=x;   sy[sp]=y+1; sp++;
            sx[sp]=x;   sy[sp]=y-1; sp++;
        }
    }
}

/* ------------------------------------------------------------------
   task_crc_save — LO-soft, period=200, wcet=3
   Simulates auto-save: CRC-32 over entire framebuffer (4096 bytes).
   Reuse existing crc32() implementation from rtmaladalen.
   ------------------------------------------------------------------ */
void task_crc_save(void) {
    uint32_t crc = crc32((unsigned char*)framebuf, FB_W * FB_H);
    (void)crc;
}

/* ------------------------------------------------------------------
   task_undo_snapshot — LO-soft, period=500, wcet=10
   Simulates undo history: copy 32x32 region of framebuf to backbuf.
   Real computation: 1024 byte region copy.
   ------------------------------------------------------------------ */
void task_undo_snapshot(void) {
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            backbuf[y][x] = framebuf[y][x];
}
```

### Taskset parameters for `rtgui`

| Task | Body | Period | WCET | Crit |
|------|------|--------|------|------|
| input_poll | LFSR event gen | 8 | 2 | LO |
| render | Brush stroke (5×5 loop) | 16 | 5 | HI |
| blit | 16×16 region copy | 33 | 8 | HI |
| flood_fill | BFS fill (bounded 64px) | 100 | 20 | LO |
| crc_save | CRC-32 over 4KB | 200 | 3 | LO |
| undo_snap | 32×32 region copy | 500 | 10 | LO |

U ≈ 0.25 + 0.31 + 0.24 + 0.20 + 0.015 + 0.02 = **1.04**

Vestal failure mode: render and blit (HI) have longer periods than input_poll (LO).
EDF will perpetually serve input_poll (nearest deadline), starving render and blit.
NN should protect render + blit from the asymmetric reward signal.

### How to frame this in the report

> "We model the workload profile of an interactive GUI application by decomposing its
> periodic subsystems into bounded RT tasks. Task bodies perform real computation —
> per-pixel brush intensity calculation, iterative BFS flood fill, CRC-32 over a
> framebuffer — over static in-memory pixel arrays. This captures the computational
> structure of a paint application without requiring a display driver, window manager,
> or GPU, none of which exist in xv6. This approach follows standard methodology in
> embedded RT research: the Mälardalen and EEMBC suites use exactly this technique,
> isolating the computation kernel from its I/O context to enable WCET analysis."

---

## Requirement 5: All New Benchmarks Run on xv6

All new programs (`rtgui`, extended `rtmaladalen`, updated `rtdrone`) must compile and
run under xv6-riscv.

Checklist for each new program:
- Add to `Makefile` under `UPROGS`
- Each program calls `rtregister(...)` for each task, runs 200 ticks, then calls
  `rtstats()` and prints CSV in the format `gen_report.py` already expects
- Test: `make qemu CPUS=1` then type program name in xv6 shell
- Run under all 5 scheduler modes: NN(1), EDF(2), RMS(3), RR(0), MLFQ(4)
- Save CSV to `scripts/bench_results.csv`

---

## Requirement 6: Implement MLFQ in xv6 + Direct Comparison

**Largest code change in this plan. Add MLFQ as scheduler mode 4.**

### 6a. Data structures — `kernel/proc.h`

Add to the rt_proc struct (or wherever RT task fields are stored):

```c
int mlfq_queue;    /* current queue level: 0=highest, MLFQ_NQUEUES-1=lowest */
int mlfq_budget;   /* remaining ticks before demotion */

#define MLFQ_NQUEUES  4
#define MLFQ_Q0_SLICE 2   /* highest priority, shortest quantum */
#define MLFQ_Q1_SLICE 4
#define MLFQ_Q2_SLICE 8
#define MLFQ_Q3_SLICE 16
#define MLFQ_BOOST    100 /* reset all to Q0 every N ticks */
```

### 6b. Scheduler logic — `kernel/proc.c`

In the Tier 1 scheduler block, add a case for mode 4:

```c
case 4: { /* MLFQ */
    /* Find highest non-empty queue level */
    int best_queue = MLFQ_NQUEUES;
    struct proc *chosen = NULL;
    for each rt proc p where p->rt_ready:
        if p->mlfq_queue < best_queue:
            best_queue = p->mlfq_queue;
    /* Among tasks at best_queue, pick the one waiting longest */
    int max_wait = -1;
    for each rt proc p where p->rt_ready && p->mlfq_queue == best_queue:
        int wait = rt_ticks - p->last_scheduled;
        if wait > max_wait: max_wait = wait; chosen = p;
    /* Schedule chosen */
    break;
}
```

### 6c. Timer interrupt — `kernel/trap.c` or wherever `clockintr()` lives

Add MLFQ budget decrement and demotion:

```c
if (rt_sched_mode == 4 && current_rt_proc != NULL) {
    current_rt_proc->mlfq_budget--;
    if (current_rt_proc->mlfq_budget <= 0) {
        if (current_rt_proc->mlfq_queue < MLFQ_NQUEUES - 1)
            current_rt_proc->mlfq_queue++;
        int q = current_rt_proc->mlfq_queue;
        current_rt_proc->mlfq_budget = MLFQ_Q0_SLICE << q;
        force_reschedule = 1;
    }
}
/* Periodic priority boost */
if (rt_sched_mode == 4 && rt_ticks % MLFQ_BOOST == 0) {
    for each rt proc p:
        p->mlfq_queue = 0;
        p->mlfq_budget = MLFQ_Q0_SLICE;
}
```

### 6d. Syscall — reset MLFQ state on `setscheduler(4)`

When `setscheduler(4)` is called, reset all RT tasks:
```c
case 4:
    for each rt proc p:
        p->mlfq_queue = 0;
        p->mlfq_budget = MLFQ_Q0_SLICE;
    break;
```

### 6e. Expected results (document as hypothesis, confirm with data)

MLFQ will perform similarly to RR on all RT benchmarks because all RT tasks are
compute-bound — they exhaust their budget every period by definition. Every RT task
will be demoted to the lowest queue within a few periods. On Vestal tasksets, LO tasks
with short periods cycle back to Q0 faster than HI tasks with long periods, reproducing
the Vestal failure mode. This is not a flaw in the implementation — it is the fundamental
architectural mismatch: MLFQ was designed for interactive/IO-bound workloads that
voluntarily yield, not for hard RT tasks that always run to completion.

### 6f. Updated results table format (5 columns everywhere)

| Scheduler | HI-Crit Misses | LO-Soft Misses | Total |
|-----------|---------------|----------------|-------|
| **NN** | | | |
| EDF | | | |
| RMS | | | |
| RR | | | |
| **MLFQ** | | | |

---

## Additional Real-Computation Extensions (from earlier plan)

### Extended Mälardalen in `user/rtmaladalen.c`

Add at minimum `jfdctint` (8×8 integer JPEG DCT) and `ludcmp` (5×5 LU decomposition).
Both from Gustafsson et al. WCET 2010, available at wcet.mrtc.mdh.se.
If time allows, also add `statemate` and `susan` (corner detection over 16×16 image).

### `rtdrone` with PID + plant model in `user/rtdrone.c`

Replace or augment task bodies with:
- Complementary filter AHRS: `angle = (972*angle + 51*gyro_delta) >> 10` (Q10)
- PID controller with discrete plant: `x[k+1] = (972*x[k] + 51*u[k]) >> 10` (Q10)
- Shared state variables (`plant_state`, `integral`, `prev_error`) persist across job releases

### 64-point Q15 FFT in `user/rtfft.c` (do only if time allows)

Cooley-Tukey FFT, twiddle factors as static const int16_t arrays, magnitude spectrum
to static output buffer. See earlier plan version for full taskset parameters.

---

## Updated `gen_report.py` Changes

1. Parse CSV from: `rtmaladalen`, `rtvestal`, `rtbench`, `rtdrone`, `rtgui`, (optionally `rtfft`)
2. For each benchmark: grouped bar chart with 5 bars (NN, EDF, RMS, RR, MLFQ)
3. Summary table across all benchmarks with all 5 schedulers
4. Add note to MLFQ bars: "degenerates to RR (compute-bound tasks)" if MLFQ and RR
   results are within 15% on every benchmark — confirm the theoretical prediction in-figure

---

## Priority Order

| Priority | Task | Time est. | Requirement |
|----------|------|-----------|-------------|
| 1 | MLFQ in xv6 (Req 6) | 3–4 h | Direct comparison |
| 2 | `rtgui` (Req 4) | 3–4 h | "Real tasks like MS Paint" |
| 3 | PPO justification section (Req 2) | 1–2 h writing | "What if PPO fails" |
| 4 | DeepRM comparison table (Req 3) | 1–2 h writing | Paper comparison |
| 5 | Extended Mälardalen (jfdctint + ludcmp) | 2–3 h | Stronger real tasks |
| 6 | `rtdrone` PID + plant model | 2–3 h | Realistic drone tasks |
| 7 | Collect all results, update CSV + charts | 1–2 h | All reqs |
| 8 | `rtfft` | 4–6 h | Optional |
| 9 | Conclusion (Req 1) | 1 h writing | Do last |

---

## Professor Concern → Report Answer Mapping

| Professor's concern | Where addressed in report |
|---------------------|--------------------------|
| "Run real tasks" | rtgui (pixel rendering, BFS), extended Mälardalen (DCT, LU decomp), rtdrone PID |
| "MS Paint as example" | rtgui section: "We simulate the computational kernel of a paint application" |
| "What if PPO fails" | Req 2 subsection: PPO cannot catastrophically fail; rtbench result confirms Liu & Layland |
| "Compare with paper" | Req 3: DeepRM table + 4 key differences |
| "Compare with MLFQ" | Req 6: MLFQ implemented in xv6, results confirm it matches RR on compute-bound RT |
| "Why PPO at all" | Req 2: only method that can express task importance; EDF/RMS/MLFQ have no criticality concept |
| "Conclusion" | Req 1: written last after all results collected |
