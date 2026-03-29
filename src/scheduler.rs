/// Tick-based preemptive scheduler.
///
/// Tick lifecycle (mirrors rtos_env.py exactly):
///   1. check_deadlines  — before releases, catches implicit-deadline misses
///   2. do_releases      — refresh jobs whose period boundary arrived
///   3. build_state      — construct 24-element Q10 observation
///   4. policy::infer    — NN picks action
///   5. execute          — run selected task for one tick
///   6. tick += 1
use crate::policy;
use crate::task::Task;

#[cfg(not(test))]
use cortex_m_semihosting::hprintln;

/// No-op stub so run() compiles under `cargo test --target x86_64-unknown-linux-gnu`.
#[cfg(test)]
macro_rules! hprintln {
    ($($arg:tt)*) => {
        {}
    };
}

const NUM_TASKS: usize = 6;
const STATE_SIZE: usize = NUM_TASKS * 4;
const Q10: i32 = 1024;
/// Largest deadline across both tasksets — used for normalization.
const MAX_DEADLINE: i32 = 100;
/// Largest period across both tasksets — used for normalization.
const MAX_PERIOD: i32 = 100;

pub struct Scheduler {
    pub tasks: [Task; NUM_TASKS],
    pub tick: u32,
    pub current_task: Option<usize>,
    pub total_misses: u32,
    pub total_completions: u32,
    pub context_switches: u32,
}

impl Scheduler {
    pub fn new(tasks: [Task; NUM_TASKS]) -> Self {
        Self {
            tasks,
            tick: 0,
            current_task: None,
            total_misses: 0,
            total_completions: 0,
            context_switches: 0,
        }
    }

    /// Step 1: record deadline misses. Must run before do_releases().
    fn check_deadlines(&mut self) {
        for t in self.tasks.iter_mut() {
            if t.check_deadline(self.tick) {
                self.total_misses += 1;
            }
        }
    }

    /// Step 2: release tasks whose period boundary has arrived.
    fn do_releases(&mut self) {
        for t in self.tasks.iter_mut() {
            if self.tick >= t.next_release {
                t.release(self.tick);
            }
        }
    }

    /// Step 3: build the Q10-encoded state vector sent to the policy.
    /// Layout: [ttd, tss, rem_ratio, is_ready] × 6 tasks.
    /// Non-ready tasks emit all zeros.
    fn build_state(&self) -> [i32; STATE_SIZE] {
        let mut state = [0i32; STATE_SIZE];
        for (i, t) in self.tasks.iter().enumerate() {
            let base = i * 4;
            if t.ready {
                // time_to_deadline: (abs_deadline - tick) / MAX_DEADLINE
                let ttd = if t.abs_deadline > self.tick {
                    (t.abs_deadline - self.tick) as i32 * Q10 / MAX_DEADLINE
                } else {
                    0
                };
                state[base] = ttd.clamp(0, Q10);

                // time_since_scheduled: (tick - last_scheduled) / MAX_PERIOD
                // 1.0 (Q10) if this task has never been scheduled.
                let since = if t.last_scheduled >= 0 {
                    ((self.tick as i32 - t.last_scheduled) * Q10 / MAX_PERIOD).clamp(0, Q10)
                } else {
                    Q10
                };
                state[base + 1] = since;

                // remaining / wcet
                state[base + 2] = (t.remaining as i32 * Q10 / t.wcet as i32).clamp(0, Q10);

                // is_ready
                state[base + 3] = Q10;
            }
        }
        state
    }

    /// Execute one scheduler tick.
    pub fn tick_once(&mut self) {
        // 1. Deadlines before releases
        self.check_deadlines();
        // 2. Release new jobs
        self.do_releases();
        // 3. Observe and decide
        let state = self.build_state();
        let action = policy::infer(&state);

        // 4. Count context switches (only when the new task is actually ready)
        if action < NUM_TASKS && self.tasks[action].ready {
            if let Some(prev) = self.current_task {
                if prev != action {
                    self.context_switches += 1;
                }
            }
        }

        // 5. Execute selected task (check ready, not a state enum)
        if action < NUM_TASKS && self.tasks[action].ready {
            if self.tasks[action].tick_execute(self.tick) {
                self.total_completions += 1;
            }
            self.current_task = Some(action);
        } else {
            self.current_task = None;
        }

        // 6. Advance tick
        self.tick += 1;
    }

    /// Run the scheduler for `total_ticks` ticks, logging every 50.
    pub fn run(&mut self, total_ticks: u32) {
        let _ = hprintln!("Scheduler starting for {} ticks", total_ticks);

        for _ in 0..total_ticks {
            self.tick_once();

            if self.tick % 50 == 0 {
                let _ = hprintln!(
                    "tick={} misses={} completions={} switches={}",
                    self.tick,
                    self.total_misses,
                    self.total_completions,
                    self.context_switches
                );
            }
        }

        let _ = hprintln!("\n=== Final Stats ===");
        let _ = hprintln!("Total ticks:     {}", self.tick);
        let _ = hprintln!("Completions:     {}", self.total_completions);
        let _ = hprintln!("Deadline misses: {}", self.total_misses);
        let _ = hprintln!("Context switches:{}", self.context_switches);
        for _t in &self.tasks {
            let _ = hprintln!("  Task {}: misses={}", _t.id, _t.deadline_misses);
        }
    }
}
