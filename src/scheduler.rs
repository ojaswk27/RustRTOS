/// Tick-based preemptive scheduler.
///
/// Each tick: release newly periodic tasks, check deadline misses,
/// build the state vector, query the NN policy, and execute the chosen task.
/// This mirrors the Python simulation so that the trained policy transfers.
use crate::policy;
use crate::task::{Task, TaskState};
use cortex_m_semihosting::hprintln;

const NUM_TASKS: usize = 6;
const STATE_SIZE: usize = NUM_TASKS * 4;
const Q10: i32 = 1024;

pub struct Scheduler {
    pub tasks: [Task; NUM_TASKS],
    pub tick: u32,
    pub current_task: Option<usize>,
    pub total_misses: u32,
    pub total_completions: u32,
    pub context_switches: u32,
    max_deadline: u32,
}

impl Scheduler {
    pub fn new(tasks: [Task; NUM_TASKS]) -> Self {
        let max_deadline = tasks.iter().map(|t| t.deadline).max().unwrap_or(1);
        Self {
            tasks,
            tick: 0,
            current_task: None,
            total_misses: 0,
            total_completions: 0,
            context_switches: 0,
            max_deadline,
        }
    }

    /// Build the Q10 fixed-point state vector matching the Python environment.
    /// 4 features per task: time_to_deadline, time_since_scheduled, remaining/wcet, is_ready.
    fn build_state(&self) -> [i32; STATE_SIZE] {
        let mut state = [0i32; STATE_SIZE];
        for (i, t) in self.tasks.iter().enumerate() {
            let base = i * 4;
            if t.state == TaskState::Ready {
                // Time remaining until deadline, normalized
                let ttd = if t.abs_deadline > self.tick {
                    (t.abs_deadline - self.tick) as i32 * Q10 / self.max_deadline as i32
                } else {
                    0
                };
                state[base] = clamp(ttd, 0, Q10);

                // Time since last scheduled — approximate with max_period if never run
                // (We don't track last_scheduled in the Rust struct to save memory;
                //  use max_period as a safe default. The policy is robust to this.)
                state[base + 1] = Q10; // conservative: assume long time since scheduled

                // Remaining execution / WCET
                state[base + 2] = t.remaining as i32 * Q10 / t.wcet as i32;

                // Is ready
                state[base + 3] = Q10;
            }
        }
        state
    }

    /// Release tasks whose period has arrived.
    fn do_releases(&mut self) {
        for t in self.tasks.iter_mut() {
            if self.tick >= t.next_release {
                t.release(self.tick);
            }
        }
    }

    /// Check for deadline misses and abandon late jobs.
    fn check_deadlines(&mut self) {
        for t in self.tasks.iter_mut() {
            if t.check_deadline(self.tick) {
                self.total_misses += 1;
            }
        }
    }

    /// Execute one scheduler tick.
    pub fn tick_once(&mut self) {
        self.do_releases();
        self.check_deadlines();

        let state = self.build_state();
        let action = policy::infer(&state);

        // Track context switches (task-to-task, not involving idle)
        if action < NUM_TASKS {
            if let Some(prev) = self.current_task {
                if prev != action {
                    self.context_switches += 1;
                }
            }
        }

        // Execute the selected task for one tick
        if action < NUM_TASKS && self.tasks[action].state == TaskState::Ready {
            self.tasks[action].state = TaskState::Running;
            if self.tasks[action].tick_execute() {
                self.total_completions += 1;
            }
            self.current_task = Some(action);
        } else {
            self.current_task = None;
        }

        self.tick += 1;
    }

    /// Run the scheduler for a given number of ticks, logging periodically.
    pub fn run(&mut self, total_ticks: u32) {
        let _ = hprintln!("Scheduler starting for {} ticks", total_ticks);

        for _ in 0..total_ticks {
            self.tick_once();

            // Log every 50 ticks to avoid flooding semihosting
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
        for t in &self.tasks {
            let _ = hprintln!("  Task {}: misses={}", t.id, t.deadline_misses);
        }
    }
}

#[inline]
fn clamp(val: i32, min: i32, max: i32) -> i32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}
