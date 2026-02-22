/// Task states in a preemptive RTOS.
/// Ready: released and waiting for CPU. Running: currently executing.
/// Blocked: waiting on a resource (unused here but included for completeness).
/// Completed: finished execution for this period.
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Completed,
}

/// Represents a periodic real-time task.
/// In real-time systems each task has a period (how often it runs), a deadline
/// (when it must finish by), and a worst-case execution time (WCET).
#[derive(Clone, Copy)]
pub struct Task {
    pub id: usize,
    pub period: u32,
    pub deadline: u32,
    pub wcet: u32,
    pub remaining: u32,
    pub next_release: u32,
    pub abs_deadline: u32,
    pub state: TaskState,
    pub deadline_misses: u32,
}

impl Task {
    pub const fn new(id: usize, period: u32, deadline: u32, wcet: u32) -> Self {
        Self {
            id,
            period,
            deadline,
            wcet,
            remaining: 0,
            next_release: 0,
            abs_deadline: 0,
            state: TaskState::Completed,
            deadline_misses: 0,
        }
    }

    /// Release a new job of this task. Called when the period boundary arrives.
    /// Sets remaining work to WCET and computes the absolute deadline.
    pub fn release(&mut self, tick: u32) {
        self.remaining = self.wcet;
        self.abs_deadline = tick + self.deadline;
        self.next_release = tick + self.period;
        self.state = TaskState::Ready;
    }

    /// Simulate one tick of execution. Returns true if the task just completed.
    pub fn tick_execute(&mut self) -> bool {
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.state = TaskState::Completed;
                return true;
            }
        }
        false
    }

    /// Check if this task missed its deadline. Returns true on a miss.
    pub fn check_deadline(&mut self, tick: u32) -> bool {
        if self.state == TaskState::Ready && tick >= self.abs_deadline {
            self.deadline_misses += 1;
            // Abandon this job — it will re-release next period
            self.state = TaskState::Completed;
            self.remaining = 0;
            return true;
        }
        false
    }
}
