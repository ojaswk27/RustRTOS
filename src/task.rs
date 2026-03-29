/// Represents a periodic real-time task.
///
/// A task has a job pending when `ready == true`. It stays ready until it
/// either completes (remaining hits 0) or misses its deadline. There is no
/// intermediate "Running" state — the scheduler simply calls `tick_execute`
/// each tick it allocates CPU to this task.
#[derive(Clone, Copy)]
pub struct Task {
    pub id: usize,
    pub period: u32,
    pub deadline: u32,
    pub wcet: u32,
    pub remaining: u32,
    pub next_release: u32,
    pub abs_deadline: u32,
    pub ready: bool,
    pub last_scheduled: i32,  // tick of last CPU allocation; -1 if never scheduled
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
            ready: false,
            last_scheduled: -1,
            deadline_misses: 0,
        }
    }

    /// Release a new job. Called when the period boundary arrives.
    pub fn release(&mut self, tick: u32) {
        self.remaining = self.wcet;
        self.abs_deadline = tick + self.deadline;
        self.next_release = tick + self.period;
        self.ready = true;
    }

    /// Execute one tick of CPU work. Returns true if the task just completed.
    /// Caller must ensure `ready == true` and `remaining > 0` before calling.
    pub fn tick_execute(&mut self, tick: u32) -> bool {
        debug_assert!(self.ready && self.remaining > 0, "tick_execute called on non-ready task");
        self.last_scheduled = tick as i32;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.ready = false;
            true
        } else {
            false
        }
    }

    /// Check for a deadline miss. Returns true if a miss occurred.
    /// Must be called BEFORE release() on the same tick.
    pub fn check_deadline(&mut self, tick: u32) -> bool {
        if self.ready && tick >= self.abs_deadline {
            self.deadline_misses += 1;
            self.ready = false;
            self.remaining = 0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_release_sets_fields() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0);
        assert!(t.ready);
        assert_eq!(t.remaining, 3);
        assert_eq!(t.abs_deadline, 10);
        assert_eq!(t.next_release, 10);
        assert_eq!(t.last_scheduled, -1); // not yet scheduled
    }

    #[test]
    fn test_tick_execute_partial() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0);
        let done = t.tick_execute(0);
        assert!(!done);
        assert_eq!(t.remaining, 2);
        assert!(t.ready);
        assert_eq!(t.last_scheduled, 0);
    }

    #[test]
    fn test_tick_execute_completes() {
        let mut t = Task::new(0, 10, 10, 2);
        t.release(0);
        t.tick_execute(0);
        let done = t.tick_execute(1);
        assert!(done);
        assert_eq!(t.remaining, 0);
        assert!(!t.ready);
        assert_eq!(t.last_scheduled, 1);
    }

    #[test]
    fn test_check_deadline_no_miss_before_boundary() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline = 10
        assert!(!t.check_deadline(9));
        assert!(t.ready);
        assert_eq!(t.deadline_misses, 0);
    }

    #[test]
    fn test_check_deadline_miss_at_boundary() {
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline = 10
        assert!(t.check_deadline(10));
        assert!(!t.ready);
        assert_eq!(t.remaining, 0);
        assert_eq!(t.deadline_misses, 1);
    }

    #[test]
    fn test_check_deadline_not_ready() {
        let mut t = Task::new(0, 10, 10, 3);
        // Never released — ready is false
        assert!(!t.check_deadline(100));
        assert_eq!(t.deadline_misses, 0);
    }

    #[test]
    fn test_check_before_release_catches_miss() {
        // Critical ordering test: check_deadline at period boundary fires
        // and records a miss BEFORE release() overwrites abs_deadline.
        let mut t = Task::new(0, 10, 10, 3);
        t.release(0); // abs_deadline=10, next_release=10
        // At tick 10: check first, then release
        let miss = t.check_deadline(10);
        assert!(miss, "miss must be recorded at the period boundary");
        assert_eq!(t.deadline_misses, 1);
        // Now release the next job (scheduler calls this second)
        t.release(10);
        assert!(t.ready);
        assert_eq!(t.abs_deadline, 20);
        // Miss count survives the release
        assert_eq!(t.deadline_misses, 1);
    }
}
