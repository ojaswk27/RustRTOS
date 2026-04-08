/// Task Control Block for a periodic real-time task.
///
/// Each task owns a saved stack pointer (`sp`) that PendSV uses for
/// context switching. The `sp` field MUST be at offset 0 so the
/// assembly handler can load/store it with a single `ldr/str r0, [r2]`.

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Dead,
}

#[repr(C)]
pub struct Task {
    /// Saved stack pointer — offset 0, accessed directly by PendSV asm.
    pub sp: *mut u32,

    pub id: usize,
    pub period_ticks: u32,
    pub deadline_ticks: u32,
    pub wcet_ticks: u32,
    pub remaining: u32,
    pub next_release: u32,
    pub abs_deadline: u32,
    pub last_scheduled: i32,
    pub state: TaskState,
    pub misses: u32,
    pub completions: u32,
}

/// Sentinel value for "never been scheduled".
const NEVER_SCHEDULED: i32 = -1;

impl Task {
    /// Create a new task. The stack pointer is null until `init_stack` is called.
    pub const fn new(id: usize, period: u32, deadline: u32, wcet: u32) -> Self {
        Self {
            sp: core::ptr::null_mut(),
            id,
            period_ticks: period,
            deadline_ticks: deadline,
            wcet_ticks: wcet,
            remaining: 0,
            next_release: 0,
            abs_deadline: 0,
            last_scheduled: NEVER_SCHEDULED,
            state: TaskState::Blocked,
            misses: 0,
            completions: 0,
        }
    }

    /// Release a new job at the given tick.
    pub fn release(&mut self, tick: u32) {
        self.remaining = self.wcet_ticks;
        self.abs_deadline = tick + self.deadline_ticks;
        self.next_release = tick + self.period_ticks;
        self.state = TaskState::Ready;
    }

    /// Check for deadline miss. Returns true if a miss was recorded.
    /// Must be called BEFORE release() on the same tick.
    pub fn check_deadline(&mut self, tick: u32) -> bool {
        if (self.state == TaskState::Ready || self.state == TaskState::Running)
            && tick >= self.abs_deadline
        {
            self.misses += 1;
            self.state = TaskState::Blocked;
            self.remaining = 0;
            true
        } else {
            false
        }
    }
}

/// Initialise a task's stack frame so PendSV can "return" to it.
///
/// The frame matches what Cortex-M hardware pushes on exception entry
/// plus the software-saved r4-r11 that PendSV manages.
///
/// Stack layout (high address at top):
/// ```text
///   xPSR          ← hardware frame (8 words)
///   PC  (entry)
///   LR  (task_exit)
///   R12
///   R3..R0
///   R11..R4       ← software frame (8 words, pushed by PendSV)
///   ^-- returned sp points here
/// ```
///
/// # Safety
/// `stack` must be a valid, exclusively owned, 8-byte aligned buffer.
/// `entry` must be a valid function pointer with signature `fn() -> !`.
pub unsafe fn init_stack(stack: &mut [u8], entry: fn() -> !) -> *mut u32 {
    let top = stack.as_mut_ptr().add(stack.len()) as *mut u32;

    // Hardware exception frame (pushed by hardware on exception entry)
    let hw = top.sub(8);
    hw.add(7).write(0x0100_0000); // xPSR — Thumb bit
    hw.add(6).write(entry as u32); // PC — task entry point
    hw.add(5).write(task_exit as *const () as u32); // LR — fallback if task returns
    hw.add(4).write(0); // R12
    hw.add(3).write(0); // R3
    hw.add(2).write(0); // R2
    hw.add(1).write(0); // R1
    hw.add(0).write(0); // R0

    // Software frame (r4-r11, pushed/popped by PendSV)
    let sw = hw.sub(8);
    for i in 0..8 {
        sw.add(i).write(0);
    }

    sw // initial saved sp
}

/// Called if a task function ever returns (it shouldn't).
fn task_exit() -> ! {
    loop {
        cortex_m::asm::bkpt();
    }
}
