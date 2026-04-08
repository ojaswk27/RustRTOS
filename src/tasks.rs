/// Task entry-point functions.
///
/// Each task runs an infinite loop: burn CPU for some nop cycles
/// (simulating real-time work), then yield so the scheduler can
/// re-block us until the next period release.

use crate::scheduler;

/// Simulate computation with nop loops, then yield.
macro_rules! define_task {
    ($name:ident) => {
        pub fn $name() -> ! {
            loop {
                // The actual "work" is tracked by the scheduler decrementing
                // `remaining` each tick. The task body just needs to be running
                // (consuming CPU) so the scheduler has something to preempt.
                // When remaining hits 0, the SysTick handler blocks us.
                // We call yield_task which issues WFI to sleep until the
                // scheduler releases us again next period.
                cortex_m::asm::nop();
                scheduler::yield_task();
            }
        }
    };
}

define_task!(task0);
define_task!(task1);
define_task!(task2);
define_task!(task3);
define_task!(task4);
define_task!(task5);

/// Idle task — runs when no real task is ready.
pub fn idle_task() -> ! {
    loop {
        cortex_m::asm::wfi();
    }
}

/// Array of task entry points, indexed by task id.
pub static TASK_ENTRIES: [fn() -> !; 6] = [task0, task1, task2, task3, task4, task5];
