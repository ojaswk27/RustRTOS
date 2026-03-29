//! RL-Based Adaptive RTOS Scheduler — bare-metal entry point.
//!
//! Runs on ARM Cortex-M4 (STM32F411) under QEMU. Defines the same 6-task
//! periodic taskset used in Python training, then runs the scheduler for
//! one hyperperiod (300 ticks). Output goes via semihosting to the QEMU console.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]

mod policy;
mod scheduler;
mod task;

#[cfg(not(test))]
use cortex_m_rt::entry;
#[cfg(not(test))]
use cortex_m_semihosting::{debug, hprintln};
#[cfg(not(test))]
use panic_halt as _;

#[cfg(not(test))]
#[entry]
fn main() -> ! {
    let _ = hprintln!("========================================");
    let _ = hprintln!("  RL-RTOS Scheduler — Cortex-M4 Demo");
    let _ = hprintln!("========================================\n");

    // Same taskset as Python training: (period, deadline, wcet)
    // Total utilization ≈ 1.03 — intentionally overloaded.
    let tasks = [
        task::Task::new(0, 10, 10, 2),
        task::Task::new(1, 15, 15, 3),
        task::Task::new(2, 20, 20, 4),
        task::Task::new(3, 30, 30, 5),
        task::Task::new(4, 50, 50, 8),
        task::Task::new(5, 100, 100, 10),
    ];

    let mut sched = scheduler::Scheduler::new(tasks);

    // Run for one hyperperiod: LCM(10,15,20,30,50,100) = 300 ticks
    sched.run(300);

    let _ = hprintln!("\nScheduler finished. Halting.");
    debug::exit(debug::EXIT_SUCCESS);

    loop {}
}
