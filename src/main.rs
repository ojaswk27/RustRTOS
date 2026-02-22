//! RL-Based Adaptive RTOS Scheduler — bare-metal entry point.
//!
//! Runs on ARM Cortex-M4 (STM32F411) under QEMU. Defines the same 6-task
//! periodic taskset used in Python training, then runs the scheduler for
//! one hyperperiod (300 ticks). Output goes via semihosting to the QEMU console.

#![no_std]
#![no_main]

mod policy;
mod scheduler;
mod task;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_halt as _;

use scheduler::Scheduler;
use task::Task;

#[entry]
fn main() -> ! {
    let _ = hprintln!("========================================");
    let _ = hprintln!("  RL-RTOS Scheduler — Cortex-M4 Demo");
    let _ = hprintln!("========================================\n");

    // Same taskset as Python training: (period, deadline, wcet)
    // Total utilization ≈ 1.03 — intentionally overloaded to show
    // how the RL policy minimizes deadline misses under pressure.
    let tasks = [
        Task::new(0, 10, 10, 2),
        Task::new(1, 15, 15, 3),
        Task::new(2, 20, 20, 4),
        Task::new(3, 30, 30, 5),
        Task::new(4, 50, 50, 8),
        Task::new(5, 100, 100, 10),
    ];

    let mut sched = Scheduler::new(tasks);

    // Run for one hyperperiod: LCM(10,15,20,30,50,100) = 300 ticks
    sched.run(300);

    let _ = hprintln!("\nScheduler finished. Halting.");
    debug::exit(debug::EXIT_SUCCESS);

    loop {}
}
