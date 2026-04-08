//! RL-Based Adaptive RTOS Scheduler — bare-metal Cortex-M4.
//!
//! Real preemptive scheduler running on STM32F411 (QEMU).
//! SysTick fires every 1ms, the RL policy network decides which
//! task to run, and PendSV performs the actual context switch.
//!
//! Build & run:  cargo run --release

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

mod policy;
mod scheduler;
mod stacks;
mod switch;
mod task;
mod tasks;

use cortex_m::peripheral::syst::SystClkSource;
use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use panic_halt as _;

#[entry]
fn main() -> ! {
    let _ = hprintln!("========================================");
    let _ = hprintln!("  RL-RTOS Scheduler — Cortex-M4");
    let _ = hprintln!("  Real preemptive scheduling demo");
    let _ = hprintln!("========================================\n");

    let mut core = cortex_m::Peripherals::take().unwrap();

    // Configure SysTick: 1ms tick at 16MHz (QEMU default)
    core.SYST.set_clock_source(SystClkSource::Core);
    core.SYST.set_reload(15_999); // 16000 cycles = 1ms at 16MHz
    core.SYST.clear_current();
    core.SYST.enable_counter();
    core.SYST.enable_interrupt();

    // PendSV must be lowest priority so it runs after SysTick
    unsafe {
        core.SCB.set_priority(cortex_m::peripheral::scb::SystemHandler::PendSV, 0xFF);
    }

    let _ = hprintln!("SysTick configured: 1ms tick");
    let _ = hprintln!("Initializing {} tasks...", scheduler::NUM_TASKS);

    // Init task stacks and release all tasks at tick 0
    scheduler::init();

    let _ = hprintln!("Starting scheduler...\n");

    // Start the scheduler — this triggers the first PendSV and
    // we never return from here (SysTick drives everything).
    scheduler::start();

    // Unreachable — scheduler::start() triggers PendSV which
    // switches to the first task. SysTick handler exits via
    // semihosting when MAX_TICKS is reached.
    loop {
        cortex_m::asm::wfi();
    }
}
