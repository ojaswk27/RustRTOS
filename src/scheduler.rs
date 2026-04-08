/// Real preemptive tick-based scheduler.
///
/// SysTick fires every tick (1ms at 16MHz). The handler:
///   1. Checks deadlines
///   2. Releases new jobs
///   3. Decrements current task's remaining work
///   4. Builds state vector and runs RL policy
///   5. Sets NEXT_TASK
///   6. Triggers PendSV for context switch
///
/// PendSV (lowest priority) runs after SysTick exits and performs
/// the actual register save/restore.

use core::sync::atomic::{AtomicU32, Ordering};

use cortex_m::peripheral::SCB;
#[cfg(not(test))]
use cortex_m_semihosting::hprintln;

use crate::policy;
use crate::stacks;
use crate::task::{self, Task, TaskState};
use crate::tasks;

#[cfg(test)]
macro_rules! hprintln {
    ($($arg:tt)*) => {{}};
}

// ── Constants ──────────────────────────────────────────────────────────

pub const NUM_TASKS: usize = 6;
const STATE_SIZE: usize = NUM_TASKS * 4;
const Q10: i32 = 1024;
const MAX_DEADLINE: i32 = 100;
#[allow(dead_code)]
const MAX_PERIOD: i32 = 100;
const MAX_TICKS: u32 = 300;
const PRINT_INTERVAL: u32 = 50;

// ── Global state ───────────────────────────────────────────────────────

/// Tick counter, incremented by SysTick.
static TICK: AtomicU32 = AtomicU32::new(0);

/// The task array — 6 periodic tasks.
static mut TASKS: [Task; NUM_TASKS] = [
    Task::new(0, 10, 10, 2),
    Task::new(1, 15, 15, 3),
    Task::new(2, 20, 20, 4),
    Task::new(3, 30, 30, 5),
    Task::new(4, 50, 50, 8),
    Task::new(5, 100, 100, 10),
];

/// Idle task TCB.
static mut IDLE_TASK: Task = Task::new(0xFF, 0, 0, 0);

/// Current and next task pointers, read/written by both Rust and PendSV asm.
#[no_mangle]
pub static mut CURRENT_TASK: *mut Task = core::ptr::null_mut();
#[no_mangle]
pub static mut NEXT_TASK: *mut Task = core::ptr::null_mut();

/// Context switch counter.
static CONTEXT_SWITCHES: AtomicU32 = AtomicU32::new(0);

// ── Init ────────────────────────────────────────────────────────────────

/// Initialise all task stacks and the idle task stack.
/// Must be called once from main() before `start()`.
pub fn init() {
    unsafe {
        // Init each task's stack frame
        for i in 0..NUM_TASKS {
            let sp = task::init_stack(
                &mut stacks::TASK_STACKS[i].data,
                tasks::TASK_ENTRIES[i],
            );
            TASKS[i].sp = sp;
        }

        // Init idle task stack
        let idle_sp = task::init_stack(
            &mut stacks::IDLE_STACK.data,
            tasks::idle_task,
        );
        IDLE_TASK.sp = idle_sp;
        IDLE_TASK.state = TaskState::Ready;

        // Start with idle task as current
        CURRENT_TASK = &mut IDLE_TASK as *mut Task;
        NEXT_TASK = &mut IDLE_TASK as *mut Task;

        // Release all tasks at tick 0
        for t in TASKS.iter_mut() {
            t.release(0);
        }
    }
}

/// Start the scheduler — switch from MSP to PSP, trigger first PendSV.
///
/// After this call, we're running in thread mode on PSP. SysTick
/// drives everything from here.
pub fn start() {
    unsafe {
        // Run the initial scheduling decision
        let state = build_state(0);
        let action = policy::infer(&state);
        NEXT_TASK = resolve_action(action);

        // Set PSP to idle task stack
        cortex_m::register::psp::write(IDLE_TASK.sp as u32);

        // Switch thread mode to use PSP: set CONTROL.SPSEL = 1
        let ctrl: u32;
        core::arch::asm!("mrs {}, CONTROL", out(reg) ctrl);
        core::arch::asm!("msr CONTROL, {}", in(reg) ctrl | 0x2);
        cortex_m::asm::isb();

        // Trigger first context switch
        SCB::set_pendsv();
    }
}

// ── SysTick handler ─────────────────────────────────────────────────────

/// SysTick exception handler — called every tick.
/// Runs the scheduling algorithm and triggers PendSV for context switch.
#[cortex_m_rt::exception]
fn SysTick() {
    let tick = TICK.fetch_add(1, Ordering::Relaxed) + 1;

    unsafe {
        // 1. Check deadlines (before releases)
        for t in TASKS.iter_mut() {
            t.check_deadline(tick);
        }

        // 2. Release tasks whose period boundary has arrived
        for t in TASKS.iter_mut() {
            if tick >= t.next_release {
                t.release(tick);
            }
        }

        // 3. Decrement current task's remaining work
        if !CURRENT_TASK.is_null() {
            let cur = &mut *CURRENT_TASK;
            if cur.state == TaskState::Running && cur.remaining > 0 {
                cur.last_scheduled = tick as i32;
                cur.remaining -= 1;
                if cur.remaining == 0 {
                    cur.completions += 1;
                    cur.state = TaskState::Blocked;
                }
            }
        }

        // 4. Build state vector and run RL policy
        let state = build_state(tick);
        let action = policy::infer(&state);

        // 5. Set NEXT_TASK
        let next = resolve_action(action);

        // Track context switches
        if !CURRENT_TASK.is_null() && next != CURRENT_TASK {
            let cur = &*CURRENT_TASK;
            let nxt = &*next;
            // Only count task-to-task switches (not idle transitions)
            if cur.id != 0xFF && nxt.id != 0xFF {
                CONTEXT_SWITCHES.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Mark the next task as Running
        (*next).state = TaskState::Running;
        NEXT_TASK = next;

        // 6. Print stats periodically
        if tick % PRINT_INTERVAL == 0 {
            let total_misses: u32 = TASKS.iter().map(|t| t.misses).sum();
            let total_completions: u32 = TASKS.iter().map(|t| t.completions).sum();
            let switches = CONTEXT_SWITCHES.load(Ordering::Relaxed);
            let _ = hprintln!(
                "tick={} misses={} completions={} switches={}",
                tick, total_misses, total_completions, switches
            );
        }

        // 7. Check if simulation is done
        if tick >= MAX_TICKS {
            print_final_stats(tick);
            cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
        }

        // 8. Trigger PendSV (lowest priority — runs after we return)
        SCB::set_pendsv();
    }
}

// ── State vector construction ───────────────────────────────────────────

/// Build the 24-element Q10 state vector matching the Python environment.
/// Layout: [time_to_deadline, time_since_scheduled, remaining_ratio, urgency_rank] x 6
fn build_state(tick: u32) -> [i32; STATE_SIZE] {
    let mut state = [0i32; STATE_SIZE];

    // Compute urgency ranks (nearest deadline = highest rank)
    let mut ready_indices: [Option<usize>; NUM_TASKS] = [None; NUM_TASKS];
    let mut n_ready = 0usize;

    unsafe {
        // Collect ready task indices
        for (i, t) in TASKS.iter().enumerate() {
            if t.state == TaskState::Ready || t.state == TaskState::Running {
                ready_indices[n_ready] = Some(i);
                n_ready += 1;
            }
        }

        // Sort by absolute deadline (insertion sort — only 6 elements max)
        for i in 1..n_ready {
            let mut j = i;
            while j > 0 {
                let a = ready_indices[j - 1].unwrap();
                let b = ready_indices[j].unwrap();
                if TASKS[a].abs_deadline > TASKS[b].abs_deadline {
                    ready_indices.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }

        // Build rank map: most urgent → n_ready/n_ready = 1.0, least → 1/n_ready
        let mut rank_q10 = [0i32; NUM_TASKS];
        if n_ready > 0 {
            for (rank, slot) in ready_indices[..n_ready].iter().enumerate() {
                let idx = slot.unwrap();
                // (n_ready - rank) / n_ready in Q10
                rank_q10[idx] = ((n_ready - rank) as i32 * Q10) / n_ready as i32;
            }
        }

        // Fill state vector
        for (i, t) in TASKS.iter().enumerate() {
            let base = i * 4;
            if t.state == TaskState::Ready || t.state == TaskState::Running {
                // time_to_deadline
                let ttd = if t.abs_deadline > tick {
                    (t.abs_deadline - tick) as i32 * Q10 / MAX_DEADLINE
                } else {
                    0
                };
                state[base] = clamp(ttd, 0, Q10);

                // time_since_scheduled (normalised by own period)
                let since = if t.last_scheduled >= 0 {
                    (tick as i32 - t.last_scheduled) * Q10 / t.period_ticks as i32
                } else {
                    Q10
                };
                state[base + 1] = clamp(since, 0, Q10);

                // remaining / wcet
                let rem_ratio = t.remaining as i32 * Q10 / t.wcet_ticks as i32;
                state[base + 2] = clamp(rem_ratio, 0, Q10);

                // urgency rank
                state[base + 3] = rank_q10[i];
            }
            // else: all zeros (not ready)
        }
    }

    state
}

/// Map policy action to a task pointer.
unsafe fn resolve_action(action: usize) -> *mut Task {
    if action < NUM_TASKS {
        let t = &mut TASKS[action];
        if t.state == TaskState::Ready || t.state == TaskState::Running {
            return t as *mut Task;
        }
    }
    // Idle — either action == 6 or selected task isn't ready
    &mut IDLE_TASK as *mut Task
}

/// Called when simulation reaches MAX_TICKS.
unsafe fn print_final_stats(tick: u32) {
    let total_misses: u32 = TASKS.iter().map(|t| t.misses).sum();
    let total_completions: u32 = TASKS.iter().map(|t| t.completions).sum();
    let switches = CONTEXT_SWITCHES.load(Ordering::Relaxed);

    let _ = hprintln!("\n========================================");
    let _ = hprintln!("  RL-RTOS Final Stats — {} ticks", tick);
    let _ = hprintln!("========================================");
    let _ = hprintln!("Total completions:    {}", total_completions);
    let _ = hprintln!("Total deadline misses:{}", total_misses);
    let _ = hprintln!("Context switches:     {}", switches);
    for t in TASKS.iter() {
        let _ = hprintln!(
            "  T{}: completions={} misses={}",
            t.id, t.completions, t.misses
        );
    }
    let _ = hprintln!("========================================");
}

/// Yield the current task — called by task functions when their work is done.
/// Sets state to Blocked and waits for interrupt (SysTick will re-release).
pub fn yield_task() {
    unsafe {
        if !CURRENT_TASK.is_null() {
            let cur = &mut *CURRENT_TASK;
            if cur.remaining == 0 {
                cur.state = TaskState::Blocked;
                cortex_m::asm::wfi();
            }
        }
    }
}

#[inline]
fn clamp(val: i32, lo: i32, hi: i32) -> i32 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}
