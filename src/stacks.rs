/// Static stack allocations for all tasks + the idle task.
///
/// Each stack is 1024 bytes (256 words), 8-byte aligned as required
/// by the ARM ABI. The idle task gets 512 bytes since it does nothing.

/// 1 KiB per task — enough for nop-loop workloads + ISR nesting.
const TASK_STACK_SIZE: usize = 1024;
/// Idle task barely uses any stack.
const IDLE_STACK_SIZE: usize = 512;

#[repr(C, align(8))]
pub struct TaskStack {
    pub data: [u8; TASK_STACK_SIZE],
}

#[repr(C, align(8))]
pub struct IdleStack {
    pub data: [u8; IDLE_STACK_SIZE],
}

/// Six task stacks + one idle task stack, all statically allocated.
/// `static mut` because the scheduler writes initial stack frames
/// into them at boot before tasks start.
pub static mut TASK_STACKS: [TaskStack; 6] = [
    TaskStack { data: [0; TASK_STACK_SIZE] },
    TaskStack { data: [0; TASK_STACK_SIZE] },
    TaskStack { data: [0; TASK_STACK_SIZE] },
    TaskStack { data: [0; TASK_STACK_SIZE] },
    TaskStack { data: [0; TASK_STACK_SIZE] },
    TaskStack { data: [0; TASK_STACK_SIZE] },
];

pub static mut IDLE_STACK: IdleStack = IdleStack { data: [0; IDLE_STACK_SIZE] };
