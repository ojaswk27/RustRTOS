/// PendSV context-switch handler in inline assembly.
///
/// Saves r4-r11 to current task's stack, swaps the TCB pointer,
/// restores r4-r11 from the next task's stack, and returns via
/// EXC_RETURN so hardware pops the rest of the frame.

use core::arch::naked_asm;

/// PendSV handler — called by hardware at lowest exception priority.
///
/// CURRENT_TASK and NEXT_TASK are `*mut Task` globals defined in
/// scheduler.rs. Task.sp is at offset 0 in the TCB struct.
#[no_mangle]
#[unsafe(naked)]
pub unsafe extern "C" fn PendSV() {
    naked_asm!(
        // Save current context
        "mrs     r0, psp",           // r0 = current task's PSP
        "stmdb   r0!, {{r4-r11}}",   // push r4-r11 onto task stack

        // Store updated sp into CURRENT_TASK->sp (offset 0)
        "ldr     r1, =CURRENT_TASK",
        "ldr     r2, [r1]",          // r2 = &current TCB
        "str     r0, [r2, #0]",      // current_tcb.sp = r0

        // Load next context
        "ldr     r3, =NEXT_TASK",
        "ldr     r2, [r3]",          // r2 = &next TCB
        "str     r2, [r1]",          // CURRENT_TASK = NEXT_TASK

        "ldr     r0, [r2, #0]",      // r0 = next_tcb.sp
        "ldmia   r0!, {{r4-r11}}",   // pop r4-r11 from next task stack

        "msr     psp, r0",           // update PSP to next task's stack

        // Return to thread mode using PSP
        "ldr     lr, =0xFFFFFFFD",   // EXC_RETURN: thread mode, PSP
        "bx      lr",
    );
}
