/* PendSV handler — performs the actual context switch.
 *
 * On entry the hardware has already stacked r0-r3, r12, lr, pc, xpsr
 * onto the current task's PSP. We save r4-r11 (callee-saved), swap
 * the stack pointer stored in the TCB, restore the next task's
 * r4-r11, update PSP, and return via EXC_RETURN so hardware pops
 * the rest.
 *
 * Globals (defined in scheduler.rs):
 *   CURRENT_TASK: *mut Task   — pointer to currently running TCB
 *   NEXT_TASK:    *mut Task   — pointer to TCB we're switching to
 *
 * Both TCBs have `sp: *mut u32` at offset 0.
 */

    .syntax unified
    .thumb
    .cpu cortex-m4

    .global PendSV
    .thumb_func
    .type PendSV, %function
PendSV:
    /* ── Save current context ───────────────────────────────── */
    mrs     r0, psp                 /* r0 = current task's PSP          */
    stmdb   r0!, {r4-r11}          /* push r4-r11 onto task stack      */

    /* Store updated sp into CURRENT_TASK->sp (offset 0) */
    ldr     r1, =CURRENT_TASK
    ldr     r2, [r1]               /* r2 = &current TCB                */
    str     r0, [r2, #0]           /* current_tcb.sp = r0              */

    /* ── Load next context ──────────────────────────────────── */
    ldr     r3, =NEXT_TASK
    ldr     r2, [r3]               /* r2 = &next TCB                   */
    str     r2, [r1]               /* CURRENT_TASK = NEXT_TASK         */

    ldr     r0, [r2, #0]           /* r0 = next_tcb.sp                 */
    ldmia   r0!, {r4-r11}          /* pop r4-r11 from next task stack  */

    msr     psp, r0                /* update PSP to next task's stack  */

    /* Return to thread mode using PSP (EXC_RETURN = 0xFFFFFFFD) */
    ldr     lr, =0xFFFFFFFD
    bx      lr

    .size PendSV, . - PendSV
