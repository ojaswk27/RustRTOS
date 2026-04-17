#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"
#include "vm.h"

extern struct proc proc[];

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  kexit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return kfork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return kwait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int t;
  int n;

  argint(0, &n);
  argint(1, &t);
  addr = myproc()->sz;

  if(t == SBRK_EAGER || n < 0) {
    if(growproc(n) < 0) {
      return -1;
    }
  } else {
    // Lazily allocate memory for this process: increase its memory
    // size but don't allocate memory. If the processes uses the
    // memory, vmfault() will allocate it.
    if(addr + n < addr)
      return -1;
    if(addr + n > TRAPFRAME)
      return -1;
    myproc()->sz += n;
  }
  return addr;
}

uint64
sys_pause(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  if(n < 0)
    n = 0;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kkill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// Register calling process as a real-time task.
// args: rt_id, period, deadline, wcet, criticality
uint64
sys_rtregister(void)
{
  int rt_id, period, deadline, wcet, crit;
  argint(0, &rt_id);
  argint(1, &period);
  argint(2, &deadline);
  argint(3, &wcet);
  argint(4, &crit);

  if(rt_id < 0 || rt_id >= 6)
    return -1;
  if(period <= 0 || deadline <= 0 || wcet <= 0)
    return -1;

  // Check for duplicate rt_id
  struct proc *pp;
  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp != myproc() && pp->is_rt && pp->rt_id == rt_id)
      return -1;
  }

  struct proc *p = myproc();
  acquire(&p->lock);
  p->is_rt = 1;
  p->rt_id = rt_id;
  p->period = period;
  p->deadline = deadline;
  p->wcet = wcet;
  p->criticality = crit;
  p->remaining = 0;
  p->next_release = 0;  // will be released on next clockintr
  p->abs_deadline = 0;
  p->last_scheduled = -1;
  p->misses = 0;
  p->completions = 0;
  p->rt_ready = 0;
  p->mlfq_level = 0;
  release(&p->lock);

  return 0;
}

// Signal that current job is done; sleep until next period release.
uint64
sys_rtjobdone(void)
{
  struct proc *p = myproc();
  if(!p->is_rt)
    return -1;

  // Hold tickslock as condition lock — clockintr holds it too when releasing
  // jobs, so sleep() is atomic with respect to the wakeup. Never pass p->lock
  // to sleep() because sleep() itself acquires p->lock (double-acquire panic).
  acquire(&tickslock);
  // If rt_ready=1, the task completed its job this period (before deadline).
  // Count it here because the tick-based remaining path races with this syscall.
  if(p->rt_ready)
    p->completions++;
  p->rt_ready = 0;
  p->remaining = 0;
  sleep(p, &tickslock);   // releases tickslock, sleeps, reacquires on wakeup
  release(&tickslock);
  return 0;
}

// Read stats for RT task with given rt_id.
// Writes misses and completions to user pointers.
uint64
sys_rtstats(void)
{
  int rt_id;
  uint64 misses_addr, completions_addr;
  argint(0, &rt_id);
  argaddr(1, &misses_addr);
  argaddr(2, &completions_addr);

  struct proc *p;
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->is_rt && p->rt_id == rt_id){
      int m = p->misses;
      int c = p->completions;
      struct proc *me = myproc();
      if(copyout(me->pagetable, misses_addr, (char*)&m, sizeof(m)) < 0)
        return -1;
      if(copyout(me->pagetable, completions_addr, (char*)&c, sizeof(c)) < 0)
        return -1;
      return 0;
    }
  }
  return -1;
}

// Set scheduler mode: 1 = NN, 0 = round-robin for RT tasks
extern int use_nn_scheduler;

// Return remaining CPU budget (ticks) for the calling RT task.
// Task spins on this to know when its job's CPU budget is consumed.
uint64
sys_rtremaining(void)
{
  struct proc *p = myproc();
  if(!p->is_rt)
    return 0;
  return p->remaining;
}

uint64
sys_setscheduler(void)
{
  int mode;
  argint(0, &mode);
  // 0=RR, 1=NN, 2=EDF, 3=RMS, 4=MLFQ
  if(mode < 0 || mode > 4)
    return -1;
  use_nn_scheduler = mode;
  return 0;
}
