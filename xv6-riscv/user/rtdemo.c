// rtdemo.c — RL Scheduler Demo
//
// Launches 6 periodic real-time tasks, waits for the simulation
// to run (350 ticks), then prints deadline miss and completion stats.
//
// Usage:
//   rtdemo        — run with NN scheduler (default)
//   rtdemo rr     — run with round-robin for comparison

#include "kernel/types.h"
#include "user/user.h"

struct rt_config {
  int period;
  int deadline;
  int wcet;
  int criticality;  // 1 = critical, 0 = soft
};

// Very hard taskset — high enough U that misses are unavoidable
// U_nom = 1.87, forces trade-off decisions
// T0-T2 = critical (sensors/control), T3-T5 = soft (logging/telemetry)
static struct rt_config taskset[] = {
  {10,  10,  5,  1},  // T0 critical
  {15,  15,  6,  1},  // T1 critical
  {20,  20,  7,  1},  // T2 critical
  {30,  30,  8,  0},  // T3 soft
  {50,  50,  12, 0},  // T4 soft
  {100, 100, 20, 0},  // T5 soft
};

#define NTASKS 6
#define SIM_TICKS 350

int
main(int argc, char *argv[])
{
  int pids[NTASKS];
  int use_nn = 1;

  if(argc > 1 && strcmp(argv[1], "rr") == 0)
    use_nn = 0;

  setscheduler(use_nn);

  printf("========================================\n");
  printf("  RL Scheduler Demo - xv6\n");
  printf("  Mode: %s\n", use_nn ? "Neural Network" : "Round Robin");
  printf("  Tasks: %d (T0-T2 critical, T3-T5 soft)\n", NTASKS);
  printf("========================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){
      printf("fork failed for task %d\n", i);
      exit(1);
    }
    if(pids[i] == 0){
      struct rt_config *t = &taskset[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("T%d: rtregister failed\n", i);
        exit(1);
      }

      // Sleep until first period release
      rtjobdone();

      // Periodic task loop:
      //   1. Spin until the kernel says our CPU budget (remaining) is consumed.
      //      The kernel decrements remaining each tick we are RUNNING, so this
      //      accurately measures actual CPU time — not wall clock.
      //   2. Call rtjobdone() to count completion and sleep until next release.
      while(1){
        // Poll remaining; between syscalls spin to keep CPU busy so the
        // kernel tick handler can decrement the budget while we run.
        while(rtremaining() > 0){
          for(volatile int spin = 0; spin < 10000; spin++)
            ;
        }
        rtjobdone();
      }
    }
  }

  printf("All %d tasks launched. Running for %d ticks...\n\n", NTASKS, SIM_TICKS);
  pause(SIM_TICKS);

  int total_misses = 0, total_completions = 0;
  int crit_misses = 0, soft_misses = 0;

  printf("========================================\n");
  printf("  Results (%s)\n", use_nn ? "NN" : "RR");
  printf("========================================\n");
  printf("  Task  Type      Completions  Misses\n");
  printf("  ----  --------  -----------  ------\n");

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *type = taskset[i].criticality ? "critical" : "soft    ";
    printf("  T%d    %s  %d          %d\n", i, type, c, m);
    total_misses += m;
    total_completions += c;
    if(taskset[i].criticality)
      crit_misses += m;
    else
      soft_misses += m;
  }

  printf("  ----  --------  -----------  ------\n");
  printf("  Total           %d          %d\n", total_completions, total_misses);
  printf("  Critical misses: %d\n", crit_misses);
  printf("  Soft misses:     %d\n", soft_misses);
  printf("========================================\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
