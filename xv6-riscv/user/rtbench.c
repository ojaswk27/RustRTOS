// rtbench.c — Realistic Workload RT Scheduler Benchmark
//
// Simulates 6 tasks named after real embedded/desktop app workloads.
// Runs one scheduler mode and prints a results table + CSV lines.
//
// Usage:
//   rtbench        — NN scheduler (default)
//   rtbench edf    — Earliest Deadline First
//   rtbench rms    — Rate Monotonic Scheduling
//   rtbench rr     — Round-Robin

#include "kernel/types.h"
#include "user/user.h"

struct task_cfg {
  char *name;
  int period;
  int deadline;
  int wcet;
  int criticality;
};

// Mixed-criticality workload based on embedded/IoT application patterns.
// U_total = 1.90, U_critical = 1.05 (intentionally overloaded)
static struct task_cfg tasks[] = {
  {"sensor_read",     10,  10,  4, 1},  // T0 critical: IMU/ADC at 100Hz
  {"control_loop",    20,  20,  7, 1},  // T1 critical: PID at 50Hz
  {"display_render",  33,  33, 10, 1},  // T2 critical: UI at 30fps
  {"network_send",   100, 100, 25, 0},  // T3 soft: telemetry at 10Hz
  {"data_logging",   200, 200, 60, 0},  // T4 soft: file write at 5Hz
  {"background_sync",500, 500,150, 0},  // T5 soft: cloud sync at 2Hz
};

#define NTASKS   6
#define SIM_TICKS 200

static const char *mode_name(int mode)
{
  if(mode == 1) return "NN";
  if(mode == 2) return "EDF";
  if(mode == 3) return "RMS";
  return "RR";
}

int
main(int argc, char *argv[])
{
  int pids[NTASKS];
  int mode = 1;  // default: NN

  if(argc > 1){
    if(strcmp(argv[1], "edf") == 0)       mode = 2;
    else if(strcmp(argv[1], "rms") == 0)  mode = 3;
    else if(strcmp(argv[1], "rr") == 0)   mode = 0;
    // else "nn" or anything else -> mode=1
  }

  setscheduler(mode);

  printf("==============================================\n");
  printf("  RT Scheduler Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  Realistic mixed-criticality workload\n");
  printf("  U_total=1.90  U_critical=1.05\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){
      printf("fork failed for task %d\n", i);
      exit(1);
    }
    if(pids[i] == 0){
      struct task_cfg *t = &tasks[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();  // sleep until first release

      while(1){
        while(rtremaining() > 0){
          for(volatile int spin = 0; spin < 10000; spin++)
            ;
        }
        rtjobdone();
      }
    }
  }

  printf("Running %d tasks for %d ticks...\n\n", NTASKS, SIM_TICKS);
  pause(SIM_TICKS);

  int total_misses = 0, total_completions = 0;
  int crit_misses = 0, soft_misses = 0;
  int crit_completions = 0, soft_completions = 0;

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *crit = tasks[i].criticality ? "CRIT" : "soft";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, crit, tasks[i].name, c, m);
    total_misses += m;
    total_completions += c;
    if(tasks[i].criticality){ crit_misses += m; crit_completions += c; }
    else                     { soft_misses += m; soft_completions += c; }
  }

  printf("  Total: completions=%d  misses=%d\n", total_completions, total_misses);
  printf("  Critical: completions=%d  misses=%d\n", crit_completions, crit_misses);
  printf("  Soft:     completions=%d  misses=%d\n", soft_completions, soft_misses);
  printf("==============================================\n\n");

  // CSV output for gen_report.py
  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("BENCH,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
