// rtvestal.c — Vestal (2007) Mixed-Criticality Benchmark
//
// Implements the canonical MC scheduling scenario where EDF fails:
// HI-critical tasks have LONGER periods (distant deadlines) than
// LO-soft tasks (frequent, near deadlines). EDF blindly serves
// LO tasks first — HI tasks starve. NN was trained to protect
// criticality-weighted positions regardless of deadline distance.
//
// Taskset design:
//   U_LO  = 0.817  (soft tasks alone fill 82% CPU)
//   U_HI  = 0.250  (critical tasks need 25%)
//   U_tot = 1.067  (system is overloaded — misses forced)
//
//   EDF expected: serves T3-T5 first (deadlines 10,15,20),
//                 HI tasks (deadlines 50,75,100) routinely late.
//   NN  expected: protects slots 0-2 (trained on crit penalty -15).
//
// Usage:
//   rtvestal        — NN
//   rtvestal edf    — Earliest Deadline First
//   rtvestal rms    — Rate Monotonic
//   rtvestal rr     — Round Robin

#include "kernel/types.h"
#include "user/user.h"

struct mc_task {
  char *name;
  int period;
  int deadline;
  int wcet;
  int criticality;  // 1=HI, 0=LO
};

// Vestal-inverted taskset: LO tasks have SHORT periods and U_LO > 1
// (they alone overflow CPU), critical tasks need the residual.
// Under EDF, LO tasks always have nearer deadlines and monopolise CPU —
// HI tasks starve and miss. The NN was trained to protect criticality
// regardless of deadline proximity, so it should diverge from EDF here.
static struct mc_task vestal[] = {
  {"flight_ctrl",  50,  50,  5, 1},  // T0 HI: U=0.100
  {"safety_mon",   75,  75,  6, 1},  // T1 HI: U=0.080
  {"actuator_cmd", 100, 100, 7, 1},  // T2 HI: U=0.070
  {"sensor_poll",   5,   5,  2, 0},  // T3 LO: U=0.400
  {"telemetry",     8,   8,  3, 0},  // T4 LO: U=0.375
  {"log_write",    10,  10,  3, 0},  // T5 LO: U=0.300
};
// U_HI=0.25, U_LO=1.075 (overflows!), U_tot=1.325
// EDF serves LO first (deadlines 5,8,10 vs 50,75,100) -> HI starves

#define NTASKS    6
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
  int mode = 1;

  if(argc > 1){
    if(strcmp(argv[1], "edf") == 0)      mode = 2;
    else if(strcmp(argv[1], "rms") == 0) mode = 3;
    else if(strcmp(argv[1], "rr") == 0)  mode = 0;
  }

  setscheduler(mode);

  printf("==============================================\n");
  printf("  Vestal MC Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  HI tasks: long periods (deadline 50-100)\n");
  printf("  LO tasks: short periods (deadline 10-20)\n");
  printf("  U_HI=0.25  U_LO=1.08  U_tot=1.33\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct mc_task *t = &vestal[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();

      while(1){
        while(rtremaining() > 0){
          for(volatile int spin = 0; spin < 10000; spin++)
            ;
        }
        rtjobdone();
      }
    }
  }

  printf("Running for %d ticks...\n\n", SIM_TICKS);
  pause(SIM_TICKS);

  int total_miss = 0, crit_miss = 0, soft_miss = 0;
  int total_comp = 0, crit_comp = 0, soft_comp = 0;

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *lv = vestal[i].criticality ? "HI" : "LO";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, lv, vestal[i].name, c, m);
    total_miss += m; total_comp += c;
    if(vestal[i].criticality){ crit_miss += m; crit_comp += c; }
    else                      { soft_miss += m; soft_comp += c; }
  }

  printf("  ---\n");
  printf("  Total:    completions=%d  misses=%d\n", total_comp, total_miss);
  printf("  HI-crit:  completions=%d  misses=%d\n", crit_comp, crit_miss);
  printf("  LO-soft:  completions=%d  misses=%d\n", soft_comp, soft_miss);
  printf("==============================================\n\n");

  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("VESTAL,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, vestal[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){ kill(pids[i]); wait(0); }
  exit(0);
}
