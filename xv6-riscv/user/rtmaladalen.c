// rtmaladalen.c — Mälardalen WCET Benchmark Suite Ports
//
// Ports 6 programs from the Mälardalen WCET benchmark suite
// (https://www.mrtc.mdh.se/projects/wcet/benchmarks.html) as
// real-time task bodies inside xv6.
//
// Each task executes the benchmark function in a tight loop until
// its kernel CPU budget (remaining) hits zero, then calls rtjobdone.
// This gives realistic computation profiles vs pure spin loops.
//
// Benchmarks ported:
//   T0 [CRIT] matmul   — integer 10x10 matrix multiply
//   T1 [CRIT] bsort100 — bubble sort, 100 elements
//   T2 [CRIT] crc      — CRC-32 over 64-byte buffer
//   T3 [soft] prime    — primality test (trial division to sqrt(N))
//   T4 [soft] cnt      — count negatives in 100-element array
//   T5 [soft] fibcall  — iterative Fibonacci up to N=47
//
// Taskset: Vestal-inverted (soft=short period, crit=long period)
//   U_crit=0.51  U_soft=0.65  U_tot=1.16
//
// Usage:
//   rtmaladalen        — NN
//   rtmaladalen edf    — EDF
//   rtmaladalen rms    — RMS
//   rtmaladalen rr     — Round Robin

#include "kernel/types.h"
#include "user/user.h"

// ---------------------------------------------------------------
// Benchmark implementations (self-contained, no stdlib required)
// ---------------------------------------------------------------

// matmul: 10x10 integer matrix multiply  C = A * B
static void bench_matmul(volatile int *out)
{
  static int A[10][10], B[10][10];
  int C[10][10];
  // seed with simple pattern (volatile read to prevent hoisting)
  for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++){
      A[i][j] = (i * 3 + j * 7 + 1) & 0xFF;
      B[i][j] = (i * 5 + j * 2 + 3) & 0xFF;
    }
  for(int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++){
      int s = 0;
      for(int k = 0; k < 10; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  *out = C[9][9];  // prevent dead-code elimination
}

// bsort100: bubble sort 100 integers in-place
static void bench_bsort100(volatile int *out)
{
  int data[100];
  for(int i = 0; i < 100; i++) data[i] = (i * 17 + 3) & 0xFF;
  for(int i = 0; i < 99; i++)
    for(int j = 0; j < 99 - i; j++)
      if(data[j] > data[j+1]){
        int t = data[j]; data[j] = data[j+1]; data[j+1] = t;
      }
  *out = data[99];
}

// crc: CRC-32 over a 64-byte buffer (standard polynomial)
static unsigned int crc32_table[16];
static int crc_table_init = 0;

static void crc_init(void)
{
  unsigned int poly = 0xEDB88320u;
  for(unsigned int i = 0; i < 16; i++){
    unsigned int crc = i;
    for(int j = 0; j < 8; j++)
      crc = (crc >> 1) ^ (crc & 1 ? poly : 0);
    crc32_table[i] = crc;
  }
  crc_table_init = 1;
}

static void bench_crc(volatile int *out)
{
  if(!crc_table_init) crc_init();
  static const char data[64] =
    "The quick brown fox jumps over the lazy dog. 0123456789ABCDEF!";
  unsigned int crc = 0xFFFFFFFFu;
  for(int i = 0; i < 64; i++){
    unsigned int byte = (unsigned char)data[i];
    crc = (crc >> 4) ^ crc32_table[(crc ^ byte) & 0xF];
    crc = (crc >> 4) ^ crc32_table[(crc ^ (byte >> 4)) & 0xF];
  }
  *out = (int)(crc ^ 0xFFFFFFFFu);
}

// prime: trial division primality test
static void bench_prime(volatile int *out)
{
  // Test whether 999983 is prime (it is)
  int n = 999983;
  int is_prime = 1;
  if(n < 2){ is_prime = 0; }
  else {
    for(int i = 2; (long long)i * i <= n; i++){
      if(n % i == 0){ is_prime = 0; break; }
    }
  }
  *out = is_prime;
}

// cnt: count negative numbers in 100-element array
static void bench_cnt(volatile int *out)
{
  int data[100];
  for(int i = 0; i < 100; i++) data[i] = (i % 3 == 0) ? -(i + 1) : (i + 1);
  int count = 0;
  for(int i = 0; i < 100; i++) if(data[i] < 0) count++;
  *out = count;
}

// fibcall: iterative Fibonacci, F(47) = 2971215073
static void bench_fibcall(volatile int *out)
{
  unsigned int a = 0, b = 1;
  for(int i = 2; i <= 47; i++){
    unsigned int c = a + b;
    a = b; b = c;
  }
  *out = (int)b;
}

// ---------------------------------------------------------------
// Task configuration
// ---------------------------------------------------------------
struct mal_task {
  char *name;
  int   period;
  int   deadline;
  int   wcet;
  int   criticality;
  void (*fn)(volatile int *);  // benchmark function
};

// Vestal-inverted: critical tasks have LONGER periods than soft tasks.
// Soft tasks (T3-T5) have short periods → always urgent under EDF/RMS.
// Critical tasks (T0-T2) have long periods → EDF will starve them.
//
// U_crit = 12/60 + 8/50 + 6/40  = 0.200 + 0.160 + 0.150 = 0.510
// U_soft  = 3/10  + 3/15 + 3/20  = 0.300 + 0.200 + 0.150 = 0.650
// U_total = 1.160
static struct mal_task tasks[] = {
  {"matmul",   60, 60, 12, 1, bench_matmul   },  // T0 HI: intensive compute
  {"bsort100", 50, 50,  8, 1, bench_bsort100 },  // T1 HI: sort workload
  {"crc",      40, 40,  6, 1, bench_crc      },  // T2 HI: integrity check
  {"prime",    10, 10,  3, 0, bench_prime    },  // T3 LO: math probe
  {"cnt",      15, 15,  3, 0, bench_cnt      },  // T4 LO: monitoring
  {"fibcall",  20, 20,  3, 0, bench_fibcall  },  // T5 LO: background
};

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
  printf("  Maladalen Ports - xv6 (%s)\n", mode_name(mode));
  printf("  Real WCET benchmarks, Vestal-inverted\n");
  printf("  U_crit=0.51  U_soft=0.65  U_tot=1.16\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct mal_task *t = &tasks[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();

      volatile int sink = 0;
      while(1){
        while(rtremaining() > 0){
          t->fn(&sink);  // run the actual benchmark
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
    char *lv = tasks[i].criticality ? "HI" : "LO";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, lv, tasks[i].name, c, m);
    total_miss += m; total_comp += c;
    if(tasks[i].criticality){ crit_miss += m; crit_comp += c; }
    else                     { soft_miss += m; soft_comp += c; }
  }

  printf("  ---\n");
  printf("  Total:   completions=%d  misses=%d\n", total_comp, total_miss);
  printf("  HI-crit: completions=%d  misses=%d\n", crit_comp, crit_miss);
  printf("  LO-soft: completions=%d  misses=%d\n", soft_comp, soft_miss);
  printf("==============================================\n\n");

  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("MALA,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){ kill(pids[i]); wait(0); }
  exit(0);
}
