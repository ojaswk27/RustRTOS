// rtgui.c — GUI Application Workload Simulation
//
// Models the computational workload profile of an interactive paint application
// (like MS Paint) by decomposing its periodic subsystems into bounded RT tasks.
// Each task body performs real computation over static in-memory pixel arrays.
//
// This follows standard methodology in embedded RT research (Mälardalen/EEMBC
// suites): isolate the computation kernel from its I/O context to enable WCET
// analysis. Task bodies use only bounded loops, integer arithmetic, and
// static memory — no system calls inside the workload.
//
// Taskset (Vestal-inverted: HI tasks have LONGER periods than LO input_poll):
//   T0 input_poll  LO period=8,   wcet=2  — LFSR event generation
//   T1 render      HI period=16,  wcet=5  — per-pixel brush stroke loop
//   T2 blit        HI period=33,  wcet=8  — double-buffer region copy
//   T3 flood_fill  LO period=100, wcet=20 — bounded BFS paint-bucket fill
//   T4 crc_save    LO period=200, wcet=3  — CRC-32 over framebuffer
//   T5 undo_snap   LO period=500, wcet=10 — undo history snapshot
//
// U_total = 0.25+0.31+0.24+0.20+0.015+0.02 = 1.04
// Vestal scenario: EDF serves input_poll (deadline=8) before render (deadline=16),
// starving the HI render task. NN protects render and blit from asymmetric reward.
//
// Usage:
//   rtgui        — NN scheduler (default)
//   rtgui edf    — EDF
//   rtgui rms    — RMS
//   rtgui rr     — Round-Robin
//   rtgui mlfq   — MLFQ

#include "kernel/types.h"
#include "user/user.h"

// ---------------------------------------------------------------
// Static "framebuffer" — 64×64 pixels, 1 byte per pixel (grayscale)
// ---------------------------------------------------------------
#define FB_W 64
#define FB_H 64
static unsigned char framebuf[FB_H][FB_W];
static unsigned char backbuf[FB_H][FB_W];

// Synthetic event queue — circular buffer of (x, y, type) tuples
#define EV_MAX 16
struct gui_event { short x, y, type; };
static struct gui_event evqueue[EV_MAX];
static int ev_head, ev_tail;

// ---------------------------------------------------------------
// T0: input_poll — LO-soft, period=8, wcet=2
// LFSR generates deterministic synthetic pointer events.
// Bounded, known execution time.
// ---------------------------------------------------------------
static void task_input_poll(volatile int *out)
{
  static unsigned int lfsr = 0xACE1u;
  lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
  short x = (short)((lfsr >> 8) & (FB_W - 1));
  short y = (short)((lfsr >> 0) & (FB_H - 1));
  evqueue[ev_tail % EV_MAX].x = x;
  evqueue[ev_tail % EV_MAX].y = y;
  evqueue[ev_tail % EV_MAX].type = 1;
  ev_tail++;
  *out = (int)lfsr;
}

// ---------------------------------------------------------------
// T1: render — HI-critical, period=16, wcet=5
// Scanline brush stroke: per-pixel Gaussian-like intensity over 5×5 patch.
// Nested loop with integer arithmetic, bounded at 25 iterations.
// ---------------------------------------------------------------
static void task_render(volatile int *out)
{
  if(ev_head == ev_tail){ *out = 0; return; }
  struct gui_event e = evqueue[ev_head++ % EV_MAX];
  int result = 0;
  for(int dy = -2; dy <= 2; dy++){
    for(int dx = -2; dx <= 2; dx++){
      int px = (int)e.x + dx, py = (int)e.y + dy;
      if(px < 0 || px >= FB_W || py < 0 || py >= FB_H) continue;
      int intensity = 255 - 40 * (dx*dx + dy*dy);
      if(intensity < 0) intensity = 0;
      framebuf[py][px] = (unsigned char)intensity;
      result += intensity;
    }
  }
  *out = result;
}

// ---------------------------------------------------------------
// T2: blit — HI-critical, period=33, wcet=8
// Double-buffer blit: copy 16×16 region from backbuf to framebuf.
// 256 byte-copy operations, deterministic.
// ---------------------------------------------------------------
static void task_blit(volatile int *out)
{
  int sum = 0;
  for(int y = 0; y < 16; y++)
    for(int x = 0; x < 16; x++){
      framebuf[y][x] = backbuf[y][x];
      sum += framebuf[y][x];
    }
  *out = sum;
}

// ---------------------------------------------------------------
// T3: flood_fill — LO-soft, period=100, wcet=20
// Iterative BFS paint-bucket fill, bounded at 64 pixels per job.
// Static stack to avoid dynamic allocation.
// ---------------------------------------------------------------
#define FILL_STACK 256
static short fill_sx[FILL_STACK], fill_sy[FILL_STACK];

static void task_flood_fill(volatile int *out)
{
  int sp = 0, visited = 0;
  fill_sx[sp] = 32; fill_sy[sp] = 32; sp++;
  while(sp > 0 && visited < 64){
    sp--;
    short x = fill_sx[sp], y = fill_sy[sp];
    if(x < 0 || x >= FB_W || y < 0 || y >= FB_H) continue;
    if(framebuf[y][x] >= 128) continue;
    framebuf[y][x] = 200;
    visited++;
    if(sp + 4 < FILL_STACK){
      fill_sx[sp]=x+1; fill_sy[sp]=y;   sp++;
      fill_sx[sp]=x-1; fill_sy[sp]=y;   sp++;
      fill_sx[sp]=x;   fill_sy[sp]=y+1; sp++;
      fill_sx[sp]=x;   fill_sy[sp]=y-1; sp++;
    }
  }
  *out = visited;
}

// ---------------------------------------------------------------
// T4: crc_save — LO-soft, period=200, wcet=3
// CRC-32 over entire 4 KB framebuffer. Auto-save integrity check.
// ---------------------------------------------------------------
static unsigned int crc32_gui_table[16];
static int crc_gui_init = 0;

static void crc_gui_build(void)
{
  unsigned int poly = 0xEDB88320u;
  for(unsigned int i = 0; i < 16; i++){
    unsigned int crc = i;
    for(int j = 0; j < 8; j++) crc = (crc >> 1) ^ (crc & 1 ? poly : 0);
    crc32_gui_table[i] = crc;
  }
  crc_gui_init = 1;
}

static void task_crc_save(volatile int *out)
{
  if(!crc_gui_init) crc_gui_build();
  unsigned int crc = 0xFFFFFFFFu;
  unsigned char *buf = (unsigned char*)framebuf;
  for(int i = 0; i < FB_W * FB_H; i++){
    unsigned int byte = buf[i];
    crc = (crc >> 4) ^ crc32_gui_table[(crc ^ byte) & 0xF];
    crc = (crc >> 4) ^ crc32_gui_table[(crc ^ (byte >> 4)) & 0xF];
  }
  *out = (int)(crc ^ 0xFFFFFFFFu);
}

// ---------------------------------------------------------------
// T5: undo_snap — LO-soft, period=500, wcet=10
// Undo history: copy 32×32 region of framebuf to backbuf.
// 1024 byte-copy operations.
// ---------------------------------------------------------------
static void task_undo_snap(volatile int *out)
{
  int sum = 0;
  for(int y = 0; y < 32; y++)
    for(int x = 0; x < 32; x++){
      backbuf[y][x] = framebuf[y][x];
      sum += backbuf[y][x];
    }
  *out = sum;
}

// ---------------------------------------------------------------
// Task table
// ---------------------------------------------------------------
struct gui_task {
  char *name;
  int period, deadline, wcet, criticality;
  void (*fn)(volatile int *);
};

// U_total = 2/8 + 5/16 + 8/33 + 20/100 + 3/200 + 10/500 = 1.04
// Vestal-inverted: LO input_poll (T=8) has nearer deadlines than HI render (T=16)
static struct gui_task gui_tasks[] = {
  {"input_poll", 8,   8,   2, 0, task_input_poll},  // T0 LO: event gen 125Hz
  {"render",    16,  16,   5, 1, task_render    },  // T1 HI: brush stroke 62Hz
  {"blit",      33,  33,   8, 1, task_blit      },  // T2 HI: buf flip 30Hz
  {"flood_fill",100, 100, 20, 0, task_flood_fill},  // T3 LO: BFS fill 10Hz
  {"crc_save",  200, 200,  3, 0, task_crc_save  },  // T4 LO: integrity 5Hz
  {"undo_snap", 500, 500, 10, 0, task_undo_snap },  // T5 LO: snapshot 2Hz
};

#define NTASKS    6
#define SIM_TICKS 200

static const char *mode_name(int mode)
{
  if(mode == 1) return "NN";
  if(mode == 2) return "EDF";
  if(mode == 3) return "RMS";
  if(mode == 4) return "MLFQ";
  return "RR";
}

int
main(int argc, char *argv[])
{
  int pids[NTASKS];
  int mode = 1;

  if(argc > 1){
    if(strcmp(argv[1], "edf") == 0)        mode = 2;
    else if(strcmp(argv[1], "rms") == 0)   mode = 3;
    else if(strcmp(argv[1], "rr") == 0)    mode = 0;
    else if(strcmp(argv[1], "mlfq") == 0)  mode = 4;
  }

  setscheduler(mode);

  printf("==============================================\n");
  printf("  GUI Paint Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  Simulates MS Paint computational workload\n");
  printf("  U_total=1.04  HI=render+blit  LO=input+fill\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct gui_task *t = &gui_tasks[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();

      volatile int sink = 0;
      while(1){
        while(rtremaining() > 0){
          t->fn(&sink);
        }
        rtjobdone();
      }
    }
  }

  printf("Running %d tasks for %d ticks...\n\n", NTASKS, SIM_TICKS);
  pause(SIM_TICKS);

  int total_misses = 0, total_completions = 0;
  int hi_misses = 0, lo_misses = 0;
  int hi_completions = 0, lo_completions = 0;

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *crit = gui_tasks[i].criticality ? "HI" : "LO";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, crit, gui_tasks[i].name, c, m);
    total_misses += m;
    total_completions += c;
    if(gui_tasks[i].criticality){ hi_misses += m; hi_completions += c; }
    else                         { lo_misses += m; lo_completions += c; }
  }

  printf("  Total:    completions=%d  misses=%d\n", total_completions, total_misses);
  printf("  HI-crit:  completions=%d  misses=%d\n", hi_completions, hi_misses);
  printf("  LO-soft:  completions=%d  misses=%d\n", lo_completions, lo_misses);
  printf("==============================================\n\n");

  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("GUI,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, gui_tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
