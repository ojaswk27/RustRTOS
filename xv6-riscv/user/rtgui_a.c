// rtgui_a.c — GUI Benchmark Option A: Heavier Render + Blit
//
// Higher-load variant of rtgui.c with:
//   T1 render:  7x7 brush stroke + 3x3 box blur pass (was 5x5 brush only)
//   T2 blit:    32x32 alpha blend at 75% (was 16x16 plain copy)
//
// Taskset parameters adjusted for U_HI ≈ 0.88, U_total ≈ 1.37:
//   T0 input_poll  LO period=8,   wcet=2  — LFSR event generation
//   T1 render      HI period=25,  wcet=12 — 7x7 brush + blur (heavy)
//   T2 blit        HI period=50,  wcet=20 — 32x32 alpha blend (heavy)
//   T3 flood_fill  LO period=100, wcet=20 — bounded BFS paint-bucket
//   T4 crc_save    LO period=200, wcet=3  — CRC-32 over framebuffer
//   T5 undo_snap   LO period=500, wcet=10 — undo snapshot copy
//
// U_HI  = 12/25 + 20/50                          = 0.480 + 0.400 = 0.880
// U_LO  = 2/8   + 20/100 + 3/200 + 10/500        = 0.250 + 0.200 + 0.015 + 0.020 = 0.485
// U_tot = 1.365
//
// Vestal structure preserved: LO input_poll (T=8) has nearer deadlines
// than HI render (T=25) and blit (T=50). EDF perpetually serves input_poll
// first, starving HI tasks once overload forces misses.
//
// CSV prefix: GUIA
//
// Usage:
//   rtgui_a          — NN scheduler (default)
//   rtgui_a edf      — EDF
//   rtgui_a rms      — RMS
//   rtgui_a rr       — Round-Robin
//   rtgui_a mlfq     — MLFQ

#include "kernel/types.h"
#include "user/user.h"

#define FB_W 64
#define FB_H 64
static unsigned char framebuf[FB_H][FB_W];
static unsigned char backbuf[FB_H][FB_W];

#define EV_MAX 16
struct gui_event_a { short x, y, type; };
static struct gui_event_a evqueue[EV_MAX];
static int ev_head, ev_tail;

// ---------------------------------------------------------------
// T0: input_poll — LO-soft, period=8, wcet=2
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
// T1: render — HI-critical, period=25, wcet=12
// Pass 1: 7x7 brush stroke with quadratic intensity falloff.
// Pass 2: 3x3 box blur over the 5x5 brushed region.
// ---------------------------------------------------------------
static void task_render(volatile int *out)
{
  if(ev_head == ev_tail){ *out = 0; return; }
  struct gui_event_a e = evqueue[ev_head++ % EV_MAX];
  int result = 0;

  /* Pass 1: 7x7 brush stroke */
  for(int dy = -3; dy <= 3; dy++){
    for(int dx = -3; dx <= 3; dx++){
      int px = (int)e.x + dx, py = (int)e.y + dy;
      if(px < 0 || px >= FB_W || py < 0 || py >= FB_H) continue;
      int intensity = 255 - 18 * (dx*dx + dy*dy);
      if(intensity < 0) intensity = 0;
      framebuf[py][px] = (unsigned char)intensity;
      result += intensity;
    }
  }

  /* Pass 2: 3x3 box blur over [-2,2] x [-2,2] region */
  for(int dy = -2; dy <= 2; dy++){
    for(int dx = -2; dx <= 2; dx++){
      int px = (int)e.x + dx, py = (int)e.y + dy;
      if(px < 1 || px >= FB_W-1 || py < 1 || py >= FB_H-1) continue;
      int sum = 0;
      for(int ky = -1; ky <= 1; ky++)
        for(int kx = -1; kx <= 1; kx++)
          sum += framebuf[py+ky][px+kx];
      framebuf[py][px] = (unsigned char)(sum / 9);
    }
  }

  *out = result;
}

// ---------------------------------------------------------------
// T2: blit — HI-critical, period=50, wcet=20
// Alpha blend backbuf into framebuf over 32x32 region.
// alpha=0.75: out = (192*back + 64*front) >> 8
// ---------------------------------------------------------------
static void task_blit(volatile int *out)
{
  int sum = 0;
  for(int y = 0; y < 32; y++){
    for(int x = 0; x < 32; x++){
      int blended = (192 * (int)backbuf[y][x] + 64 * (int)framebuf[y][x]) >> 8;
      framebuf[y][x] = (unsigned char)blended;
      sum += blended;
    }
  }
  *out = sum;
}

// ---------------------------------------------------------------
// T3: flood_fill — LO-soft, period=100, wcet=20
// ---------------------------------------------------------------
#define FILL_STACK_A 256
static short fill_sx[FILL_STACK_A], fill_sy[FILL_STACK_A];

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
    if(sp + 4 < FILL_STACK_A){
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
// ---------------------------------------------------------------
static unsigned int crc32_a_table[16];
static int crc_a_init = 0;

static void crc_a_build(void)
{
  unsigned int poly = 0xEDB88320u;
  for(unsigned int i = 0; i < 16; i++){
    unsigned int crc = i;
    for(int j = 0; j < 8; j++) crc = (crc >> 1) ^ (crc & 1 ? poly : 0);
    crc32_a_table[i] = crc;
  }
  crc_a_init = 1;
}

static void task_crc_save(volatile int *out)
{
  if(!crc_a_init) crc_a_build();
  unsigned int crc = 0xFFFFFFFFu;
  unsigned char *buf = (unsigned char*)framebuf;
  for(int i = 0; i < FB_W * FB_H; i++){
    unsigned int byte = buf[i];
    crc = (crc >> 4) ^ crc32_a_table[(crc ^ byte) & 0xF];
    crc = (crc >> 4) ^ crc32_a_table[(crc ^ (byte >> 4)) & 0xF];
  }
  *out = (int)(crc ^ 0xFFFFFFFFu);
}

// ---------------------------------------------------------------
// T5: undo_snap — LO-soft, period=500, wcet=10
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
struct guia_task {
  char *name;
  int period, deadline, wcet, criticality;
  void (*fn)(volatile int *);
};

static struct guia_task guia_tasks[] = {
  {"input_poll", 8,   8,   2,  0, task_input_poll},
  {"render",    25,  25,  12,  1, task_render    },
  {"blit",      50,  50,  20,  1, task_blit      },
  {"flood_fill",100, 100, 20,  0, task_flood_fill},
  {"crc_save",  200, 200,  3,  0, task_crc_save  },
  {"undo_snap", 500, 500, 10,  0, task_undo_snap },
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
  printf("  GUI-A Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  Heavy render (7x7+blur) + 32x32 alpha blit\n");
  printf("  U_HI=0.88  U_tot=1.37\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct guia_task *t = &guia_tasks[i];
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

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *crit = guia_tasks[i].criticality ? "HI" : "LO";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, crit, guia_tasks[i].name, c, m);
    total_misses += m;
    total_completions += c;
    if(guia_tasks[i].criticality) hi_misses += m;
    else                          lo_misses += m;
  }

  printf("  Total:    completions=%d  misses=%d\n", total_completions, total_misses);
  printf("  HI-crit:  misses=%d\n", hi_misses);
  printf("  LO-soft:  misses=%d\n", lo_misses);
  printf("==============================================\n\n");

  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("GUIA,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, guia_tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
