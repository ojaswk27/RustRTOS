// rtgui_b.c — GUI Benchmark Option B: Compositor Task (Vestal LO Overload)
//
// Higher-load variant of rtgui.c. Replaces undo_snap with a compositor task
// that simulates window compositing (4-layer max-blend over 32x32). The
// compositor is LO-soft with period=12 — shorter than HI render (period=16)
// and HI blit (period=33). This creates the Vestal (2007) failure scenario:
// EDF perpetually serves compositor (deadline=12) before render (deadline=16),
// starving the HI-critical render task.
//
// Taskset:
//   T0 input_poll  LO period=8,   wcet=2  — LFSR event generation
//   T1 render      HI period=16,  wcet=5  — 5x5 brush (same as rtgui)
//   T2 blit        HI period=33,  wcet=8  — 16x16 copy (same as rtgui)
//   T3 compositor  LO period=12,  wcet=8  — 4-layer max-blend 32x32 (NEW)
//   T4 flood_fill  LO period=100, wcet=20 — bounded BFS
//   T5 crc_save    LO period=200, wcet=3  — CRC-32
//
// U_HI = 5/16 + 8/33                           = 0.313 + 0.242 = 0.555
// U_LO = 2/8  + 8/12 + 20/100 + 3/200          = 0.250 + 0.667 + 0.200 + 0.015 = 1.132
// U_tot = 1.687
//
// Vestal structure: compositor (LO, T=12) has shorter period than render
// (HI, T=16) and blit (HI, T=33). EDF will serve compositor first in almost
// every scheduling cycle, creating systematic HI starvation.
//
// CSV prefix: GUIB
//
// Usage:
//   rtgui_b          — NN scheduler (default)
//   rtgui_b edf      — EDF
//   rtgui_b rms      — RMS
//   rtgui_b rr       — Round-Robin
//   rtgui_b mlfq     — MLFQ

#include "kernel/types.h"
#include "user/user.h"

#define FB_W 64
#define FB_H 64
static unsigned char framebuf[FB_H][FB_W];
static unsigned char backbuf[FB_H][FB_W];

#define N_LAYERS 4
static unsigned char layers[N_LAYERS][32][32];
static int layers_init = 0;

#define EV_MAX 16
struct gui_event_b { short x, y, type; };
static struct gui_event_b evqueue[EV_MAX];
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
// T1: render — HI-critical, period=16, wcet=5 (same as rtgui.c)
// 5x5 brush stroke with Gaussian-like intensity.
// ---------------------------------------------------------------
static void task_render(volatile int *out)
{
  if(ev_head == ev_tail){ *out = 0; return; }
  struct gui_event_b e = evqueue[ev_head++ % EV_MAX];
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
// T2: blit — HI-critical, period=33, wcet=8 (same as rtgui.c)
// Double-buffer blit: copy 16x16 region from backbuf to framebuf.
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
// T3: compositor — LO-soft, period=12, wcet=8 (NEW)
// Window compositor: max-blend 4 static layers into framebuf.
// Short period ensures EDF perpetually prefers this over HI render.
// ---------------------------------------------------------------
static void task_compositor(volatile int *out)
{
  if(!layers_init){
    for(int l = 0; l < N_LAYERS; l++)
      for(int y = 0; y < 32; y++)
        for(int x = 0; x < 32; x++)
          layers[l][y][x] = (unsigned char)((l * 64 + y + x) & 0xFF);
    layers_init = 1;
  }
  int sum = 0;
  for(int y = 0; y < 32; y++){
    for(int x = 0; x < 32; x++){
      unsigned char mx = 0;
      for(int l = 0; l < N_LAYERS; l++)
        if(layers[l][y][x] > mx) mx = layers[l][y][x];
      framebuf[y][x] = mx;
      sum += mx;
    }
  }
  *out = sum;
}

// ---------------------------------------------------------------
// T4: flood_fill — LO-soft, period=100, wcet=20
// ---------------------------------------------------------------
#define FILL_STACK_B 256
static short fill_sx[FILL_STACK_B], fill_sy[FILL_STACK_B];

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
    if(sp + 4 < FILL_STACK_B){
      fill_sx[sp]=x+1; fill_sy[sp]=y;   sp++;
      fill_sx[sp]=x-1; fill_sy[sp]=y;   sp++;
      fill_sx[sp]=x;   fill_sy[sp]=y+1; sp++;
      fill_sx[sp]=x;   fill_sy[sp]=y-1; sp++;
    }
  }
  *out = visited;
}

// ---------------------------------------------------------------
// T5: crc_save — LO-soft, period=200, wcet=3
// ---------------------------------------------------------------
static unsigned int crc32_b_table[16];
static int crc_b_init = 0;

static void crc_b_build(void)
{
  unsigned int poly = 0xEDB88320u;
  for(unsigned int i = 0; i < 16; i++){
    unsigned int crc = i;
    for(int j = 0; j < 8; j++) crc = (crc >> 1) ^ (crc & 1 ? poly : 0);
    crc32_b_table[i] = crc;
  }
  crc_b_init = 1;
}

static void task_crc_save(volatile int *out)
{
  if(!crc_b_init) crc_b_build();
  unsigned int crc = 0xFFFFFFFFu;
  unsigned char *buf = (unsigned char*)framebuf;
  for(int i = 0; i < FB_W * FB_H; i++){
    unsigned int byte = buf[i];
    crc = (crc >> 4) ^ crc32_b_table[(crc ^ byte) & 0xF];
    crc = (crc >> 4) ^ crc32_b_table[(crc ^ (byte >> 4)) & 0xF];
  }
  *out = (int)(crc ^ 0xFFFFFFFFu);
}

// ---------------------------------------------------------------
// Task table
// ---------------------------------------------------------------
struct guib_task {
  char *name;
  int period, deadline, wcet, criticality;
  void (*fn)(volatile int *);
};

static struct guib_task guib_tasks[] = {
  {"input_poll", 8,   8,   2, 0, task_input_poll},
  {"render",    16,  16,   5, 1, task_render    },
  {"blit",      33,  33,   8, 1, task_blit      },
  {"compositor",12,  12,   8, 0, task_compositor},
  {"flood_fill",100, 100, 20, 0, task_flood_fill},
  {"crc_save",  200, 200,  3, 0, task_crc_save  },
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
  printf("  GUI-B Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  Compositor (LO T=12) vs Render (HI T=16)\n");
  printf("  U_HI=0.55  U_LO=1.13  U_tot=1.69\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct guib_task *t = &guib_tasks[i];
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
    char *crit = guib_tasks[i].criticality ? "HI" : "LO";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, crit, guib_tasks[i].name, c, m);
    total_misses += m;
    total_completions += c;
    if(guib_tasks[i].criticality) hi_misses += m;
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
    printf("GUIB,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, guib_tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
