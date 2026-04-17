// rtdrone.c — Drone Flight Controller RT Benchmark
//
// Simulates a drone's mixed-criticality task workload using Q10 fixed-point
// arithmetic with a discrete plant model. State persists across job releases.
//
// Q10 format: 1.0 = 1024 (2^10). All angles/velocities scaled by 1024.
// Discrete-time plant: x[k+1] = (972*x[k] + 51*u[k]) >> 10
//   where 972/1024 ≈ 0.950 (pole near but inside unit circle), 51/1024 ≈ 0.050.
// This models a first-order linear system with a 20Hz bandwidth.
//
// Complementary AHRS: angle = (972*angle + 51*gyro_delta) >> 10
// PID: err = ref - x; integral += err; deriv = err - prev_err; u = Kp*err + Ki*I + Kd*D
//
// Taskset (U_total=1.50, U_critical=1.05):
//   T0 imu_read       HI T=10  WCET=3  (100Hz sensor)
//   T1 ahrs_filter    HI T=10  WCET=4  (100Hz attitude)
//   T2 pid_control    HI T=20  WCET=5  (50Hz control)
//   T3 actuator_upd   HI T=20  WCET=2  (50Hz motor mix)
//   T4 telemetry      LO T=100 WCET=20 (10Hz GCS)
//   T5 data_log       LO T=200 WCET=50 (5Hz recorder)
//
// Usage:
//   rtdrone        — NN scheduler (default)
//   rtdrone edf    — EDF
//   rtdrone rms    — RMS
//   rtdrone rr     — Round-Robin
//   rtdrone mlfq   — MLFQ

#include "kernel/types.h"
#include "user/user.h"

// Q10: 1.0 = 1024
#define Q10_ONE 1024
static int q10_mul(int a, int b) { return (int)(((long long)a * b) >> 10); }
static int q10_abs(int x)        { return x < 0 ? -x : x; }

// Shared plant/controller state (allocated in static BSS, one instance per fork)
// These persist across rtjobdone() calls because they are outside the job loop.

// imu_read outputs synthetic accel+gyro
static volatile int imu_ax = 0, imu_ay = 0, imu_az = Q10_ONE;
static volatile int imu_gx = 0, imu_gy = 0, imu_gz = 0;

// AHRS quaternion state [q0 q1 q2 q3] in Q10
static volatile int ahrs_q[4] = {Q10_ONE, 0, 0, 0};

// Plant states for 3 axes (roll, pitch, yaw) in Q10
static volatile int plant_x[3] = {0, 0, 0};

// PID integrators and previous errors
static volatile int pid_integral[3]  = {0, 0, 0};
static volatile int pid_prev_err[3]  = {0, 0, 0};

// Motor setpoints (Q10 thrust percentage)
static volatile int motor[4] = {Q10_ONE/2, Q10_ONE/2, Q10_ONE/2, Q10_ONE/2};

// Reference angles (level flight)
static const int ref_angle[3] = {0, 0, 0};

struct drone_task {
  char *name;
  int period, deadline, wcet, criticality;
};

// U_total=1.50, U_critical=1.05
static struct drone_task drone_tasks[] = {
  {"imu_read",       10,  10,  3, 1},  // T0: 100Hz sensor
  {"ahrs_filter",    10,  10,  4, 1},  // T1: 100Hz attitude
  {"pid_control",    20,  20,  5, 1},  // T2: 50Hz PID
  {"actuator_upd",   20,  20,  2, 1},  // T3: 50Hz motor mix
  {"telemetry",     100, 100, 20, 0},  // T4: 10Hz GCS
  {"data_log",      200, 200, 50, 0},  // T5: 5Hz recorder
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
  printf("  Drone RT Benchmark - xv6 (%s)\n", mode_name(mode));
  printf("  Q10 AHRS+PID+plant model, persistent state\n");
  printf("  U_total=1.50  U_crit=1.05\n");
  printf("==============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct drone_task *t = &drone_tasks[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();

      while(1){
        while(rtremaining() > 0){
          switch(i){
            case 0: {
              // imu_read: synthetic gyro update (16-bit LFSR drives pseudo-sensor)
              static unsigned int lfsr = 0xDEAD;
              lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
              imu_gx = (int)(lfsr & 0x1F) - 16;    // ±16 LSB
              imu_gy = (int)((lfsr >> 5) & 0x1F) - 16;
              imu_gz = (int)((lfsr >> 10) & 0x7) - 4;
              imu_ax = 0; imu_ay = 0; imu_az = Q10_ONE;
              break;
            }
            case 1: {
              // ahrs_filter: Q10 complementary filter
              // angle[k+1] = (972*angle[k] + 51*gyro_delta) >> 10
              int gx = imu_gx, gy = imu_gy;
              int roll  = (int)ahrs_q[1];
              int pitch = (int)ahrs_q[2];
              roll  = (972 * roll  + 51 * gx) >> 10;
              pitch = (972 * pitch + 51 * gy) >> 10;
              ahrs_q[1] = roll;
              ahrs_q[2] = pitch;
              // quaternion magnitude normalise (q0 approximated)
              int mag = q10_abs(roll) + q10_abs(pitch) + Q10_ONE;
              ahrs_q[0] = (Q10_ONE * Q10_ONE) / (mag ? mag : 1);
              break;
            }
            case 2: {
              // pid_control: 3-axis PID with Q10 discrete plant
              // plant: x[k+1] = (972*x[k] + 51*u[k]) >> 10
              for(int ax = 0; ax < 3; ax++){
                int meas = (int)plant_x[ax];
                int err  = ref_angle[ax] - meas;
                pid_integral[ax] += err;
                if(pid_integral[ax] >  4096) pid_integral[ax] =  4096;
                if(pid_integral[ax] < -4096) pid_integral[ax] = -4096;
                int deriv = err - pid_prev_err[ax];
                pid_prev_err[ax] = err;
                // Kp=800, Ki=10, Kd=200 (all Q10-scaled)
                int u = q10_mul(800, err)
                      + q10_mul(10,  pid_integral[ax])
                      + q10_mul(200, deriv);
                // Update plant state
                plant_x[ax] = (972 * meas + 51 * u) >> 10;
              }
              break;
            }
            case 3: {
              // actuator_upd: mixer — convert roll/pitch PID output to motor thrusts
              int roll_out  = (int)plant_x[0];
              int pitch_out = (int)plant_x[1];
              int base = Q10_ONE / 2;
              motor[0] = base + roll_out - pitch_out;
              motor[1] = base - roll_out - pitch_out;
              motor[2] = base - roll_out + pitch_out;
              motor[3] = base + roll_out + pitch_out;
              break;
            }
            default: {
              // telemetry / data_log: burn time proportional to wcet
              for(volatile int spin = 0; spin < 5000; spin++) ;
              break;
            }
          }
        }
        rtjobdone();
      }
    }
  }

  printf("Running %d tasks for %d ticks...\n\n", NTASKS, SIM_TICKS);
  pause(SIM_TICKS);

  int total_misses = 0, crit_misses = 0, soft_misses = 0;
  int total_completions = 0, crit_completions = 0, soft_completions = 0;

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *crit = drone_tasks[i].criticality ? "CRIT" : "soft";
    printf("  T%d [%s] %s  completions=%d  misses=%d\n",
           i, crit, drone_tasks[i].name, c, m);
    total_misses += m;
    total_completions += c;
    if(drone_tasks[i].criticality){ crit_misses += m; crit_completions += c; }
    else                           { soft_misses += m; soft_completions += c; }
  }

  printf("  Total:    completions=%d  misses=%d\n", total_completions, total_misses);
  printf("  HI-crit:  completions=%d  misses=%d\n", crit_completions, crit_misses);
  printf("  LO-soft:  completions=%d  misses=%d\n", soft_completions, soft_misses);
  printf("==============================================\n\n");

  printf("CSV_BEGIN\n");
  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    printf("DRONE,%s,%d,%s,%d,%d\n",
           mode_name(mode), i, drone_tasks[i].name, c, m);
  }
  printf("CSV_END\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
