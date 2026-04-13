// rtdrone.c — Drone Flight Controller RT Demo
//
// Simulates a drone's mixed-criticality task workload:
//   Critical: IMU read, AHRS filter, PID control, actuator update
//   Soft:     telemetry transmission, black-box data logging
//
// Task body: for critical tasks, computes a lightweight fixed-point
// operation representing the actual sensor/filter/control math.
// Designed to connect to a real IMU via UART (see Track B of the plan).
//
// Usage:
//   rtdrone        — NN scheduler (default)
//   rtdrone edf    — EDF
//   rtdrone rms    — RMS
//   rtdrone rr     — Round-Robin

#include "kernel/types.h"
#include "user/user.h"

struct drone_task {
  char *name;
  int period;
  int deadline;
  int wcet;
  int criticality;
};

// Drone taskset — based on real quadrotor flight controller timing.
// Inspired by PX4/ArduPilot task rates.
// U_total = 1.50, U_critical = 1.05
static struct drone_task drone_tasks[] = {
  {"imu_read",       10,  10,  3, 1},  // T0: 100Hz sensor read
  {"ahrs_filter",    10,  10,  4, 1},  // T1: 100Hz attitude estimate
  {"pid_control",    20,  20,  5, 1},  // T2: 50Hz roll/pitch/yaw PID
  {"actuator_update",20,  20,  2, 1},  // T3: 50Hz motor setpoints
  {"telemetry",     100, 100, 20, 0},  // T4: 10Hz GCS uplink
  {"data_log",      200, 200, 50, 0},  // T5: 5Hz black-box write
};

#define NTASKS    6
#define SIM_TICKS 200

// Lightweight fixed-point math mimicking real filter computations.
// q14 format: 1.0 = 16384
static int fp_mul(int a, int b) { return (int)(((long long)a * b) >> 14); }
static int fp_abs(int x)        { return x < 0 ? -x : x; }

// Simulate a Mahony complementary filter step (roll/pitch from accel+gyro).
// Just enough arithmetic to represent real compute load.
static void sim_ahrs_step(volatile int *state)
{
  int q0 = state[0], q1 = state[1], q2 = state[2], q3 = state[3];
  // Integration step: q += 0.5 * dt * q x omega (simplified)
  int gx = state[4], gy = state[5], gz = state[6];
  state[0] = q0 + fp_mul(-q1, gx) + fp_mul(-q2, gy) + fp_mul(-q3, gz);
  state[1] = q1 + fp_mul( q0, gx) + fp_mul(-q3, gy) + fp_mul( q2, gz);
  state[2] = q2 + fp_mul( q3, gx) + fp_mul( q0, gy) + fp_mul(-q1, gz);
  state[3] = q3 + fp_mul(-q2, gx) + fp_mul( q1, gy) + fp_mul( q0, gz);
  // Normalize magnitude (approximate)
  int mag = fp_abs(state[0]) + fp_abs(state[1]) + fp_abs(state[2]) + fp_abs(state[3]);
  if(mag == 0) mag = 1;
  for(int i = 0; i < 4; i++)
    state[i] = (state[i] * 16384) / mag;
}

// Simulate a PID controller step for one axis.
static void sim_pid_step(volatile int *s, int setpoint, int measurement)
{
  int error = setpoint - measurement;
  s[0] += error;                       // integral
  int derivative = error - s[1];       // derivative
  s[1] = error;                        // prev error
  s[2] = fp_mul(800, error) + fp_mul(10, s[0]) + fp_mul(200, derivative);  // output
}

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
    if(strcmp(argv[1], "edf") == 0)       mode = 2;
    else if(strcmp(argv[1], "rms") == 0)  mode = 3;
    else if(strcmp(argv[1], "rr") == 0)   mode = 0;
  }

  setscheduler(mode);

  printf("============================================\n");
  printf("  Drone RT Demo - xv6 (%s)\n", mode_name(mode));
  printf("  Critical: IMU+AHRS+PID+Actuator (U=1.05)\n");
  printf("  Soft:     Telemetry+DataLog (U=0.45)\n");
  printf("============================================\n\n");

  for(int i = 0; i < NTASKS; i++){
    pids[i] = fork();
    if(pids[i] < 0){ printf("fork failed\n"); exit(1); }
    if(pids[i] == 0){
      struct drone_task *t = &drone_tasks[i];
      if(rtregister(i, t->period, t->deadline, t->wcet, t->criticality) < 0){
        printf("%s: rtregister failed\n", t->name);
        exit(1);
      }
      rtjobdone();  // wait for first release

      volatile int sim_state[8] = {16384, 0, 0, 0, 100, 50, 200, 0};

      while(1){
        while(rtremaining() > 0){
          // Simulate the task's actual computation
          switch(i){
            case 0:  // imu_read: read sensor registers (simulate with spin)
              sim_state[4] += 1;  // gyro_x
              sim_state[5] -= 1;  // gyro_y
              break;
            case 1:  // ahrs_filter: attitude estimation
              sim_ahrs_step(sim_state);
              break;
            case 2:  // pid_control: 3-axis PID
              sim_pid_step(sim_state,     0, sim_state[0] >> 6);  // roll
              sim_pid_step(sim_state + 3, 0, sim_state[1] >> 6);  // pitch
              break;
            case 3:  // actuator_update: mix motor outputs
              sim_state[7] = (sim_state[2] + sim_state[5]) >> 1;
              break;
            default:  // telemetry / data_log: just burn time
              for(volatile int spin = 0; spin < 5000; spin++) ;
              break;
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

  printf("  Task              Type      Completions  Misses\n");
  printf("  ----------------  --------  -----------  ------\n");

  for(int i = 0; i < NTASKS; i++){
    int m = 0, c = 0;
    rtstats(i, &m, &c);
    char *type = drone_tasks[i].criticality ? "critical" : "soft    ";
    printf("  %-16s  %s  %d          %d\n", drone_tasks[i].name, type, c, m);
    total_misses += m;
    total_completions += c;
    if(drone_tasks[i].criticality){ crit_misses += m; crit_completions += c; }
    else                           { soft_misses += m; soft_completions += c; }
  }

  printf("  ----------------  --------  -----------  ------\n");
  printf("  Total                       %d          %d\n",
         total_completions, total_misses);
  printf("\n");
  printf("  [%s] Critical misses: %d  completions: %d\n",
         mode_name(mode), crit_misses, crit_completions);
  printf("  [%s] Soft     misses: %d  completions: %d\n",
         mode_name(mode), soft_misses, soft_completions);
  printf("============================================\n");

  for(int i = 0; i < NTASKS; i++){
    kill(pids[i]);
    wait(0);
  }

  exit(0);
}
