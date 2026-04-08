#ifndef POLICY_H
#define POLICY_H

#define NN_IN        24
#define NN_HIDDEN    32
#define NN_OUT        7
#define Q10_SCALE  1024
#define NUM_RT_TASKS  6
#define IDLE_ACTION   6
#define MAX_DEADLINE 100

// Run Q10 fixed-point NN inference on a state vector.
// Returns action index 0-6 (0-5 = run task, 6 = idle).
int nn_infer(int state[NN_IN]);

#endif
