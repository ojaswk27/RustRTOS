"""
RTOS Gymnasium Environment — simulates a preemptive tick-based RTOS.

The agent selects which task to run each tick. Tasks are periodic with
implicit deadlines (deadline == period). The environment rewards completing
tasks on time and penalizes deadline misses.

State: 6 tasks x 4 features (time-to-deadline, time-since-scheduled,
       remaining-work, urgency-rank) = 24 floats in [0,1].
Action: 0..5 = run task i, 6 = idle.

Variants:
  RTOSEnv        — fixed taskset, deterministic execution times
  RandomRTOSEnv  — fresh random taskset every episode, optional variable exec
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Default taskset (period, deadline, wcet) — U ≈ 1.03
NORMAL_TASKSET = [
    (10, 10, 2),
    (15, 15, 3),
    (20, 20, 4),
    (30, 30, 5),
    (50, 50, 8),
    (100, 100, 10),
]

# Stressed taskset — U ≈ 1.15
STRESSED_TASKSET = [
    (10, 10, 3),
    (15, 15, 3),
    (20, 20, 4),
    (30, 30, 5),
    (50, 50, 8),
    (100, 100, 12),
]

# Periods used when randomly generating tasksets
CANDIDATE_PERIODS = [5, 10, 15, 20, 25, 30, 50, 100]

MAX_TASKS = 6
FEATURES_PER_TASK = 4
IDLE_ACTION = MAX_TASKS


class TaskSim:
    """Internal task state for simulation."""

    __slots__ = (
        "period",
        "deadline",
        "wcet",
        "remaining",
        "next_release",
        "abs_deadline",
        "ready",
        "last_scheduled",
    )

    def __init__(self, period: int, deadline: int, wcet: int):
        self.period = period
        self.deadline = deadline
        self.wcet = wcet
        self.remaining = 0
        self.next_release = 0
        self.abs_deadline = 0
        self.ready = False
        self.last_scheduled = -1


class RTOSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        taskset=None,
        max_ticks=300,
        completion_reward: float = 1.0,
        miss_penalty: float = -2.0,
        tick_cost: float = -0.01,
        context_switch_penalty: float = -0.05,
        urgency_weight: float = 0.0,
        variable_exec: bool = False,
    ):
        super().__init__()
        self.taskset_cfg = taskset or NORMAL_TASKSET
        assert len(self.taskset_cfg) <= MAX_TASKS
        self.n_tasks = len(self.taskset_cfg)
        self.max_ticks = max_ticks

        # Observation: 24 floats. Action: 7 discrete choices.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(MAX_TASKS * FEATURES_PER_TASK,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_TASKS + 1)

        # Normalization constants derived from taskset
        self.max_deadline = max(d for _, d, _ in self.taskset_cfg)
        self.max_period = max(p for p, _, _ in self.taskset_cfg)

        self.completion_reward = completion_reward
        self.miss_penalty = miss_penalty
        self.tick_cost = tick_cost
        self.context_switch_penalty = context_switch_penalty
        self.urgency_weight = urgency_weight
        self.variable_exec = variable_exec

        self.tasks = []
        self.tick = 0
        self.last_action = IDLE_ACTION
        self.deadline_misses = 0
        self.completions = 0

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(MAX_TASKS * FEATURES_PER_TASK, dtype=np.float32)

        # Rank ready tasks by urgency (nearest deadline = highest rank)
        ready_indices = sorted(
            [i for i, t in enumerate(self.tasks) if t.ready],
            key=lambda i: self.tasks[i].abs_deadline,
        )
        n_ready = len(ready_indices)
        # Most urgent → 1.0, least urgent → 1/n_ready, not ready → 0.0
        urgency_rank = {
            idx: (n_ready - rank) / n_ready
            for rank, idx in enumerate(ready_indices)
        }

        for i, t in enumerate(self.tasks):
            base = i * FEATURES_PER_TASK
            if t.ready:
                obs[base] = np.clip(
                    (t.abs_deadline - self.tick) / self.max_deadline, 0.0, 1.0
                )
                since = (
                    (self.tick - t.last_scheduled)
                    if t.last_scheduled >= 0
                    else self.max_period
                )
                obs[base + 1] = np.clip(since / self.max_period, 0.0, 1.0)
                obs[base + 2] = t.remaining / t.wcet if t.wcet > 0 else 0.0
                obs[base + 3] = urgency_rank[i]
            # else: all zeros (not ready)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        self.last_action = IDLE_ACTION
        self.deadline_misses = 0
        self.completions = 0
        self.tasks = [TaskSim(p, d, w) for p, d, w in self.taskset_cfg]
        for t in self.tasks:
            t.next_release = 0
        self._do_releases()
        return self._build_obs(), {}

    def _do_releases(self):
        """Release tasks whose period boundary has arrived. Runs AFTER _check_deadlines."""
        for t in self.tasks:
            if self.tick >= t.next_release:
                if self.variable_exec:
                    # Sample actual execution time from [bcet, wcet]
                    bcet = max(1, t.wcet // 2)
                    t.remaining = int(self.np_random.integers(bcet, t.wcet + 1))
                else:
                    t.remaining = t.wcet
                t.abs_deadline = self.tick + t.deadline
                t.ready = True
                t.next_release = self.tick + t.period

    def _check_deadlines(self) -> int:
        """Check for deadline misses. Must run BEFORE _do_releases."""
        misses = 0
        for t in self.tasks:
            if t.ready and self.tick >= t.abs_deadline:
                misses += 1
                t.ready = False
                t.remaining = 0
        return misses

    def step(self, action: int):
        reward = self.tick_cost

        # Execute action
        completions = 0
        if action != IDLE_ACTION and action < self.n_tasks:
            t = self.tasks[action]
            if t.ready and t.remaining > 0:
                t.remaining -= 1
                t.last_scheduled = self.tick
                if self.urgency_weight != 0.0 and t.deadline > 0:
                    # Bonus scales from 0 (just released) to 1 (at the deadline)
                    urgency = 1.0 - (t.abs_deadline - self.tick) / t.deadline
                    reward += self.urgency_weight * max(0.0, urgency)
                if t.remaining == 0:
                    t.ready = False
                    completions = 1
                    reward += self.completion_reward

        # Context switch penalty (task-to-task only)
        if (
            action != self.last_action
            and action != IDLE_ACTION
            and self.last_action != IDLE_ACTION
        ):
            reward += self.context_switch_penalty
        self.last_action = action

        self.tick += 1

        # 1. Check deadlines (before releases — catches misses at period boundaries)
        misses = self._check_deadlines()
        reward += self.miss_penalty * misses

        # 2. Release new jobs
        self._do_releases()

        self.deadline_misses += misses
        self.completions += completions

        obs = self._build_obs()
        terminated = self.tick >= self.max_ticks
        return (
            obs,
            reward,
            terminated,
            False,
            {
                "misses": self.deadline_misses,
                "completions": self.completions,
            },
        )


class RandomRTOSEnv(RTOSEnv):
    """RTOSEnv that samples a fresh random taskset on every reset.

    Forces the agent to learn a general scheduling policy rather than
    memorizing a fixed taskset. The utilization_range parameter controls
    the difficulty; use it for curriculum learning.
    """

    def __init__(self, utilization_range=(0.7, 1.1), **kwargs):
        # Start with NORMAL_TASKSET; replaced on first reset
        super().__init__(taskset=NORMAL_TASKSET, **kwargs)
        self.utilization_range = utilization_range

    def reset(self, seed=None, options=None):
        # Sample taskset using a temporary RNG (np_random not yet initialized)
        tmp_rng = np.random.default_rng(seed)
        taskset = self._sample_taskset(tmp_rng)

        # Update taskset-dependent attributes before calling super().reset()
        self.taskset_cfg = taskset
        self.n_tasks = len(taskset)
        self.max_deadline = max(d for _, d, _ in taskset)
        self.max_period = max(p for p, _, _ in taskset)

        return super().reset(seed=seed, options=options)

    def _sample_taskset(self, rng) -> list:
        """Sample a random valid taskset with utilization in self.utilization_range."""
        target_u = float(rng.uniform(*self.utilization_range))

        for _ in range(200):  # retry until we get a valid taskset
            periods = sorted(
                rng.choice(CANDIDATE_PERIODS, size=MAX_TASKS, replace=False)
            )
            # Sample per-task utilization shares and scale to target
            shares = rng.uniform(0.05, 0.35, size=MAX_TASKS).astype(float)
            shares = shares / shares.sum() * target_u

            tasks = []
            valid = True
            for p, u in zip(periods, shares):
                w = max(1, round(float(p) * float(u)))
                if w >= p:  # wcet must be strictly less than period
                    valid = False
                    break
                tasks.append((int(p), int(p), int(w)))

            if valid:
                return tasks

        return list(NORMAL_TASKSET)  # fallback — should rarely trigger
