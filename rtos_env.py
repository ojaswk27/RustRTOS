"""
RTOS Gymnasium Environment — simulates a preemptive tick-based RTOS.

The agent selects which task to run each tick. Tasks are periodic with
deadlines. The environment rewards completing tasks on time and penalizes
deadline misses, encouraging the agent to learn an optimal scheduling policy.

State: 6 tasks x 4 features (time-to-deadline, time-since-scheduled,
       remaining execution, is_ready) = 24 floats in [0,1].
Action: 0..5 = run task i, 6 = idle.
"""

import gymnasium as gym
import numpy as np
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

# Stressed taskset — U ≈ 1.15, tighter deadlines on some tasks
STRESSED_TASKSET = [
    (10, 10, 3),
    (15, 15, 3),
    (20, 20, 4),
    (30, 30, 5),
    (50, 50, 8),
    (100, 100, 12),
]

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
        "completed_this_period",
    )

    def __init__(self, period: int, deadline: int, wcet: int):
        self.period = period
        self.deadline = deadline
        self.wcet = wcet
        self.remaining = 0
        self.next_release = 0
        self.abs_deadline = deadline
        self.ready = False
        self.last_scheduled = -1
        self.completed_this_period = False


class RTOSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, taskset=None, max_ticks=300):
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

        self.tasks = []
        self.tick = 0
        self.last_action = IDLE_ACTION
        self.deadline_misses = 0
        self.completions = 0

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(MAX_TASKS * FEATURES_PER_TASK, dtype=np.float32)
        for i, t in enumerate(self.tasks):
            base = i * FEATURES_PER_TASK
            if t.ready:
                # Time remaining until deadline, normalized and clamped
                obs[base] = np.clip(
                    (t.abs_deadline - self.tick) / self.max_deadline, 0.0, 1.0
                )
                # Time since last scheduled, normalized
                since = (
                    (self.tick - t.last_scheduled)
                    if t.last_scheduled >= 0
                    else self.max_period
                )
                obs[base + 1] = np.clip(since / self.max_period, 0.0, 1.0)
                # Remaining execution normalized by wcet
                obs[base + 2] = t.remaining / t.wcet if t.wcet > 0 else 0.0
                obs[base + 3] = 1.0
            # else: all zeros (not ready)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        self.last_action = IDLE_ACTION
        self.deadline_misses = 0
        self.completions = 0
        self.tasks = [TaskSim(p, d, w) for p, d, w in self.taskset_cfg]
        # Release all tasks at tick 0
        for t in self.tasks:
            t.next_release = 0
        self._do_releases()
        return self._build_obs(), {}

    def _do_releases(self):
        """Release tasks whose period boundary has arrived."""
        for t in self.tasks:
            if self.tick >= t.next_release:
                t.remaining = t.wcet
                t.abs_deadline = self.tick + t.deadline
                t.ready = True
                t.completed_this_period = False
                t.next_release = self.tick + t.period

    def _check_deadlines(self) -> int:
        """Check and count deadline misses."""
        misses = 0
        for t in self.tasks:
            if t.ready and not t.completed_this_period and self.tick >= t.abs_deadline:
                misses += 1
                # Abandon this job — task will re-release next period
                t.ready = False
                t.remaining = 0
        return misses

    def step(self, action: int):
        reward = -0.01  # small per-tick cost encourages urgency

        # Execute action
        completions = 0
        if action != IDLE_ACTION and action < self.n_tasks:
            t = self.tasks[action]
            if t.ready and t.remaining > 0:
                t.remaining -= 1
                t.last_scheduled = self.tick
                if t.remaining == 0:
                    t.ready = False
                    t.completed_this_period = True
                    completions = 1
                    reward += 1.0

        # Context switch penalty
        if (
            action != self.last_action
            and action != IDLE_ACTION
            and self.last_action != IDLE_ACTION
        ):
            reward -= 0.05
        self.last_action = action

        self.tick += 1

        # Release new jobs and check deadline misses
        self._do_releases()
        misses = self._check_deadlines()
        reward -= 2.0 * misses

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
