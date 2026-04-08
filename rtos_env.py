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

# Stressed taskset — U ≈ 1.15
STRESSED_TASKSET = [
    (10, 10, 3),
    (15, 15, 3),
    (20, 20, 4),
    (30, 30, 5),
    (50, 50, 8),
    (100, 100, 12),
]

# Hard tasksets — high enough U that variable exec still forces misses
# U_nom=1.57, U_avg~1.17 with variable exec
HARD_TASKSET = [
    (10, 10, 4),
    (15, 15, 5),
    (20, 20, 5),
    (30, 30, 7),
    (50, 50, 10),
    (100, 100, 15),
]

# U_nom=1.87, U_avg~1.40 with variable exec
VERY_HARD_TASKSET = [
    (10, 10, 5),
    (15, 15, 6),
    (20, 20, 7),
    (30, 30, 8),
    (50, 50, 12),
    (100, 100, 20),
]

# Mixed-criticality: T0-T2 are CRITICAL (high-freq sensors/control),
# T3-T5 are SOFT (logging, telemetry, housekeeping).
# Default criticality weights per task index.
DEFAULT_CRITICALITY = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # uniform (no mixed crit)
MIXED_CRITICALITY = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]    # T0-T2 critical, T3-T5 soft

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
        "release_tick",
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
        self.release_tick = 0
        self.ready = False
        self.last_scheduled = -1


class RTOSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        taskset=None,
        max_ticks=300,
        completion_reward: float = 2.0,
        miss_penalty: float = -3.0,
        tick_cost: float = -0.01,
        context_switch_penalty: float = -0.05,
        urgency_weight: float = 0.1,
        variable_exec: bool = False,
        starvation_weight: float = 0.0,
        jitter_weight: float = 0.0,
        criticality=None,
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
        self.max_wcet = max(w for _, _, w in self.taskset_cfg)

        self.completion_reward = completion_reward
        self.miss_penalty = miss_penalty
        self.tick_cost = tick_cost
        self.context_switch_penalty = context_switch_penalty
        self.urgency_weight = urgency_weight
        self.variable_exec = variable_exec
        self.starvation_weight = starvation_weight
        self.jitter_weight = jitter_weight
        self.criticality = criticality or DEFAULT_CRITICALITY

        self.tasks = []
        self.tick = 0
        self.last_action = IDLE_ACTION
        self.deadline_misses = 0
        self.completions = 0
        self._starvation_total = 0.0
        self._jitter_total = 0.0
        self.context_switch_count = 0
        self._critical_misses = 0
        self._soft_misses = 0

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
            idx: (n_ready - rank) / n_ready for rank, idx in enumerate(ready_indices)
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
                # Normalize by own period so fast tasks show starvation correctly
                obs[base + 1] = np.clip(since / t.period, 0.0, 1.0)
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
        self._starvation_total = 0.0
        self._jitter_total = 0.0
        self.context_switch_count = 0
        self._critical_misses = 0
        self._soft_misses = 0
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
                    bcet = max(1, t.wcet // 2)
                    t.remaining = int(self.np_random.integers(bcet, t.wcet + 1))
                else:
                    t.remaining = t.wcet
                t.abs_deadline = self.tick + t.deadline
                t.release_tick = self.tick
                t.ready = True
                t.next_release = self.tick + t.period

    def _check_deadlines(self) -> list:
        """Check for deadline misses. Must run BEFORE _do_releases.
        Returns list of task indices that missed."""
        missed_indices = []
        for i, t in enumerate(self.tasks):
            if t.ready and self.tick >= t.abs_deadline:
                missed_indices.append(i)
                t.ready = False
                t.remaining = 0
        return missed_indices

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

                if self.jitter_weight != 0.0:
                    response_time = self.tick - t.release_tick
                    reward += self.jitter_weight * (-response_time / t.deadline)
                    self._jitter_total += response_time / t.deadline

                if t.remaining == 0:
                    t.ready = False
                    completions = 1
                    reward += self.completion_reward * self.criticality[action]

        # Context switch penalty (task-to-task only)
        if (
            action != self.last_action
            and action != IDLE_ACTION
            and self.last_action != IDLE_ACTION
        ):
            reward += self.context_switch_penalty
            self.context_switch_count += 1
        self.last_action = action

        self.tick += 1

        # Starvation: sum of normalised wait time across all ready tasks.
        # Always computed unconditionally so eval metrics are valid for all schedulers.
        starvation = sum(
            (self.tick - t.last_scheduled if t.last_scheduled >= 0 else self.tick) / t.period
            for t in self.tasks if t.ready
        )
        self._starvation_total += starvation
        if self.starvation_weight != 0.0:
            reward += self.starvation_weight * (-starvation)

        # 1. Check deadlines (before releases — catches misses at period boundaries)
        missed_indices = self._check_deadlines()
        for idx in missed_indices:
            reward += self.miss_penalty * self.criticality[idx]
            if self.criticality[idx] > 1.0:
                self._critical_misses += 1
            else:
                self._soft_misses += 1

        # 2. Release new jobs
        self._do_releases()

        self.deadline_misses += len(missed_indices)
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
                "context_switches": self.context_switch_count,
                "starvation_total": self._starvation_total,
                "jitter_total": self._jitter_total,
                "critical_misses": self._critical_misses,
                "soft_misses": self._soft_misses,
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
        tmp_rng = np.random.default_rng(seed)
        taskset = self._sample_taskset(tmp_rng)

        self.taskset_cfg = taskset
        self.n_tasks = len(taskset)
        self.max_deadline = max(d for _, d, _ in taskset)
        self.max_period = max(p for p, _, _ in taskset)
        self.max_wcet = max(w for _, _, w in taskset)

        return super().reset(seed=seed, options=options)

    def _sample_taskset(self, rng) -> list:
        """Sample a random valid taskset with utilization in self.utilization_range."""
        target_u = float(rng.uniform(*self.utilization_range))

        for _ in range(200):
            periods = sorted(
                rng.choice(CANDIDATE_PERIODS, size=MAX_TASKS, replace=False)
            )
            shares = rng.uniform(0.05, 0.35, size=MAX_TASKS).astype(float)
            shares = shares / shares.sum() * target_u

            tasks = []
            valid = True
            for p, u in zip(periods, shares):
                w = max(1, round(float(p) * float(u)))
                if w >= p:
                    valid = False
                    break
                tasks.append((int(p), int(p), int(w)))

            if valid:
                return tasks

        return list(NORMAL_TASKSET)
