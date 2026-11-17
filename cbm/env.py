"""Gymnasium environment for periodic-inspection CBM of a K-component series system."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .config import Scenario
from .degradation import first_passage

# Order of the per-step objective vector in info["cost_vec"] (all "lower is better").
OBJECTIVES = ("downtime", "replacement_cost", "system_failure", "n_replacements")


def action_to_vec(action: int, K: int) -> np.ndarray:
    """Integer action -> binary replacement vector (bit k = replace component k)."""
    return (int(action) >> np.arange(K)) & 1


def vec_to_action(vec) -> int:
    return int(sum(int(b) << k for k, b in enumerate(vec)))


class CBMEnv(gym.Env):
    """One step = decide replacements at an inspection, then degrade for tau.

    obs    = [o_1..o_K, d_1/delta_1..d_K/delta_K], o_i = 1 if component i has not failed
    action = integer in [0, 2^K), see action_to_vec
    reward = -(c_D * downtime + replacement cost)   (paper Eq. 6; inspection cost excluded)
    info["cost_vec"] = per-objective costs for this step, ordered as OBJECTIVES

    A failed system stays down until the failed component is replaced. By default
    nothing forces that replacement (as in the paper); scenario.forced_repair=True
    adds failed components to every action.
    Reaching the horizon truncates the episode; it never terminates.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario):
        self.sc = scenario
        self.K = scenario.K
        self.delta = np.array(scenario.delta, dtype=float)
        self.c_R = np.array(scenario.c_R, dtype=float)
        self.action_space = gym.spaces.Discrete(2 ** self.K)
        self.observation_space = gym.spaces.Box(0.0, np.inf, (2 * self.K,), np.float32)
        self.d = np.zeros(self.K)
        self.j = 0

    def _obs(self):
        o = (self.d < self.delta).astype(np.float32)
        return np.concatenate([o, (self.d / self.delta).astype(np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.d = np.zeros(self.K)
        self.j = 0
        return self._obs(), {}

    def step(self, action):
        sc = self.sc
        a = action_to_vec(action, self.K)
        if sc.forced_repair:
            a = a | (self.d >= self.delta)
        self.d = np.where(a == 1, 0.0, self.d)  # perfect, instantaneous replacement
        n_rep = int(a.sum())
        repl_cost = float(a @ self.c_R) - sc.joint_discount * max(n_rep - 1, 0)

        n_comp_failed = 0
        if (self.d >= self.delta).any():  # failed component left in place: still down
            downtime, failure = sc.tau, 0
        else:
            times, incs = sc.process.sample_path(self.np_random, sc.tau)
            self.d, t_fail = first_passage(self.d, times, incs, self.delta)
            failure = int(t_fail is not None)
            downtime = sc.tau - t_fail if failure else 0.0
            n_comp_failed = int((self.d >= self.delta).sum())

        self.j += 1
        reward = -(sc.c_D * downtime + repl_cost)
        info = {
            "cost_vec": np.array([downtime, repl_cost, failure, n_rep], dtype=float),
            "n_comp_failed": n_comp_failed,
        }
        return self._obs(), reward, False, self.j >= sc.n_steps, info
