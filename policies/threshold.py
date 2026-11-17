"""Threshold heuristics on the normalised degradation d_i / delta_i.

Both always replace failed components (otherwise the system would stay down).
"""
from __future__ import annotations

import numpy as np

from cbm.env import vec_to_action


def _split(obs):
    K = len(obs) // 2
    return obs[:K] == 0, obs[K:]  # failed flags, degradation ratios


class SingleThreshold:
    """Replace component i when d_i / delta_i >= theta_i (scalar theta = shared level)."""

    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=float)

    def act(self, obs):
        failed, ratio = _split(obs)
        return vec_to_action(failed | (ratio >= self.theta))


class Opportunistic:
    """Two-level rule: a visit is triggered when any component is failed or above
    theta_U; during a visit every component above theta_L is replaced as well."""

    def __init__(self, theta_L, theta_U):
        self.theta_L = np.asarray(theta_L, dtype=float)
        self.theta_U = np.asarray(theta_U, dtype=float)

    def act(self, obs):
        failed, ratio = _split(obs)
        if not (failed | (ratio >= self.theta_U)).any():
            return 0
        return vec_to_action(failed | (ratio >= self.theta_L))
