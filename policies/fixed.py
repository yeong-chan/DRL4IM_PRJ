"""Fixed-rule baselines a-e from Zhang & Si (2020) Table 3."""
from __future__ import annotations

from cbm.env import vec_to_action


class Constant:
    """Always take the same replacement vector (baselines a, b, c, d)."""

    def __init__(self, vec):
        self.action = vec_to_action(vec)

    def act(self, obs):
        return self.action


class ReplaceFailed:
    """Baseline e: replace every component that has reached its threshold."""

    def act(self, obs):
        K = len(obs) // 2
        return vec_to_action(obs[:K] == 0)


def paper_baselines(K: int) -> dict:
    """Baselines defined for this K (b and c only exist for K=2)."""
    out = {"a: do nothing": Constant([0] * K)}
    if K == 2:
        out["b: replace 1"] = Constant([1, 0])
        out["c: replace 2"] = Constant([0, 1])
    out["d: replace all"] = Constant([1] * K)
    out["e: replace failed"] = ReplaceFailed()
    return out
