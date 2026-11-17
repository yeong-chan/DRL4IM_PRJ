"""Grid search for threshold heuristics, recording every objective at every grid point.

The same table then answers both multi-objective questions:
  best_weighted(): which thresholds are best for a given weight vector w
  pareto_front():  which thresholds are not dominated on the chosen objectives
"""
from __future__ import annotations

import ast
import itertools

import numpy as np
import pandas as pd

from cbm.config import Scenario
from cbm.metrics import evaluate
from .threshold import Opportunistic, SingleThreshold

GRID = [round(0.10 + 0.05 * i, 2) for i in range(19)]  # 0.10 .. 1.00
# Episode totals that correspond to env OBJECTIVES, in the same order.
OBJECTIVE_COLS = ["downtime", "tc_R", "n_system_failures", "n_replacements"]


def candidates(K: int, per_component: bool):
    """Yield (family, params, policy). Per-component grids are only feasible for small K."""
    if per_component:
        for theta in itertools.product(GRID, repeat=K):
            yield "single", theta, SingleThreshold(theta)
    else:
        for theta in GRID:
            yield "single", (theta,), SingleThreshold(theta)
    for lo, hi in itertools.product(GRID, GRID):
        if lo < hi:  # lo == hi is the shared single threshold
            yield "opportunistic", (lo, hi), Opportunistic(lo, hi)


def grid_search(scenario: Scenario, n_episodes: int = 100, seed: int = 0,
                per_component: bool | None = None) -> pd.DataFrame:
    """Mean of every metric for every candidate. Tuning seeds differ from evaluation seeds."""
    if per_component is None:
        per_component = scenario.K <= 2
    rows = []
    for family, params, policy in candidates(scenario.K, per_component):
        m = evaluate(scenario, policy, n_episodes=n_episodes, seed=seed).mean()
        rows.append({"family": family, "params": params, **m})
    return pd.DataFrame(rows)


def weighted_cost(df: pd.DataFrame, w) -> pd.Series:
    """Scalarised objective w . (downtime, replacement cost, failures, replacements)."""
    return df[OBJECTIVE_COLS] @ np.asarray(w, dtype=float)


def best_weighted(df: pd.DataFrame, w) -> pd.Series:
    return df.loc[weighted_cost(df, w).idxmin()]


def pareto_front(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Rows not dominated on `cols` (all minimised)."""
    x = df[cols].to_numpy()
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = (x <= x[i]).all(axis=1) & (x < x[i]).any(axis=1)
        keep[i] = not dominated.any()
    return df[keep].sort_values(cols[0])


def make_policy(row):
    if row["family"] == "single":
        return SingleThreshold(row["params"])
    return Opportunistic(*row["params"])


def load_grid(path) -> pd.DataFrame:
    """Read a saved grid_search table, restoring the threshold tuples."""
    return pd.read_csv(path, converters={"params": ast.literal_eval})
