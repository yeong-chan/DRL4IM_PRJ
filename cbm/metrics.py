"""Shared evaluation: every policy (heuristic, tabular, deep) is scored by this code.

A policy is any object with `act(obs) -> int`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Scenario
from .env import CBMEnv


def run_episode(env: CBMEnv, policy, seed: int) -> dict:
    sc = env.sc
    obs, _ = env.reset(seed=seed)
    tot = np.zeros(4)
    comp_fail = 0
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        tot += info["cost_vec"]
        comp_fail += info["n_comp_failed"]
        done = term or trunc
    downtime, repl_cost, sys_fail, n_rep = tot
    tc_I = sc.c_I * sc.n_steps
    tc_D = sc.c_D * downtime
    return {
        "cost_rate": (tc_I + tc_D + repl_cost) / sc.H,
        "tc_I": tc_I, "tc_D": tc_D, "tc_R": repl_cost,
        "downtime": downtime,
        "n_system_failures": sys_fail,
        "n_component_failures": comp_fail,
        "n_replacements": n_rep,
    }


def evaluate(scenario: Scenario, policy, n_episodes: int = 1000, seed: int = 10_000) -> pd.DataFrame:
    """One row per episode. Episode i always uses seed + i, so policies share seeds."""
    env = CBMEnv(scenario)
    return pd.DataFrame([run_episode(env, policy, seed + i) for i in range(n_episodes)])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD of every metric."""
    return pd.DataFrame({"mean": df.mean(), "sd": df.std(ddof=1)})
