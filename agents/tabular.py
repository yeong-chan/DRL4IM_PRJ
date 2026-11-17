"""Tabular control on a discretised state: Monte Carlo, SARSA and Q-learning.

Each component's ratio d_i / delta_i is cut into `n_bins` equal bins on [0, 1),
plus one extra level for "failed", giving (n_bins + 1)^K states.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cbm.config import Scenario
from cbm.env import CBMEnv

ALGOS = ("mc", "sarsa", "qlearning")


@dataclass
class TabularConfig:
    algo: str = "qlearning"
    n_bins: int = 10
    gamma: float = 0.99
    alpha: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    episodes: int = 5_000
    # MC only: extra steps simulated after the horizon so that returns of the
    # updated steps are not cut short (this task has no terminal state).
    mc_tail: int = 200
    seed: int = 0


class TabularAgent:
    def __init__(self, scenario: Scenario, cfg: TabularConfig):
        assert cfg.algo in ALGOS
        self.sc, self.cfg, self.K = scenario, cfg, scenario.K
        self.rng = np.random.default_rng(cfg.seed)
        self.n_actions = 2 ** self.K
        self.levels = cfg.n_bins + 1
        self.Q = np.zeros((self.levels ** self.K, self.n_actions))
        self.reward_scale = scenario.c_D * scenario.tau

    def state(self, obs) -> int:
        ok, ratio = obs[:self.K] == 1, obs[self.K:]
        level = np.where(ok, np.minimum((ratio * self.cfg.n_bins).astype(int), self.cfg.n_bins - 1),
                         self.cfg.n_bins)
        return int(level @ (self.levels ** np.arange(self.K)))

    def act(self, obs):
        return int(self.Q[self.state(obs)].argmax())

    def _eps_greedy(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(self.Q[s].argmax())

    def train(self):
        cfg, sc = self.cfg, self.sc
        mc = cfg.algo == "mc"
        n_updates = sc.n_steps
        env = CBMEnv(sc.with_(H=sc.H + cfg.mc_tail * sc.tau) if mc else sc)
        log = []
        for ep in range(cfg.episodes):
            eps = max(cfg.eps_end,
                      cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / (cfg.eps_fraction * cfg.episodes))
            obs, _ = env.reset(seed=cfg.seed + ep)
            s = self.state(obs)
            a = self._eps_greedy(s, eps)
            traj, cost, done, t = [], 0.0, False, 0
            while not done:
                obs, r, _, done, info = env.step(a)
                if t < n_updates:
                    cost += -r
                r /= self.reward_scale
                s2 = self.state(obs)
                a2 = self._eps_greedy(s2, eps)
                if mc:
                    traj.append((s, a, r))
                else:
                    # SARSA bootstraps on the action it will actually take (on-policy);
                    # Q-learning on the greedy action (off-policy).
                    nxt = self.Q[s2, a2] if cfg.algo == "sarsa" else self.Q[s2].max()
                    self.Q[s, a] += cfg.alpha * (r + cfg.gamma * nxt - self.Q[s, a])
                s, a, t = s2, a2, t + 1
            if mc:  # every-visit, constant-alpha MC control
                G = 0.0
                for i in range(len(traj) - 1, -1, -1):
                    s_i, a_i, r_i = traj[i]
                    G = r_i + cfg.gamma * G
                    if i < n_updates:
                        self.Q[s_i, a_i] += cfg.alpha * (G - self.Q[s_i, a_i])
            log.append({"episode": ep, "eps": eps,
                        "train_cost_rate": (sc.c_I * sc.n_steps + cost) / sc.H})
        return log
