"""DQN and Double DQN for CBMEnv.

The two differ only in how the bootstrap action is chosen (see `_target`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from cbm.config import Scenario
from cbm.env import CBMEnv


@dataclass
class DQNConfig:
    double: bool = True
    hidden: int = 64            # paper: two hidden layers of 64 units
    gamma: float = 0.99         # paper
    # Everything below is our choice; the paper does not report it.
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    target_update: int = 500    # hard copy every N steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3   # share of training spent annealing epsilon
    total_steps: int = 200_000
    seed: int = 0


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size, obs_dim, rng, r_dim=1):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size if r_dim == 1 else (size, r_dim), np.float32)
        self.size, self.n, self.i, self.rng = size, 0, 0, rng

    def add(self, s, a, r, s2):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i] = s, a, r, s2
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, batch)
        return (torch.from_numpy(x[idx]) for x in (self.s, self.a, self.r, self.s2))


class DQNAgent:
    """Greedy policy over a trained Q-network; `act(obs) -> int` like every other policy."""

    r_dim = 1  # size of the reward stored in the replay buffer

    def __init__(self, scenario: Scenario, cfg: DQNConfig):
        self.sc, self.cfg = scenario, cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.n_actions = 2 ** scenario.K
        self.q = self._make_net()
        self.q_target = self._make_net()
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        # Rewards are divided by the cost of one fully-down interval to keep targets O(1).
        self.reward_scale = scenario.c_D * scenario.tau

    def _make_net(self):
        return QNet(2 * self.sc.K, self.n_actions, self.cfg.hidden)

    def _stored_reward(self, r, info):
        return r / self.reward_scale

    @torch.no_grad()
    def act(self, obs):
        return int(self.q(torch.as_tensor(obs).unsqueeze(0)).argmax(1))

    @torch.no_grad()
    def _target(self, r, s2):
        q_next = self.q_target(s2)
        if self.cfg.double:   # DDQN: online net picks the action, target net scores it
            a_star = self.q(s2).argmax(1, keepdim=True)
            v = q_next.gather(1, a_star).squeeze(1)
        else:                 # DQN: target net does both
            v = q_next.max(1).values
        # No terminal states: the horizon is a truncation, so we always bootstrap.
        return r + self.cfg.gamma * v

    def _update(self, buf):
        s, a, r, s2 = buf.sample(self.cfg.batch_size)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_sa, self._target(r, s2))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), q_sa.mean().item()

    def train(self, log_every: int = 10_000, verbose: bool = True):
        """Returns a list of log rows (one per finished episode)."""
        cfg, sc = self.cfg, self.sc
        env = CBMEnv(sc)
        buf = ReplayBuffer(cfg.buffer_size, 2 * sc.K, self.rng, self.r_dim)
        obs, _ = env.reset(seed=cfg.seed)
        log, ep_cost, ep, q_mean = [], np.zeros(4), 0, 0.0
        anneal = cfg.eps_fraction * cfg.total_steps
        for t in range(1, cfg.total_steps + 1):
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * t / anneal)
            a = int(self.rng.integers(self.n_actions)) if self.rng.random() < eps else self.act(obs)
            obs2, r, _, trunc, info = env.step(a)
            buf.add(obs, a, self._stored_reward(r, info), obs2)
            obs = obs2
            ep_cost += info["cost_vec"]

            if t >= cfg.learning_starts:
                _, q_mean = self._update(buf)
            if t % cfg.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            if trunc:
                cost_rate = (sc.c_I * sc.n_steps + sc.c_D * ep_cost[0] + ep_cost[1]) / sc.H
                log.append({"step": t, "episode": ep, "eps": eps, "train_cost_rate": cost_rate,
                            "q_mean": q_mean * self.reward_scale})
                ep, ep_cost = ep + 1, np.zeros(4)
                obs, _ = env.reset()
            if verbose and t % log_every == 0:
                recent = np.mean([row["train_cost_rate"] for row in log[-20:]])
                print(f"step {t:>7}  eps {eps:.2f}  train cost rate (last 20 ep) {recent:.2f}", flush=True)
        return log

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.q_target.load_state_dict(self.q.state_dict())
