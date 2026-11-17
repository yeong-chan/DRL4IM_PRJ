"""Multi-critic (D)DQN: one Q-head per objective, actions chosen by the weighted sum.

Critic m learns Q_m(s, a) = expected discounted sum of objective m alone
(downtime, replacement cost, system failures, replacements) under the policy
that is greedy for sum_m w_m Q_m. With w = (c_D, 1, 0, 0) the weighted sum is the
paper's cost, so the agent must behave like plain DDQN.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cbm.config import Scenario
from cbm.env import OBJECTIVES
from .dqn import DQNAgent, DQNConfig


class MultiCriticNet(nn.Module):
    """Shared trunk, then one block of action values per critic: output (batch, M, A)."""

    def __init__(self, obs_dim, n_actions, hidden, n_critics):
        super().__init__()
        self.shape = (n_critics, n_actions)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.Linear(hidden, n_critics * n_actions)

    def forward(self, x):
        return self.heads(self.trunk(x)).view(-1, *self.shape)


class MultiCriticAgent(DQNAgent):
    r_dim = len(OBJECTIVES)

    def __init__(self, scenario: Scenario, cfg: DQNConfig, w=None):
        # Weights are in natural units: cost per unit of each objective.
        self.w = np.array(w if w is not None else (scenario.c_D, 1.0, 0.0, 0.0), dtype=float)
        # Per-step maximum of each objective, so every critic sees rewards in [-1, 0].
        self.scales = np.array([scenario.tau, sum(scenario.c_R), 1.0, scenario.K], dtype=float)
        self.w_scaled = torch.tensor(self.w * self.scales, dtype=torch.float32).view(1, -1, 1)
        super().__init__(scenario, cfg)
        self.reward_scale = 1.0  # the weighted sum of critics is already in cost units

    def _make_net(self):
        return MultiCriticNet(2 * self.sc.K, self.n_actions, self.cfg.hidden, self.r_dim)

    def _stored_reward(self, r, info):
        return -info["cost_vec"] / self.scales

    def _scalarize(self, q):
        """(batch, M, A) -> (batch, A): the value the policy is greedy for."""
        return (q * self.w_scaled).sum(1)

    def _take(self, q, a):
        """Pick action a from every critic: (batch, M, A) -> (batch, M)."""
        return q.gather(2, a.view(-1, 1, 1).expand(-1, self.r_dim, 1)).squeeze(2)

    @torch.no_grad()
    def act(self, obs):
        return int(self._scalarize(self.q(torch.as_tensor(obs).unsqueeze(0))).argmax(1))

    @torch.no_grad()
    def _target(self, r, s2):
        q_next = self.q_target(s2)
        # One shared greedy action for all critics, chosen on the weighted sum.
        chooser = self.q(s2) if self.cfg.double else q_next
        a_star = self._scalarize(chooser).argmax(1)
        return r + self.cfg.gamma * self._take(q_next, a_star)

    def _update(self, buf):
        s, a, r, s2 = buf.sample(self.cfg.batch_size)
        q_sa = self._take(self.q(s), a)
        loss = nn.functional.smooth_l1_loss(q_sa, self._target(r, s2))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), (q_sa * self.w_scaled.squeeze(2)).sum(1).mean().item()

    @torch.no_grad()
    def critic_values(self, obs) -> dict:
        """Per-objective Q-values in natural units (negative expected discounted totals)."""
        q = self.q(torch.as_tensor(obs).unsqueeze(0))[0].numpy() * self.scales[:, None]
        return dict(zip(OBJECTIVES, q))
