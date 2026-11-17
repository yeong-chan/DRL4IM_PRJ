"""Degradation processes for d_i(t) = X_i(t) + X_s(t) (Zhang & Si 2020, Sec. 2.1).

A process exposes `sample_path(rng, tau) -> (times, incs)`:
  times: (n,) increasing event times in (0, tau]
  incs:  (n, K) degradation added to each component at each event
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _truncated_normal(rng, mu, sigma):
    """Normal draws truncated at zero (rejection sampling)."""
    x = rng.normal(mu, sigma)
    bad = x < 0
    while bad.any():
        x[bad] = rng.normal(mu[bad], sigma[bad])
        bad = x < 0
    return x


@dataclass(frozen=True)
class CompoundPoisson:
    """Shocks arrive by HPP(lam); sizes ~ N(mu, sigma^2) truncated at 0. Simulated exactly."""

    lam: tuple
    mu: tuple
    sigma: tuple
    shared: tuple | None = None  # (lam_s, mu_s, sigma_s) or None

    @property
    def K(self):
        return len(self.lam)

    def sample_path(self, rng, tau):
        K = self.K
        lam, mu, sigma = (list(self.lam), list(self.mu), list(self.sigma))
        if self.shared is not None:
            lam.append(self.shared[0]); mu.append(self.shared[1]); sigma.append(self.shared[2])
        lam, mu, sigma = np.array(lam), np.array(mu), np.array(sigma)

        counts = rng.poisson(lam * tau)
        n = int(counts.sum())
        if n == 0:
            return np.empty(0), np.empty((0, K))
        src = np.repeat(np.arange(len(lam)), counts)  # index K = shared process
        times = rng.uniform(0.0, tau, n)
        sizes = _truncated_normal(rng, mu[src], sigma[src])

        incs = np.zeros((n, K))
        own = src < K
        incs[own, src[own]] = sizes[own]
        incs[~own, :] = sizes[~own, None]  # shared shock hits every component
        order = np.argsort(times)
        return times[order], incs[order]


@dataclass(frozen=True)
class GammaProcess:
    """Increment over dt ~ Gamma(shape=alpha*dt, rate=beta). Discretised with step eta."""

    alpha: tuple
    beta: tuple
    shared: tuple | None = None  # (alpha_s, beta_s) or None
    eta: float = 0.1

    @property
    def K(self):
        return len(self.alpha)

    def sample_path(self, rng, tau):
        n = int(round(tau / self.eta))
        dt = tau / n
        alpha, beta = np.array(self.alpha), np.array(self.beta)
        incs = rng.gamma(alpha * dt, 1.0 / beta, size=(n, self.K))
        if self.shared is not None:
            a_s, b_s = self.shared
            incs = incs + rng.gamma(a_s * dt, 1.0 / b_s, size=(n, 1))
        times = dt * np.arange(1, n + 1)
        return times, incs


def first_passage(d0, times, incs, delta):
    """Advance degradation along a path until any component reaches its threshold.

    Returns (d_new, t_fail). t_fail is None if the system survives the interval;
    otherwise degradation is frozen at the failure time (the system stops).
    """
    if len(times) == 0:
        return d0, None
    path = d0 + np.cumsum(incs, axis=0)
    hit = (path >= delta).any(axis=1)
    if not hit.any():
        return path[-1], None
    i = int(np.argmax(hit))
    return path[i], float(times[i])
