"""Scenario definitions. Paper scenarios follow Zhang & Si (2020) Tables 1, 2, 6, 9, 10."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .degradation import CompoundPoisson, GammaProcess


@dataclass(frozen=True)
class Scenario:
    name: str
    process: object            # has .K and .sample_path(rng, tau)
    delta: tuple               # failure thresholds
    c_R: tuple                 # replacement cost per component
    c_I: float = 12.0          # cost per inspection
    c_D: float = 15.0          # downtime cost per unit time
    tau: float = 10.0          # inspection interval
    H: float = 1000.0          # horizon
    joint_discount: float = 0.0  # saved per extra component replaced at the same inspection
    # If True, failed components are always replaced at inspection (the agent cannot
    # leave the system down). The paper's setting is False.
    forced_repair: bool = False

    @property
    def K(self):
        return len(self.delta)

    @property
    def n_steps(self):
        return int(round(self.H / self.tau))

    def with_(self, **kw):
        return replace(self, **kw)


# --- Paper, K=2, compound Poisson (Tables 1-2) ---
_CPP = dict(lam=(0.1, 0.2), mu=(2.0, 2.5), sigma=(0.2, 0.3))
_K2 = dict(delta=(15.0, 10.0), c_R=(20.0, 50.0), c_I=12.0, c_D=15.0, tau=10.0)

CPP_A = Scenario("cpp_A", CompoundPoisson(**_CPP, shared=None), **_K2)
CPP_B = Scenario("cpp_B", CompoundPoisson(**_CPP, shared=(0.04, 5.0, 0.4)), **_K2)

# --- Paper, K=2, gamma process (Table 6). tau is not stated; 10 is implied by the baselines. ---
_GP2 = GammaProcess(alpha=(0.4, 0.6), beta=(1.0, 1.2), shared=(0.3, 1.6))
GP_K2 = Scenario("gp_K2", _GP2, **_K2)
GP_K2_JOINT = Scenario("gp_K2_joint", _GP2, **_K2, joint_discount=10.0)  # both for $60


# --- Paper, K=12, gamma process (Tables 9-10). u_i, v_i drawn once with a fixed seed. ---
def _gp_k12():
    rng = np.random.default_rng(0)
    alpha = tuple(0.6 + rng.uniform(0, 0.05, 12))
    beta = tuple(1.6 + rng.uniform(0, 0.1, 12))
    return Scenario(
        "gp_K12", GammaProcess(alpha, beta, shared=(0.3, 1.2)),
        delta=(15.0,) * 12, c_R=(20.0,) * 12, c_I=12.0, c_D=20.0, tau=10.0,
    )


GP_K12 = _gp_k12()

# --- Our own mid-size scenario for the multi-objective experiments (not from the paper). ---
GP_K4 = Scenario(
    "gp_K4",
    GammaProcess(alpha=(0.4, 0.6, 0.5, 0.7), beta=(1.0, 1.2, 1.0, 1.4), shared=(0.3, 1.6)),
    delta=(15.0, 10.0, 12.0, 12.0), c_R=(20.0, 50.0, 30.0, 40.0),
    c_I=12.0, c_D=15.0, tau=10.0,
)

# Same system with mandatory repair: needed once failures are penalised, otherwise
# "fail once and stay down" minimises the failure count.
GP_K4_FR = GP_K4.with_(name="gp_K4_fr", forced_repair=True)

SCENARIOS = {s.name: s for s in (CPP_A, CPP_B, GP_K2, GP_K2_JOINT, GP_K12, GP_K4, GP_K4_FR)}
