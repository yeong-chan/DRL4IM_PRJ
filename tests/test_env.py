import numpy as np
import pytest

from cbm.config import CPP_B, GP_K2, Scenario
from cbm.degradation import CompoundPoisson, GammaProcess, first_passage
from cbm.env import CBMEnv, action_to_vec, vec_to_action
from cbm.metrics import evaluate
from policies.fixed import Constant, ReplaceFailed


class Scripted:
    """Deterministic process: the same shocks every interval."""

    K = 2

    def __init__(self, times, incs):
        self.times, self.incs = np.array(times, float), np.array(incs, float)

    def sample_path(self, rng, tau):
        return self.times, self.incs


def make_env(times, incs, **kw):
    sc = Scenario("t", Scripted(times, incs), delta=(10.0, 10.0), c_R=(20.0, 50.0),
                  c_D=15.0, tau=10.0, H=50.0, **kw)
    env = CBMEnv(sc)
    env.reset(seed=0)
    return env


def test_action_encoding_roundtrip():
    for a in range(8):
        assert vec_to_action(action_to_vec(a, 3)) == a
    assert list(action_to_vec(2, 2)) == [0, 1]


def test_no_failure_accumulates_and_replacement_resets():
    env = make_env([2, 5], [[1, 2], [1, 2]])
    obs, r, term, trunc, info = env.step(0)
    assert np.allclose(env.d, [2, 4]) and r == 0 and not term and not trunc
    assert np.allclose(obs, [1, 1, 0.2, 0.4])
    env.step(vec_to_action([0, 1]))  # replace component 2, then it degrades again
    assert np.allclose(env.d, [4, 4])


def test_failure_freezes_degradation_and_counts_downtime():
    # component 1 crosses 10 at t=5; the shock at t=8 must not be applied
    env = make_env([2, 5, 8], [[6, 1], [6, 1], [6, 1]])
    obs, r, *_, info = env.step(0)
    assert np.allclose(env.d, [12, 2])
    assert list(info["cost_vec"]) == [5.0, 0.0, 1.0, 0.0]
    assert r == -15.0 * 5 and info["n_comp_failed"] == 1
    assert list(obs[:2]) == [0, 1]


def test_unrepaired_system_stays_down_without_new_failure():
    env = make_env([2, 5, 8], [[6, 1], [6, 1], [6, 1]])
    env.step(0)
    _, r, *_, info = env.step(vec_to_action([0, 1]))  # replaces the wrong component
    assert list(info["cost_vec"]) == [10.0, 50.0, 0.0, 1.0]
    assert r == -(15.0 * 10 + 50.0)
    assert env.d[0] == 12 and env.d[1] == 0


def test_replacing_failed_component_restarts_system():
    env = make_env([2], [[6, 1]])
    env.step(0); env.step(0)  # d1 = 12 at t=2 of the second interval
    assert env.d[0] == 12
    obs, *_ = env.step(ReplaceFailed().act(env._obs()))
    assert np.allclose(env.d, [6, 3]) and list(obs[:2]) == [1, 1]


def test_joint_discount():
    env = make_env([], np.empty((0, 2)), joint_discount=10.0)
    assert env.step(3)[1] == -60.0
    assert env.step(1)[1] == -20.0


def test_reward_matches_cost_vector():
    env = CBMEnv(CPP_B)
    env.reset(seed=1)
    for a in [0, 1, 2, 3, 0, 0, 0]:
        _, r, *_, info = env.step(a)
        downtime, repl, *_ = info["cost_vec"]
        assert r == pytest.approx(-(CPP_B.c_D * downtime + repl))


def test_episode_truncates_at_horizon():
    env = CBMEnv(CPP_B)
    env.reset(seed=0)
    flags = [env.step(3)[2:4] for _ in range(100)]
    assert all(f == (False, False) for f in flags[:-1]) and flags[-1] == (False, True)


def test_seed_reproducibility():
    a = evaluate(GP_K2, ReplaceFailed(), n_episodes=3, seed=5)
    b = evaluate(GP_K2, ReplaceFailed(), n_episodes=3, seed=5)
    assert a.equals(b)


def test_first_passage_empty_path():
    d, t = first_passage(np.array([1.0, 2.0]), np.empty(0), np.empty((0, 2)), np.array([5.0, 5.0]))
    assert t is None and list(d) == [1.0, 2.0]


def test_process_means():
    rng = np.random.default_rng(0)
    tau, n = 10.0, 4000
    gp = GammaProcess(alpha=(0.4, 0.6), beta=(1.0, 1.2), shared=(0.3, 1.6))
    cpp = CompoundPoisson(lam=(0.1, 0.2), mu=(2.0, 2.5), sigma=(0.2, 0.3), shared=(0.04, 5.0, 0.4))
    for proc, expect in [
        (gp, [(0.4 / 1.0 + 0.3 / 1.6) * tau, (0.6 / 1.2 + 0.3 / 1.6) * tau]),
        (cpp, [(0.1 * 2.0 + 0.04 * 5.0) * tau, (0.2 * 2.5 + 0.04 * 5.0) * tau]),
    ]:
        tot = np.mean([proc.sample_path(rng, tau)[1].sum(axis=0) for _ in range(n)], axis=0)
        assert np.allclose(tot, expect, rtol=0.05)


def test_replace_all_costs_only_replacement_and_inspection_when_no_failure():
    sc = Scenario("t", Scripted([], np.empty((0, 2))), delta=(10.0, 10.0), c_R=(20.0, 50.0), H=100.0)
    df = evaluate(sc, Constant([1, 1]), n_episodes=1)
    assert df.cost_rate[0] == pytest.approx((12 * 10 + 70 * 10) / 100)


def test_threshold_policies():
    from policies.threshold import Opportunistic, SingleThreshold
    obs = np.array([1, 1, 0, 0.5, 0.8, 1.2], dtype=np.float32)  # component 3 failed
    assert list(action_to_vec(SingleThreshold(0.7).act(obs), 3)) == [0, 1, 1]
    assert list(action_to_vec(SingleThreshold([0.4, 0.9, 0.9]).act(obs), 3)) == [1, 0, 1]
    assert list(action_to_vec(Opportunistic(0.4, 0.9).act(obs), 3)) == [1, 1, 1]
    healthy = np.array([1, 1, 1, 0.5, 0.8, 0.6], dtype=np.float32)
    assert Opportunistic(0.4, 0.9).act(healthy) == 0
    assert list(action_to_vec(Opportunistic(0.6, 0.8).act(healthy), 3)) == [0, 1, 1]


def test_pareto_front():
    import pandas as pd
    from policies.tuning import pareto_front
    df = pd.DataFrame({"x": [1, 2, 3, 2], "y": [3, 2, 1, 3]})
    assert list(pareto_front(df, ["x", "y"]).index) == [0, 1, 2]


def test_multicritic_with_cost_weights_matches_scalar_reward():
    """Stored reward vector, weighted by w, must equal the env's scalar reward."""
    from agents.dqn import DQNConfig
    from agents.multicritic import MultiCriticAgent
    agent = MultiCriticAgent(CPP_B, DQNConfig(total_steps=1))
    env = CBMEnv(CPP_B)
    env.reset(seed=3)
    for a in [0, 0, 0, 3, 1, 0, 2, 0]:
        _, r, *_, info = env.step(a)
        stored = agent._stored_reward(r, info)
        assert float(stored @ (agent.w * agent.scales)) == pytest.approx(r)
        assert (stored <= 0).all() and (stored >= -1).all()


def test_agents_learn_shapes():
    from agents.dqn import DQNAgent, DQNConfig
    from agents.multicritic import MultiCriticAgent
    cfg = DQNConfig(total_steps=300, learning_starts=100, target_update=100)
    for agent in (DQNAgent(CPP_B, cfg), MultiCriticAgent(CPP_B, cfg, w=(15, 1, 100, 0))):
        log = agent.train(verbose=False)
        assert len(log) == 3
        obs = CBMEnv(CPP_B).reset(seed=0)[0]
        assert 0 <= agent.act(obs) < 4
    assert set(agent.critic_values(obs)) == {"downtime", "replacement_cost", "system_failure", "n_replacements"}
    assert all(v.shape == (4,) for v in agent.critic_values(obs).values())


def test_forced_repair_replaces_failed_component_even_if_agent_does_nothing():
    env = make_env([2, 5, 8], [[6, 1], [6, 1], [6, 1]], forced_repair=True)
    env.step(0)                      # component 1 fails at t=5
    _, r, *_, info = env.step(0)     # agent does nothing; repair is forced
    assert info["cost_vec"][1] == 20.0 and info["cost_vec"][3] == 1.0
    assert info["cost_vec"][2] == 1.0  # the repaired system runs and fails again at t=5
