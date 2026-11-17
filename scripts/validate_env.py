"""E0: check the simulator against the paper's fixed-rule baselines (Tables 4, 5, 7, 8, 11).

Run from the project root:  python -m scripts.validate_env [--episodes 1000]
"""
import argparse
from pathlib import Path

import pandas as pd

from cbm.config import SCENARIOS
from cbm.env import CBMEnv
from cbm.metrics import evaluate
from policies.fixed import Constant, paper_baselines

# Cost rates reported in the paper, keyed by scenario then baseline letter.
PAPER = {
    "cpp_A": {"a": 15.85, "b": 17.82, "c": 19.95, "d": 8.49, "e": 6.43},
    "cpp_B": {"a": 15.93, "b": 17.89, "c": 20.40, "d": 9.18, "e": 7.24},
    "gp_K2": {"a": 15.96, "b": 17.94, "c": 20.74, "d": 8.35, "e": 9.12},
    "gp_K2_joint": {"d": 7.36},
    "gp_K12": {"a": 20.88, "d": 25.01, "e": 19.12},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    args = p.parse_args()

    rows = []
    for name, paper in PAPER.items():
        sc = SCENARIOS[name]
        for label, policy in paper_baselines(sc.K).items():
            if label[0] not in paper:
                continue
            df = evaluate(sc, policy, n_episodes=args.episodes)
            # The paper fixes a_0 = 0 (99 decisions); our agent also acts at t = 0.
            # For always-replace rules that is one extra, useless replacement round.
            extra = 0.0
            if isinstance(policy, Constant):
                env = CBMEnv(sc)
                env.reset(seed=0)
                extra = env.step(policy.action)[4]["cost_vec"][1] / sc.H
            rows.append({
                "scenario": name, "baseline": label,
                "ours_mean": df.cost_rate.mean() - extra, "ours_sd": df.cost_rate.std(ddof=1),
                "paper": paper[label[0]],
                "downtime": df.downtime.mean(),
                "sys_fail": df.n_system_failures.mean(),
                "n_repl": df.n_replacements.mean(),
            })
    out = pd.DataFrame(rows)
    out["diff"] = out.ours_mean - out.paper
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/e0_validation.csv", index=False)


if __name__ == "__main__":
    main()
