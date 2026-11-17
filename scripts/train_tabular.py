"""E3: train a tabular agent (MC / SARSA / Q-learning) on a discretised K=2 scenario.

Run from the project root:  python -m scripts.train_tabular --algo qlearning --bins 10
"""
import argparse
from pathlib import Path

import pandas as pd

from agents.tabular import ALGOS, TabularAgent, TabularConfig
from cbm.config import SCENARIOS
from cbm.metrics import evaluate, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="cpp_A", choices=SCENARIOS)
    p.add_argument("--algo", default="qlearning", choices=ALGOS)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=1000)
    args = p.parse_args()

    sc = SCENARIOS[args.scenario]
    agent = TabularAgent(sc, TabularConfig(algo=args.algo, n_bins=args.bins,
                                           episodes=args.episodes, seed=args.seed))
    log = agent.train()

    tag = f"{args.algo}_b{args.bins}_{sc.name}_s{args.seed}"
    out = Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    df = evaluate(sc, agent, n_episodes=args.eval_episodes)
    df.to_csv(out / "eval.csv", index=False)
    print(f"{tag}: greedy policy over {args.eval_episodes} episodes")
    print(summarize(df).to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
