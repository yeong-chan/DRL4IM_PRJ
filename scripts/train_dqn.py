"""Train DQN or DDQN on a scenario and evaluate the greedy policy.

Run from the project root, e.g.:
  python -m scripts.train_dqn --scenario cpp_A --double --seed 0
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from agents.dqn import DQNAgent, DQNConfig
from agents.multicritic import MultiCriticAgent
from cbm.config import SCENARIOS
from cbm.metrics import evaluate, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="cpp_A", choices=SCENARIOS)
    p.add_argument("--double", action="store_true", help="Double DQN (default: plain DQN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--multicritic", action="store_true", help="one critic per objective")
    p.add_argument("--fail-penalty", type=float, default=0.0,
                   help="multi-critic: extra cost per system failure")
    p.add_argument("--repl-penalty", type=float, default=0.0,
                   help="multi-critic: extra cost per replacement")
    args = p.parse_args()

    torch.set_num_threads(1)  # small MLP: one thread is fastest and lets seeds run in parallel
    sc = SCENARIOS[args.scenario]
    cfg = DQNConfig(double=args.double, seed=args.seed, total_steps=args.steps, hidden=args.hidden)
    name = "ddqn" if args.double else "dqn"
    if args.multicritic:
        w = (sc.c_D, 1.0, args.fail_penalty, args.repl_penalty)
        agent = MultiCriticAgent(sc, cfg, w)
        name = f"mc{name}-f{args.fail_penalty:g}-r{args.repl_penalty:g}"
    else:
        agent = DQNAgent(sc, cfg)
    log = agent.train()

    tag = f"{name}_{sc.name}_s{args.seed}"
    out = Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    agent.save(out / "q.pt")

    df = evaluate(sc, agent, n_episodes=args.eval_episodes)
    df.to_csv(out / "eval.csv", index=False)
    print(f"\n{tag}: greedy policy over {args.eval_episodes} episodes")
    print(summarize(df).to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
