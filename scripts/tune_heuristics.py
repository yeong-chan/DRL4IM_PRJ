"""E2: grid-search the threshold heuristics and evaluate the cost-optimal ones.

Run from the project root:  python -m scripts.tune_heuristics --scenario cpp_A
"""
import argparse
from pathlib import Path

from cbm.config import SCENARIOS
from cbm.metrics import evaluate, summarize
from policies.tuning import best_weighted, grid_search, make_policy, pareto_front


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="cpp_A", choices=SCENARIOS)
    p.add_argument("--tune-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=1000)
    args = p.parse_args()

    sc = SCENARIOS[args.scenario]
    grid = grid_search(sc, n_episodes=args.tune_episodes)
    out = Path("results")
    out.mkdir(exist_ok=True)
    grid.to_csv(out / f"heuristic_grid_{sc.name}.csv", index=False)

    w_cost = (sc.c_D, 1.0, 0.0, 0.0)  # the paper's cost objective
    print(f"{sc.name}: {len(grid)} candidates, {args.tune_episodes} episodes each")
    for family, sub in grid.groupby("family"):
        best = best_weighted(sub, w_cost)
        df = evaluate(sc, make_policy(best), n_episodes=args.eval_episodes)
        df.to_csv(out / f"heuristic_best_{family}_{sc.name}.csv", index=False)
        print(f"\nbest {family} for cost: thresholds {best['params']}")
        print(summarize(df).to_string(float_format=lambda x: f"{x:.2f}"))

    front = pareto_front(grid, ["cost_rate", "n_system_failures"])
    front.to_csv(out / f"heuristic_pareto_{sc.name}.csv", index=False)
    print(f"\nPareto front (cost rate vs system failures): {len(front)} points")
    print(front[["family", "params", "cost_rate", "n_system_failures", "downtime", "n_replacements"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
