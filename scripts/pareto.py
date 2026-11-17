"""E4: compare multi-critic DDQN with threshold heuristics under the same weights.

For every failure penalty f that a multi-critic agent was trained with, the heuristic
that minimises the same weighted objective w = (c_D, 1, f, 0) is picked from the grid
and re-evaluated on the evaluation seeds.

Run from the project root (after tune_heuristics and the multi-critic runs):
  python -m scripts.pareto --scenario gp_K4
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cbm.config import SCENARIOS
from cbm.metrics import evaluate
from policies.tuning import best_weighted, load_grid, make_policy, pareto_front

X, Y = "n_system_failures", "cost_rate"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def seed_means(pattern: str) -> pd.DataFrame:
    """Per-seed mean metrics for every results/<pattern> run directory."""
    rows = []
    for d in Path("results").glob(pattern):
        m = re.search(r"-f([\d.]+)-r", d.name)
        rows.append({"penalty": float(m[1]) if m else float("nan"),
                     **pd.read_csv(d / "eval.csv")[[X, Y]].mean()})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="gp_K4", choices=SCENARIOS)
    p.add_argument("--eval-episodes", type=int, default=1000)
    args = p.parse_args()
    sc = SCENARIOS[args.scenario]

    grid = load_grid(f"results/heuristic_grid_{sc.name}.csv")
    front = pareto_front(grid, [Y, X])
    mc = seed_means(f"mcddqn-f*-r0_{sc.name}_s*")
    ddqn = seed_means(f"ddqn_{sc.name}_s*")

    rows = []
    for f, runs in mc.groupby("penalty"):
        best = best_weighted(grid, (sc.c_D, 1.0, f, 0.0))
        h = evaluate(sc, make_policy(best), n_episodes=args.eval_episodes)[[X, Y]].mean()
        objective = lambda m: m[Y] + f * m[X] / sc.H   # weighted cost per unit time
        rows.append({
            "fail_penalty": f, "heur_family": best["family"], "heur_params": best["params"],
            "heur_cost_rate": h[Y], "heur_failures": h[X], "heur_objective": objective(h),
            "mc_seeds": len(runs), "mc_cost_rate": runs[Y].mean(), "mc_failures": runs[X].mean(),
            "mc_objective": objective(runs).mean(), "mc_objective_sd": objective(runs).std(),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if len(ddqn):
        print(f"\nDDQN ({len(ddqn)} seeds): cost rate {ddqn[Y].mean():.2f}, failures {ddqn[X].mean():.1f}")
    table.to_csv(f"results/pareto_{sc.name}.csv", index=False)

    # --- figure: both objectives are "lower is better", so down-left is better ---
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(front[X], front[Y], color=BLUE, lw=2, zorder=2,
            label="Threshold heuristics (non-dominated grid points)")
    ax.scatter(table.heur_failures, table.heur_cost_rate, s=60, marker="D", color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=3, label="Heuristic tuned for the same weights")
    ax.scatter(mc[X], mc[Y], s=14, color=ORANGE, alpha=0.35, linewidth=0, zorder=3)
    ax.scatter(table.mc_failures, table.mc_cost_rate, s=80, color=ORANGE, edgecolor=SURFACE,
               linewidth=2, zorder=4, label="Multi-critic DDQN (mean; small dots = seeds)")
    for i, r in enumerate(table.itertuples()):  # alternate above/below so neighbours don't collide
        ax.annotate(f"f={r.fail_penalty:g}", (r.mc_failures, r.mc_cost_rate),
                    textcoords="offset points", xytext=(8, 6 if i % 2 == 0 else -13),
                    fontsize=9, color=INK2)
    if len(ddqn):
        ax.scatter(ddqn[X].mean(), ddqn[Y].mean(), s=80, marker="s", color=AQUA, edgecolor=SURFACE,
                   linewidth=2, zorder=4, label="DDQN, cost only (mean)")
    ax.set_xlabel(f"System failures per episode (H = {sc.H:g})", color=INK2)
    ax.set_ylabel("Cost rate", color=INK2)
    ax.set_title(f"Cost rate vs. system failures, {sc.name} (lower-left is better)",
                 loc="left", color=INK, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"pareto_{sc.name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nsaved {out / f'pareto_{sc.name}.png'}")


if __name__ == "__main__":
    main()
