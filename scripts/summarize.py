"""Collect every results/<method>_<scenario>_s<seed>/eval.csv into one table.

Run from the project root:  python -m scripts.summarize
For learned agents: mean over seeds of the per-seed mean, and SD across seeds.
"""
import re
from pathlib import Path

import pandas as pd

from cbm.config import SCENARIOS

METRICS = ["cost_rate", "downtime", "n_system_failures", "n_replacements"]


def main():
    rows = []
    names = sorted(SCENARIOS, key=len, reverse=True)  # match gp_K2_joint before gp_K2
    for f in Path("results").glob("*/eval.csv"):
        m = re.fullmatch(rf"(.+)_({'|'.join(names)})_s(\d+)", f.parent.name)
        if m:
            rows.append({"scenario": m[2], "method": m[1], "seed": int(m[3]),
                         **pd.read_csv(f)[METRICS].mean()})
    for f in Path("results").glob("heuristic_best_*.csv"):
        m = re.fullmatch(rf"heuristic_best_(.+)_({'|'.join(names)})", f.stem)
        rows.append({"scenario": m[2], "method": f"threshold-{m[1]}", "seed": 0,
                     **pd.read_csv(f)[METRICS].mean()})
    if not rows:
        print("no results found")
        return
    df = pd.DataFrame(rows)
    g = df.groupby(["scenario", "method"])
    table = g[METRICS].mean()
    table.insert(1, "cost_rate_sd_seeds", g["cost_rate"].std())
    table.insert(0, "n_seeds", g.size())
    table = table.sort_values(["scenario", "cost_rate"])
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))
    table.to_csv("results/summary.csv")


if __name__ == "__main__":
    main()
