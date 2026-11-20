"""Plot the verified checkpoint trajectories with the four-seed range."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.curve.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(rows) != 672:
        raise ValueError("Expected 8 complete series x 84 checkpoints")
    checkpoint_steps = sorted({int(row["step"]) for row in rows})
    step_k = np.array(checkpoint_steps) / 1000
    values = {
        arm: np.array([[float(r["pooled_pct"]) for r in sorted(
            (row for row in rows if row["arm"] == arm and int(row["seed"]) == seed),
            key=lambda row: int(row["step"]),
        )] for seed in range(4)])
        for arm in ("per", "uniform")
    }
    if len(checkpoint_steps) != 84 or any(block.shape != (4, 84) for block in values.values()):
        raise ValueError("Incomplete checkpoint values")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#15273b", "text.color": "#15273b", "figure.dpi": 140})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.45]})
    colors = {"per": "#185a89", "uniform": "#aa4c20"}
    labels = {"per": "PER", "uniform": "Uniform replay"}
    panels = (
        (axes[0], "Initial learning", 0, 50),
        (axes[1], "Early checkpoints near target", 12, 100),
        (axes[2], "100k–1.2m interactions", 100, 1200),
    )
    upper_limit = max(80, np.ceil(max(series.max() for series in values.values())) + 0.5)
    for axis, title, start, stop in panels:
        chosen = (step_k >= start) & (step_k <= stop)
        for arm, data in values.items():
            view = data[:, chosen]
            axis.fill_between(step_k[chosen], view.min(0), view.max(0), color=colors[arm], alpha=0.14)
            axis.plot(step_k[chosen], view.mean(0), lw=1.6, marker=".", ms=3.5,
                      color=colors[arm], label=labels[arm])
        axis.axhline(75, color="#526576", lw=1, ls="--", label="75% target")
        axis.set(title=title, xlim=(start, stop), xlabel="Training interactions (thousands)")
        if start > 0:
            axis.set_ylim(70, upper_limit)
        axis.grid(axis="y", alpha=0.2)
    for step in (18, 40):
        axes[1].axvline(step, color="#8292a0", ls=":", lw=1)
        axes[1].text(step + 1.4, 0.05, f"{step}k", transform=axes[1].get_xaxis_transform(),
                     fontsize=8, color="#536777")
    axes[0].set_ylabel("Whole-window variance reduction (%)")
    axes[2].legend(loc="lower right", frameon=False)
    fig.suptitle("Matched convergence experiment on the archived window\n"
                 "Four-seed means with min–max bands · equal initial and final energy", fontsize=12)
    for ext in ("pdf", "png"):
        output = args.output_dir / f"checkpoint_convergence.{ext}"
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

