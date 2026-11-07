"""Run every model version and regenerate the figures and results used in docs/ml_report.md.

Usage::

    python3 -m ml.report

Writes to docs/ml_report/:
    results.json            all metrics, chosen hyperparameters and CV tables
    pred_vs_actual.png      predicted vs. actual test-set price, both models, both datasets
    knn_cv.png              KNN cross-validated RMSE for each k
    gd_loss.png             gradient descent training loss per step
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml import gradient_descent, knn  # noqa: E402
from ml.data import REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "ml_report"

# Categorical slots 1 and 2 (blue, orange) of the validated default palette; aqua for the 2nd KNN weighting.
COLOR = {"knn": "#2a78d6", "gradient_descent": "#eb6834"}
WEIGHT_COLOR = {"uniform": "#2a78d6", "distance": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
LABEL = {"knn": "KNN", "gradient_descent": "Gradient descent"}
TITLE = {"car_data": "Car_data.csv", "carprice": "CarPrice_Assignment.csv"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})


def _strip_arrays(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("y_test", "y_pred", "loss_history")}


def plot_pred_vs_actual(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, ds in zip(axes, ("car_data", "carprice")):
        rows = [r for r in results if r["dataset"] == ds]
        lo = min(min(r["y_test"] + r["y_pred"]) for r in rows)
        hi = max(max(r["y_test"] + r["y_pred"]) for r in rows)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", label="perfect prediction")
        for r, marker in zip(rows, ("o", "s")):
            ax.scatter(r["y_test"], r["y_pred"], s=22, marker=marker, color=COLOR[r["model"]],
                       alpha=0.7, edgecolors=SURFACE, linewidths=0.8,
                       label=f"{LABEL[r['model']]} (test R² {r['test_metrics']['r2']:.2f})")
        cur = rows[0]["currency"]
        ax.set_title(TITLE[ds], loc="left", fontweight="bold")
        ax.set_xlabel(f"Actual price ({cur})")
        ax.set_ylabel(f"Predicted price ({cur})")
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.legend(loc="upper left")
    fig.suptitle("Predicted vs. actual price on the held-out test set", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pred_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_knn_cv(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axes, ("car_data", "carprice")):
        r = next(x for x in results if x["dataset"] == ds and x["model"] == "knn")
        for weights in ("uniform", "distance"):
            rows = [row for row in r["cv_table"] if row["weights"] == weights]
            ax.plot([row["k"] for row in rows], [row["cv_rmse"] for row in rows], color=WEIGHT_COLOR[weights],
                    linewidth=2, marker="o", markersize=4, label=f"{weights} weighting")
        best = r["best_params"]
        best_rmse = next(row["cv_rmse"] for row in r["cv_table"]
                         if row["k"] == best["k"] and row["weights"] == best["weights"])
        ax.scatter([best["k"]], [best_rmse], s=90, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
        ax.annotate(f"chosen: k={best['k']}", (best["k"], best_rmse), xytext=(8, 8),
                    textcoords="offset points", color=INK)
        ax.set_title(TITLE[ds], loc="left", fontweight="bold")
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel(f"5-fold CV RMSE ({r['currency']})")
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend()
    fig.suptitle("KNN: choosing k by cross-validation (lower is better)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "knn_cv.png", dpi=150)
    plt.close(fig)


def plot_gd_loss(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axes, ("car_data", "carprice")):
        r = next(x for x in results if x["dataset"] == ds and x["model"] == "gradient_descent")
        # Step 0 is the starting loss; plot against step + 1 so the log x-axis can show it.
        steps = range(1, len(r["loss_history"]) + 1)
        ax.plot(steps, r["loss_history"], color=COLOR["gradient_descent"], linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLE[ds], loc="left", fontweight="bold")
        ax.text(0.98, 0.95, f"{r['n_iter']:,} steps, learning rate {r['learning_rate']:.3f}",
                transform=ax.transAxes, ha="right", va="top", color=MUTED)
        ax.set_xlabel("Gradient descent step (log scale)")
        ax.set_ylabel("Training loss J (log-price units, log scale)")
    fig.suptitle("Gradient descent: training loss falls every step until it converges",
                  x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gd_loss.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        knn.run_car_data(), knn.run_carprice(),
        gradient_descent.run_car_data(), gradient_descent.run_carprice(),
    ]
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump([_strip_arrays(r) for r in results], f, indent=2)
    plot_pred_vs_actual(results)
    plot_knn_cv(results)
    plot_gd_loss(results)
    print(f"\nWrote results and figures to {Path(OUT_DIR).relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
