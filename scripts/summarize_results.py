"""Builds the results table and convergence plots from the CSVLogger outputs.

For every regime it reads the latest finished run in ``<logs>/<regime>/`` (a run folder
with ``efficiency_stats.json``, which is written when training ends), or the folder given
with ``--run <regime>=<folder>``. It writes to ``<out>/``: ``results.md``, ``results.csv``,
``curves_val_miou.png`` and ``curves_train_loss.png``.

Usage:
    python scripts/summarize_results.py [--logs logs] [--out results] [--run lora=logs/lora/<folder>]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VAL_MIOU = "metrics/val_iou_all"
TRAIN_LOSS = "losses/train_loss_total"


def find_run_dir(logs: Path, regime: str, chosen: dict[str, Path]) -> Path | None:
    """The run folder given with --run, else the latest finished run of the regime."""
    if regime in chosen:
        return chosen[regime]
    # Run folders are named by start time, so sorting by name sorts by date
    finished = sorted(p.parent for p in (logs / regime).glob("*/efficiency_stats.json"))
    return finished[-1] if finished else None


def parse_run_args(values: list[str]) -> dict[str, Path]:
    runs = {}
    for value in values:
        regime, sep, folder = value.partition("=")
        if not sep:
            raise SystemExit(f"--run expects <regime>=<folder>, got {value!r}")
        runs[regime] = Path(folder)
    return runs


def read_series(metrics_csv: Path, key: str, x_key: str) -> tuple[list, list]:
    xs, ys = [], []
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get(key):
                xs.append(int(row[x_key]))
                ys.append(float(row[key]))
    return xs, ys


def smooth(values: list[float], window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--regimes", nargs="+", default=["full", "frozen", "lora"])
    parser.add_argument("--run", action="append", default=[], metavar="REGIME=FOLDER")
    args = parser.parse_args()
    chosen = parse_run_args(args.run)

    runs = {}
    for regime in args.regimes:
        run_dir = find_run_dir(args.logs, regime, chosen)
        if run_dir is None or not (run_dir / "metrics.csv").exists():
            print(f"Skipping {regime}: no finished run found under {args.logs / regime}")
            continue
        print(f"{regime}: {run_dir}")
        stats_path = run_dir / "efficiency_stats.json"
        runs[regime] = {
            "dir": run_dir,
            "stats": json.loads(stats_path.read_text()) if stats_path.exists() else {},
            "val": read_series(run_dir / "metrics.csv", VAL_MIOU, "epoch"),
            "train": read_series(run_dir / "metrics.csv", TRAIN_LOSS, "step"),
        }

    if not runs:
        raise SystemExit("No runs found")

    args.out.mkdir(parents=True, exist_ok=True)

    final_mious = {r: run["val"][1][-1] * 100 for r, run in runs.items() if run["val"][1]}
    reference = final_mious.get("full")

    rows = []
    for regime, run in runs.items():
        stats = run["stats"]
        miou = final_mious.get(regime)
        rows.append(
            {
                "regime": regime,
                "trainable_params": stats.get("params_trainable"),
                "trainable_pct": stats.get("params_trainable_pct"),
                "backbone_trainable_params": stats.get("params_backbone_trainable"),
                "backbone_trainable_pct": (
                    100 * stats["params_backbone_trainable"] / stats["params_backbone"]
                    if "params_backbone" in stats
                    else None
                ),
                "gflops": stats.get("gflops"),
                "final_miou": miou,
                "best_miou": max(run["val"][1]) * 100 if run["val"][1] else None,
                "delta_vs_full": (
                    miou - reference
                    if miou is not None and reference is not None
                    else None
                ),
                "train_time_h": (
                    stats["train_time_sec"] / 3600 if "train_time_sec" in stats else None
                ),
                "peak_gpu_mem_gb": stats.get("peak_gpu_mem_gb"),
                "run_dir": str(run["dir"]),
            }
        )

    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec):
        return "–" if value is None else format(value, spec)

    lines = [
        "| Regime | Trainable params | Trainable % | Backbone trainable % | GFLOPs "
        "| mIoU (final) | mIoU (best) | Δ vs full | Train time (h) | Peak mem (GB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {fmt(row['trainable_params'], ',')} "
            f"| {fmt(row['trainable_pct'], '.2f')} | {fmt(row['backbone_trainable_pct'], '.3f')} "
            f"| {fmt(row['gflops'], '.1f')} | {fmt(row['final_miou'], '.2f')} "
            f"| {fmt(row['best_miou'], '.2f')} | {fmt(row['delta_vs_full'], '+.2f')} "
            f"| {fmt(row['train_time_h'], '.2f')} | {fmt(row['peak_gpu_mem_gb'], '.1f')} |"
        )
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    for regime, run in runs.items():
        epochs, mious = run["val"]
        ax.plot([e + 1 for e in epochs], [m * 100 for m in mious], marker="o", label=regime)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mIoU (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "curves_val_miou.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for regime, run in runs.items():
        steps, losses = run["train"]
        if not losses:
            continue
        window = max(1, len(losses) // 50)
        ax.plot(steps[window - 1 :], smooth(losses, window), label=regime)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss (moving average)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "curves_train_loss.png", dpi=200)
    plt.close(fig)

    print(f"Wrote results to {args.out}/")


if __name__ == "__main__":
    main()
