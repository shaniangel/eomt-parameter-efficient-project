"""Plot sweep metrics for each configuration in a single grid or overlaid plot.

This script looks for run directories under `pilot_logs/` and optionally reads a
`sweep_summary.json` file under `runs/` to determine the run order. For each run,
it scans for CSV metrics files produced by Lightning and plots one chosen metric.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "runs" / "sweep_summary.json"
DEFAULT_OUTPUT = ROOT / "runs" / "sweep_metrics_grid.png"


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _choose_metric_name(fieldnames):
    keys = [k.strip() for k in fieldnames]
    key_map = {k.lower(): k for k in keys}

    preferred = [
        "val_miou",
        "mIoU",
        "miou",
        "val_iou",
        "iou",
        "val_loss",
        "loss",
        "train_loss",
        "validation_loss",
        "val_accuracy",
        "accuracy",
    ]
    for p in preferred:
        if p in key_map:
            return key_map[p]

    for k in keys:
        lower = k.lower()
        if lower.startswith("val_") or lower.startswith("train_") or lower.endswith("loss") or "iou" in lower or "acc" in lower:
            return k

    return keys[0] if keys else None


def _collect_epoch_series(run_dir: Path, preferred_metric: str | None = None):
    csv_paths = sorted(run_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_paths:
        return None

    for csv_path in csv_paths:
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            continue

        if not rows:
            continue

        fieldnames = [fn.strip() for fn in (reader.fieldnames or []) if fn]

        candidates = []
        if preferred_metric:
            candidates = [preferred_metric]
        else:
            candidates = ["train_loss", "loss", "train_loss_epoch", "train_loss_step", "val_loss"]

        metric_name = None
        for cand in candidates:
            for fn in fieldnames:
                if fn == cand or fn.lower() == cand.lower():
                    metric_name = fn
                    break
            if metric_name:
                break

        if not metric_name:
            for fn in fieldnames:
                if "loss" in fn.lower() or "iou" in fn.lower():
                    metric_name = fn
                    break

        if not metric_name:
            continue

        xs, ys = [], []
        for idx, row in enumerate(rows):
            y = _safe_float(row.get(metric_name))
            if y is None:
                continue

            x = _safe_float(row.get("epoch"))
            if x is None:
                x = _safe_float(row.get("step"))
            if x is None:
                x = float(idx)

            xs.append(x)
            ys.append(y)

        if len(ys) >= 1:
            return xs, ys, metric_name

    return None


def _discover_run_dirs(root: Path, pilot_root: Path | None = None, run_name: str | None = None, run_dir: Path | None = None, latest: bool = False):
    pilot_root = (pilot_root or (root / "pilot_logs")).resolve()

    summary_candidates = [pilot_root.parent / "runs" / "sweep_summary.json", (root / "runs" / "sweep_summary.json").resolve()]
    summary_path = None
    for p in summary_candidates:
        if p.exists():
            summary_path = p.resolve()
            break

    if run_dir is not None:
        return [run_dir.resolve()]

    if run_name is not None:
        candidate = (pilot_root / run_name).resolve()
        if candidate.exists() and candidate.is_dir():
            return [candidate]
        if summary_path is not None:
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                for entry in data:
                    if entry.get("run_name") == run_name:
                        run_dir_recorded = entry.get("run_dir")
                        if run_dir_recorded:
                            cand = Path(run_dir_recorded).resolve()
                            if cand.exists() and cand.is_dir():
                                return [cand]
                        matches = sorted(pilot_root.glob(f"{run_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if matches:
                            return [matches[0].resolve()]
            except Exception:
                pass
        return []

    if latest:
        candidates = sorted([p for p in pilot_root.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[:1]

    run_dirs = []
    if summary_path is not None:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in data:
                run_name = entry.get("run_name")
                if not run_name:
                    continue
                run_dir_recorded = entry.get("run_dir")
                if run_dir_recorded:
                    cand = Path(run_dir_recorded).resolve()
                    if cand.exists() and cand.is_dir():
                        run_dirs.append(cand)
                        continue

                matches = sorted(pilot_root.glob(f"{run_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    run_dirs.append(matches[0].resolve())
                else:
                    candidate = (pilot_root / run_name).resolve()
                    run_dirs.append(candidate)
        except Exception:
            run_dirs = []

    if not run_dirs:
        for p in sorted(pilot_root.glob("*")):
            if p.is_dir():
                run_dirs.append(p)

    seen = set()
    deduped = []
    for d in run_dirs:
        try:
            key = d.resolve()
        except Exception:
            key = d
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _plot_grid(run_dirs, out_path: Path, metric: str | None = None):
    if not run_dirs:
        raise FileNotFoundError("No run folders with metrics were found under pilot_logs/.")

    rows = max(1, int(len(run_dirs) ** 0.5))
    cols = (len(run_dirs) + rows - 1) // rows

    summary_path = out_path.parent / "sweep_summary.json"
    summary_info = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in data:
                run_name = entry.get("run_name")
                if run_name:
                    summary_info[run_name] = entry
        except Exception:
            summary_info = {}

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, run_dir in enumerate(run_dirs):
        ax = flat_axes[idx]
        run_name = run_dir.name
        title = run_name
        series = _collect_epoch_series(run_dir, preferred_metric=metric)

        if series is None:
            msg = "No metric curve\nfound"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
            continue

        xs, ys, metric_name = series
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)

    for i in range(len(run_dirs), len(flat_axes)):
        flat_axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid plot to: {out_path}")


def _plot_overlay(run_dirs, out_path: Path, metric: str | None = None):
    if not run_dirs:
        raise FileNotFoundError("No run folders with metrics were found under pilot_logs/.")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    metric_name_used = metric or "train_loss"

    for run_dir in run_dirs:
        run_name = run_dir.name
        series = _collect_epoch_series(run_dir, preferred_metric=metric)
        if series is None:
            continue
        xs, ys, metric_name = series
        metric_name_used = metric_name
        ax.plot(xs, ys, marker="o", linewidth=2, label=run_name)
        plotted_any = True

    if not plotted_any:
        print("No valid metric series found across any runs to overlay.")
        plt.close(fig)
        return

    ax.set_title(f"Sweep Comparison ({metric_name_used})", fontsize=12)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name_used)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pilot-logs", type=Path, default=ROOT / "pilot_logs", help="Root folder containing run folders")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--metric", type=str, default=None, help="Metric column name to plot")
    parser.add_argument("--overlay", action="store_true", help="Plot all runs overlaid on a single plot with a legend")
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_root = args.pilot_logs.resolve()

    if args.out.resolve() == DEFAULT_OUTPUT.resolve():
        filename = "sweep_metrics_overlay.png" if args.overlay else "sweep_metrics_grid.png"
        out = (pilot_root.parent / "runs" / filename).resolve()
    else:
        out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(root, pilot_root=pilot_root, run_name=args.run_name, run_dir=args.run_dir, latest=args.latest)
    if not run_dirs:
        print(f"No run directories found under {pilot_root}.")
        return 1

    if args.overlay:
        _plot_overlay(run_dirs, out, metric=args.metric)
    else:
        _plot_grid(run_dirs, out, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())