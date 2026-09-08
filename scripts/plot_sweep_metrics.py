"""Plot sweep metrics for each configuration in a single grid.

This script looks for run directories under `pilot_logs/` and optionally reads a
`sweep_summary.json` file under `runs/` to determine the run order. For each run,
it scans for CSV metrics files produced by Lightning (e.g. `metrics.csv` inside a
`lightning_logs` subfolder) and plots one chosen metric over epochs/steps.

If a run does not yet contain epoch-level metrics, it will be plotted as an empty
panel with a note instead of failing.
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
    """Return (x, y, metric_name) or None if no series can be found.

    If preferred_metric is provided, attempt to find that column (exact or substring
    match) in available CSVs and return its series even if it has only one point.
    """
    csv_paths = sorted(run_dir.glob("**/metrics.csv"))
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

        fieldnames = reader.fieldnames or []

        # If user provided a preferred metric, try to locate a matching column.
        metric_name = None
        if preferred_metric:
            # exact match
            for fn in fieldnames:
                if fn == preferred_metric:
                    metric_name = fn
                    break
            # case-insensitive exact
            if metric_name is None:
                for fn in fieldnames:
                    if fn.lower() == preferred_metric.lower():
                        metric_name = fn
                        break
            # substring match
            if metric_name is None:
                for fn in fieldnames:
                    if preferred_metric.lower() in fn.lower():
                        metric_name = fn
                        break
            if metric_name is None:
                # preferred metric asked but not found in this CSV; try next CSV
                continue
        else:
            metric_name = _choose_metric_name(fieldnames)
            if metric_name is None:
                continue

        xs, ys = [], []
        for row in rows:
            x = None
            for candidate in ["epoch", "step", "global_step", "step_idx"]:
                if candidate in row:
                    x = _safe_float(row[candidate])
                    if x is not None:
                        break
            if x is None:
                x = len(xs)

            y = _safe_float(row.get(metric_name))
            if y is not None:
                xs.append(float(x))
                ys.append(float(y))

        # If preferred_metric was requested, accept series with >=1 points, otherwise require >=2
        min_points = 1 if preferred_metric else 2
        if len(ys) >= min_points:
            return xs, ys, metric_name

    return None


def _discover_run_dirs(root: Path, pilot_root: Path | None = None, run_name: str | None = None, run_dir: Path | None = None, latest: bool = False):
    pilot_root = (pilot_root or (root / "pilot_logs")).resolve()

    # Determine sensible sweep summary path next to the pilot_logs folder (e.g., sweeps/<name>/runs/sweep_summary.json)
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
                            # If the sweeper recorded an explicit run_dir, prefer it
                            run_dir_recorded = entry.get("run_dir")
                            if run_dir_recorded:
                                cand = Path(run_dir_recorded).resolve()
                                if cand.exists() and cand.is_dir():
                                    return [cand]
                            # Otherwise try to find matching folder under pilot_root
                            matches = sorted(pilot_root.glob(f"{run_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if matches:
                                return [matches[0].resolve()]
            except Exception:
                    pass
        return []

    if latest:
        candidates = sorted([p for p in pilot_root.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[:1]

    # Prefer using the sweep summary to determine which runs to plot so we avoid
    # plotting unrelated logger folders like `wandb` or `eomt-pilot` that are not
    # per-run directories. If no summary is available, fall back to listing the
    # pilot_logs/ subfolders.
    run_dirs = []
    if summary_path is not None:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in data:
                    run_name = entry.get("run_name")
                    if not run_name:
                        continue
                    # Prefer an explicit run_dir recorded by the sweeper
                    run_dir_recorded = entry.get("run_dir")
                    if run_dir_recorded:
                        cand = Path(run_dir_recorded).resolve()
                        if cand.exists() and cand.is_dir():
                            run_dirs.append(cand)
                            continue

                    # run directories are commonly created as <logger_name>_<timestamp>.
                    matches = sorted(pilot_root.glob(f"{run_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if matches:
                        run_dirs.append(matches[0].resolve())
                    else:
                        candidate = (pilot_root / run_name).resolve()
                        run_dirs.append(candidate)
        except Exception:
            run_dirs = []

    # Fallback: list all subdirectories under pilot_logs if summary not present or empty
    if not run_dirs:
        for p in sorted(pilot_root.glob("*")):
            if p.is_dir():
                    run_dirs.append(p)

    # de-duplicate while preserving order (compare resolved paths when possible)
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
        raise FileNotFoundError("No run folders with metrics were found under pilot_logs/. Run the sweep first.")

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
        title = run_dir.name
        status = summary_info.get(run_dir.name, {})
        run_rc = status.get("return_code")
        series = _collect_epoch_series(run_dir, preferred_metric=metric)
        if series is None:
            # If there is a sweep summary with final metrics for this run, plot the final value as a single point
            if run_rc not in (None, 0):
                msg = f"Run failed\n(rc={run_rc})"
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                ax.set_axis_off()
                continue

            final_metrics = None
            if status.get("metrics"):
                # metrics in summary are stored as a list of dicts (Lightning validate result)
                try:
                    mlist = status.get("metrics")
                    if isinstance(mlist, list) and mlist:
                        final_metrics = mlist[0]
                    elif isinstance(mlist, dict):
                        final_metrics = mlist
                except Exception:
                    final_metrics = None

            if final_metrics:
                # choose the best metric key available (respect user-specified metric if provided)
                keys = list(final_metrics.keys())
                if metric:
                    # try exact, case-insensitive, or substring
                    metric_name = None
                    if metric in final_metrics:
                        metric_name = metric
                    else:
                        for k in keys:
                            if k.lower() == metric.lower():
                                metric_name = k
                                break
                        if metric_name is None:
                            for k in keys:
                                if metric.lower() in k.lower():
                                    metric_name = k
                                    break
                else:
                    metric_name = _choose_metric_name(keys)

                if metric_name and metric_name in final_metrics:
                    y = final_metrics.get(metric_name)
                    try:
                        y = float(y)
                        xs = [0]
                        ys = [y]
                        ax.plot(xs, ys, marker="o", linewidth=2)
                        ax.set_title(title, fontsize=9)
                        ax.set_xlabel("epoch/step")
                        ax.set_ylabel(metric_name)
                        ax.grid(alpha=0.25)
                    except Exception:
                        ax.text(0.5, 0.5, "No numeric final metric\nfound", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(title)
                        ax.set_axis_off()
                    continue

            # default message when no per-epoch CSV and no final metrics
            msg = "No metric curve\nfound"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_axis_off()
            continue

        xs, ys, metric_name = series
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch/step")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)

    for i in range(len(run_dirs), len(flat_axes)):
        flat_axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pilot-logs", type=Path, default=ROOT / "pilot_logs", help="Root folder containing run folders (pilot_logs)")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", type=str, default=None, help="Specific run directory name under pilot_logs/")
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact run directory path to plot")
    parser.add_argument("--latest", action="store_true", help="Plot only the newest run under pilot_logs/")
    parser.add_argument("--metric", type=str, default=None, help="Metric column name to plot (exact or substring).")
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_root = args.pilot_logs.resolve()

    # If the user didn't pass a custom --out, place the plot next to the sweep's runs folder
    default_out_resolved = DEFAULT_OUTPUT.resolve()
    if args.out.resolve() == default_out_resolved:
        candidate_out = (pilot_root.parent / "runs" / "sweep_metrics_grid.png").resolve()
        out = candidate_out
    else:
        out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(root, pilot_root=pilot_root, run_name=args.run_name, run_dir=args.run_dir, latest=args.latest)
    if not run_dirs:
        if args.run_name or args.run_dir or args.latest:
            target = args.run_name or str(args.run_dir) or "latest run"
            print(f"No matching run directory found for: {target}")
        else:
            print(f"No run directories found under {pilot_root}. Try running the sweep first.")
        return 1

    _plot_grid(run_dirs, out, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
