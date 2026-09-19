"""Plot sweep metrics and generate clean comparison CSV.

This script scans run directories under `pilot_logs/`, generates dedicated .png
plots overlaying all runs for target metrics, extracts static model efficiency stats
(Params, GFLOPs) and final evaluation metrics, and exports a clean `comparison.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Metrics to generate PNG line plots for
TARGET_METRICS = [
    "metrics/val_iou_all",
    "metrics/val_iou_present",
    "losses/train_loss_cross_entropy",
    "losses/train_loss_total",
]

# Patterns to filter out parameter-level noise from comparison.csv
IGNORE_PREFIXES = ("lr-", "lr/", "lr_")
EFFICIENCY_KEYWORDS = ("param", "flop", "mac", "fps", "throughput", "gflop")


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (int, float, str, bool)) or v is None:
            items.append((new_key, v))
    return dict(items)


def _collect_all_metrics(run_dirs: list[Path]):
    """Collects metric series across all runs for plotting."""
    data_by_metric: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    ignored_cols = {"epoch", "step", "_step", "timestamp", "created_at"}

    for run_dir in run_dirs:
        run_name = run_dir.name
        csv_paths = sorted(
            run_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not csv_paths:
            continue

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
            metric_cols = [
                f for f in fieldnames
                if f.lower() not in ignored_cols and not f.startswith(IGNORE_PREFIXES)
            ]

            for metric_name in metric_cols:
                xy_map: dict[float, float] = {}
                for idx, row in enumerate(rows):
                    y = _safe_float(row.get(metric_name))
                    if y is None:
                        continue

                    x = _safe_float(row.get("epoch"))
                    if x is None:
                        x = _safe_float(row.get("step"))
                    if x is None:
                        x = float(idx)

                    xy_map[x] = y

                if xy_map:
                    sorted_xs = sorted(xy_map.keys())
                    sorted_ys = [xy_map[k] for k in sorted_xs]

                    if metric_name not in data_by_metric:
                        data_by_metric[metric_name] = {}
                    data_by_metric[metric_name][run_name] = (sorted_xs, sorted_ys)

            break

    return data_by_metric


def _discover_run_dirs(
    root: Path,
    pilot_root: Path | None = None,
    run_name: str | None = None,
    run_dir: Path | None = None,
    latest: bool = False,
):
    pilot_root = (pilot_root or (root / "pilot_logs")).resolve()

    summary_candidates = [
        pilot_root.parent / "runs" / "sweep_summary.json",
        (root / "runs" / "sweep_summary.json").resolve(),
    ]
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
                        matches = sorted(
                            pilot_root.glob(f"{run_name}*"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if matches:
                            return [matches[0].resolve()]
            except Exception:
                pass
        return []

    if latest:
        candidates = sorted(
            [p for p in pilot_root.glob("*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
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

                matches = sorted(
                    pilot_root.glob(f"{run_name}*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
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


def _plot_individual_metrics(run_dirs, out_dir: Path, metric_filter: str | None = None):
    if not run_dirs:
        raise FileNotFoundError("No run folders found under pilot_logs/.")

    data_by_metric = _collect_all_metrics(run_dirs)

    if TARGET_METRICS:
        filtered_data = {}
        for target in TARGET_METRICS:
            target_norm = target.lower().replace("/", "_")
            matched_key = None

            for m in data_by_metric.keys():
                m_norm = m.lower().replace("/", "_")
                if m_norm == target_norm:
                    matched_key = m
                    break

            if not matched_key:
                for m in data_by_metric.keys():
                    m_norm = m.lower().replace("/", "_")
                    if m_norm in (f"{target_norm}_epoch", f"{target_norm}_step"):
                        matched_key = m
                        break

            if matched_key:
                filtered_data[matched_key] = data_by_metric[matched_key]

        data_by_metric = filtered_data

    if metric_filter:
        data_by_metric = {
            m: runs for m, runs in data_by_metric.items() if metric_filter.lower() in m.lower()
        }

    if not data_by_metric:
        print("No valid metric series found matching criteria.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, run_data in data_by_metric.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        for run_name, (xs, ys) in run_data.items():
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=run_name)

        clean_title = metric_name.replace("losses/", "").replace("metrics/", "")
        ax.set_title(clean_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("value", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="best")

        fig.tight_layout()

        file_name = (
            metric_name.lower()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            + ".png"
        )
        metric_out_path = out_dir / file_name

        fig.savefig(metric_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved metric plot: {metric_out_path}")


def _generate_comparison_csv(run_dirs: list[Path], out_dir: Path):
    """Parses static efficiency stats and final metric values into comparison.csv."""
    summary_rows = []
    ignored_time_cols = {"epoch", "step", "_step", "timestamp", "created_at"}

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_info = {"run_name": run_name}

        # 1. Safe JSON parsing (handles 0-byte or corrupt JSON files gracefully)
        for json_file in run_dir.rglob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                data = json.loads(content)
                if isinstance(data, dict):
                    flat_data = _flatten_dict(data)
                    for k, v in flat_data.items():
                        k_lower = k.lower()
                        if any(kw in k_lower for kw in EFFICIENCY_KEYWORDS):
                            run_info[k] = v
            except Exception:
                pass

        # 2. Extract logged metrics from metrics.csv, filtering out LR parameter noise
        csv_paths = sorted(
            run_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if csv_paths:
            try:
                with open(csv_paths[0], "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if rows:
                    fieldnames = [fn.strip() for fn in (reader.fieldnames or []) if fn]
                    for col in fieldnames:
                        col_lower = col.lower()

                        if col_lower in ignored_time_cols or col.startswith(IGNORE_PREFIXES):
                            continue

                        vals = [
                            _safe_float(r.get(col))
                            for r in rows
                            if r.get(col) is not None and r.get(col) != ""
                        ]
                        vals = [v for v in vals if v is not None]

                        if not vals:
                            continue

                        # Treat static efficiency metrics distinctly from epoch-varying losses
                        if any(kw in col_lower for kw in EFFICIENCY_KEYWORDS):
                            run_info[col] = vals[-1]
                        else:
                            run_info[f"{col} (final)"] = vals[-1]

            except Exception:
                pass

        summary_rows.append(run_info)

    if not summary_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = out_dir / "comparison.csv"

    # Order columns logically: Run Name -> Efficiency -> Primary Metrics -> Intermediate Block Metrics
    all_keys = set()
    for row in summary_rows:
        all_keys.update(row.keys())

    all_keys.discard("run_name")

    efficiency_keys = sorted(
        [k for k in all_keys if any(kw in k.lower() for kw in EFFICIENCY_KEYWORDS)]
    )
    primary_metrics = sorted(
        [k for k in all_keys if k not in efficiency_keys and "block_" not in k.lower()]
    )
    block_metrics = sorted([k for k in all_keys if "block_" in k.lower()])

    fieldnames = ["run_name"] + efficiency_keys + primary_metrics + block_metrics

    with open(comparison_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})

    print(f"Saved comparison CSV: {comparison_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--pilot-logs",
        type=Path,
        default=ROOT / "pilot_logs",
        help="Root folder containing run folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Custom output folder path",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Optional metric name filter",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_root = args.pilot_logs.resolve()

    if args.out is None:
        out_dir = (pilot_root.parent / "sweep_plots").resolve()
    else:
        out_dir = args.out.resolve()

    run_dirs = _discover_run_dirs(
        root,
        pilot_root=pilot_root,
        run_name=args.run_name,
        run_dir=args.run_dir,
        latest=args.latest,
    )
    if not run_dirs:
        print(f"No run directories found under {pilot_root}.")
        return 1

    _plot_individual_metrics(run_dirs, out_dir, metric_filter=args.metric)
    _generate_comparison_csv(run_dirs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())