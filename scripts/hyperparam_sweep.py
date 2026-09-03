r"""
Simple hyperparameter sweep runner for the EoMT project.

Usage:
    python scripts\hyperparam_sweep.py [--out runs] [--workers 1]

This script enumerates a small grid over training regimes and hyperparameters and
runs `python main.py fit ...` for each experiment. Each run's stdout/stderr is
saved to a log file under the output directory and the run directory created by
LightningCLI is preserved.

Notes:
- Uses the project's LightningCLI entrypoint (main.py). Adjust class paths or
  additional CLI args below if your local setup differs.
- Runs sequentially to avoid GPU contention. For parallel runs, run multiple
  copies of this script on separate machines/GPUs.
"""
from __future__ import annotations
import itertools
import json
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "main.py"
CONFIG_YAML = ROOT / "configs" / "pilot" / "eomt_lora_pilot.yaml"
OUT_DIR = ROOT / "runs"

# Default class paths used by the project's LightningCLI
DATA_CLASS = "datasets.ade20k_semantic.ADE20KSemantic"
MODEL_CLASS = "training.mask_classification_semantic.MaskClassificationSemantic"
NETWORK_CLASS = "models.eomt.EoMT"
ENCODER_CLASS = "models.vit.ViT"


def _cli_arg_value(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def build_cmd(run_name: str, cfg: dict) -> list[str]:
    """Return command list to spawn one training run via main.py fit."""
    cmd = ["python", str(MAIN_PY), "fit", "-c", str(CONFIG_YAML)]

    # Keep the config file as the source of truth for class paths and defaults.
    # Only override experiment-specific values here.

    trainer_overrides = cfg.get("trainer", {})
    for k, v in trainer_overrides.items():
        cmd += [f"--trainer.{k}", _cli_arg_value(v)]

    model_args = cfg.get("model_init", {})
    for k, v in model_args.items():
        cmd += [f"--model.init_args.{k}", _cli_arg_value(v)]

    data_args = cfg.get("data_init", {})
    for k, v in data_args.items():
        cmd += [f"--data.init_args.{k}", _cli_arg_value(v)]

    if cfg.get("data_path"):
        cmd += ["--data.init_args.path", _cli_arg_value(cfg["data_path"])]

    if cfg.get("compile_disabled", False):
        cmd += ["--compile_disabled"]

    cmd += ["--trainer.logger.init_args.name", run_name]
    return cmd


def main(out: str | Path, dry: bool = False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    data_path = str((ROOT / "data" / "ade20k").resolve())

    # Grid: regimes and a couple hyperparameters. Edit as needed.
    regimes = [
        {
            "name": "full-finetune",
            "model_init": {"freeze_backbone": False, "lora_enabled": False},
        },
        {
            "name": "frozen-backbone",
            "model_init": {"freeze_backbone": True, "lora_enabled": False},
        },
        {
            "name": "local-lora",
            "model_init": {"freeze_backbone": True, "lora_enabled": True},
        },
    ]

    lrs = [1e-4]
    weight_decays = [0.05]
    lora_ranks = [4]

    runs = []

    for regime in regimes:
        for lr, wd in itertools.product(lrs, weight_decays):
            if regime["name"] == "local-lora":
                for r in lora_ranks:
                    cfg = {
                        "model_init": {
                            **regime["model_init"],
                            "lora_config": {
                                "-3,-2,-1": {"rank": r, "alpha": 1.0, "modules": ["qkv", "proj"]}
                            },
                        },
                        "trainer": {"devices": 1},
                        "data_init": {
                            "path": data_path,
                            "img_size": [512, 512],
                        },
                    }
                    run_name = f"{regime['name']}_lr{lr}_wd{wd}_r{r}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    cfg["model_init"]["lr"] = lr
                    cfg["model_init"]["weight_decay"] = wd
                    runs.append((run_name, cfg))
            else:
                cfg = {
                    "model_init": {**regime["model_init"]},
                    "trainer": {"devices": 1},
                    "data_init": {
                        "path": data_path,
                        "img_size": [512, 512],
                    },
                }
                run_name = f"{regime['name']}_lr{lr}_wd{wd}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                cfg["model_init"]["lr"] = lr
                cfg["model_init"]["weight_decay"] = wd
                runs.append((run_name, cfg))

    summary = []

    for run_name, cfg in runs:
        print(f"Starting run: {run_name}")
        cmd = build_cmd(run_name, cfg)
        print(" ", " ".join(cmd))

        log_file = out / f"{run_name}.log"
        if dry:
            print(f"Dry run: log would go to {log_file}")
            continue

        with open(log_file, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            ret = proc.wait()

        # try to pick up validation metrics written by main.py into the run dir
        # LightningCLI creates a run dir under the logger save_dir; we attempt to
        # find the most recent run folder and copy its validation_metrics.json
        run_dirs = sorted([d for d in (ROOT / "pilot_logs").glob("**/*") if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        metrics = None
        if run_dirs:
            candidate = run_dirs[0]
            metrics_path = candidate / "validation_metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None

        summary.append({"run_name": run_name, "cmd": cmd, "return_code": ret, "metrics": metrics, "log": str(log_file)})

        # flush summary to disk after each run
        summary_path = out / "sweep_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Finished run {run_name} (rc={ret}). Log: {log_file}")

    print("Sweep completed. Summary written to", out / "sweep_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.out, dry=args.dry)
