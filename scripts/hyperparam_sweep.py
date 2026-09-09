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
import shutil

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


def build_cmd(run_name: str, cfg: dict, pilot_logs: str | None = None) -> list[str]:
    """Return command list to spawn one training run via main.py fit.

    If pilot_logs is provided, instruct the trainer logger to write to that folder
    so each sweep is self-contained.
    """
    cmd = ["python", str(MAIN_PY), "fit", "-c", str(CONFIG_YAML)]

    # Keep the config file as the source of truth for class paths and defaults.
    # Only override experiment-specific values here.

    trainer_overrides = cfg.get("trainer", {})
    for k, v in trainer_overrides.items():
        cmd += [f"--trainer.{k}", _cli_arg_value(v)]

    # If a pilot_logs path is given, make sure the trainer's logger writes there.
    if pilot_logs:
        cmd += ["--trainer.logger.init_args.save_dir", _cli_arg_value(pilot_logs)]

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

    # Use trainer logger name for run identification
    cmd += ["--trainer.logger.init_args.name", run_name]
    return cmd


def main(out: str | Path, dry: bool = False, sweep_name: str | None = None, prewarm: bool = False, prewarm_epochs: int = 1, prewarm_limit_train_batches: int = 1):
    # If a sweep_name is provided, create a sweeps/<sweep_name> layout and
    # store pilot_logs and run summaries there. Otherwise use the provided out dir.
    if sweep_name:
        base = (ROOT / "sweeps" / sweep_name).resolve()
        pilot_logs_root = base / "pilot_logs"
        out = base / "runs"
        pilot_logs_root.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        # Copy the base config into the sweep folder for reproducibility
        try:
            shutil.copy(CONFIG_YAML, out / CONFIG_YAML.name)
        except Exception:
            pass
    else:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        pilot_logs_root = (ROOT / "pilot_logs").resolve()

    data_path = str((ROOT / "data" / "ade20k").resolve())

    # Optional pre-warm: run one short init job to produce a checkpoint to reuse
    prewarm_ckpt = None
    if prewarm:
        print("Running pre-warm job to produce a shared checkpoint...")
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        prewarm_name = f"prewarm_{timestamp}"
        prewarm_cfg = {
            "trainer": {
                "devices": 1,
                "max_epochs": prewarm_epochs,
                "limit_train_batches": prewarm_limit_train_batches,
                "limit_val_batches": 1,
            },
            "model_init": {},
            "data_init": {"path": data_path, "img_size": [256, 256]},
        }

        prewarm_log = out / f"{prewarm_name}.log"
        before_dirs = {p.resolve() for p in pilot_logs_root.glob("**/*") if p.is_dir()}
        cmd = build_cmd(prewarm_name, prewarm_cfg, pilot_logs=str(pilot_logs_root))
        print(" ", " ".join(cmd))
        if not dry:
            with open(prewarm_log, "wb") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
                ret = proc.wait()

            after_dirs = sorted([p.resolve() for p in pilot_logs_root.glob("**/*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            new_dirs = [d for d in after_dirs if d not in before_dirs]
            candidate = new_dirs[0] if new_dirs else (after_dirs[0] if after_dirs else None)
            if candidate is not None:
                # search for checkpoint files under the run dir
                ckpt_paths = sorted(list(candidate.glob("**/*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
                if not ckpt_paths:
                    # common checkpoint subfolders
                    ckpt_paths = sorted(list(candidate.glob("**/checkpoints/*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpt_paths:
                    prewarm_ckpt = str(ckpt_paths[0])
                    print(f"Found prewarm checkpoint: {prewarm_ckpt}")
                else:
                    print("Prewarm completed but no checkpoint found; subsequent runs will not reuse a prewarm checkpoint.")
            else:
                print("Prewarm completed but could not locate the created run directory.")
        else:
            print(f"Dry run mode: prewarm command would run and log to {prewarm_log}")

    # LoRA Structural Exploration Space
    # Uniform rank per run to keep search space feasible
    '''lora_ranks = [8, 32]

    # Testing depth variations (Local vs Mid-to-Deep vs Target Block)
    target_layer_sets = {
        "last1": "-1",  # Very local (final query layer)
        "last3": "-3,-2,-1",  # Standard local baseline
        "last6": "-6,-5,-4,-3,-2,-1",  # Deep adaptation
    }

    # Regime-tailored grids (Coarse screening settings)
    regime_configs = [
        {
            "name": "full-finetune",
            "model_init": {"freeze_backbone": False, "lora_enabled": False},
            "lrs": [1e-4, 3e-4],
            "weight_decays": [0.05],
        },
        {
            "name": "frozen-backbone",
            "model_init": {"freeze_backbone": True, "lora_enabled": False},
            "lrs": [1e-3, 5e-3],
            "weight_decays": [0.01],
        },
        {
            "name": "local-lora",
            "model_init": {"freeze_backbone": True, "lora_enabled": True},
            "lrs": [5e-4, 1e-3],
            "weight_decays": [0.01],
        },
    ]'''

    lora_ranks = [32]

    # Testing depth variations (Local vs Mid-to-Deep vs Target Block)
    target_layer_sets = {
        "last1": "-1",  # Very local (final query layer)
    }

    # Regime-tailored grids (Coarse screening settings)
    regime_configs = [
        {
            "name": "full-finetune",
            "model_init": {"freeze_backbone": False, "lora_enabled": False},
            "lrs": [1e-3],
            "weight_decays": [0.01],
        },
        {
            "name": "frozen-backbone",
            "model_init": {"freeze_backbone": True, "lora_enabled": False},
            "lrs": [1e-3],
            "weight_decays": [0.01],
        },
        {
            "name": "local-lora",
            "model_init": {"freeze_backbone": True, "lora_enabled": True},
            "lrs": [1e-3],
            "weight_decays": [0.01],
        },
    ]

    runs = []
    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')

    for regime in regime_configs:
        regime_name = regime["name"]

        for lr, wd in itertools.product(regime["lrs"], regime["weight_decays"]):
            if regime_name == "local-lora":
                for r, (layer_tag, layers) in itertools.product(lora_ranks, target_layer_sets.items()):
                    run_name = f"{regime_name}_lr{lr}_wd{wd}_r{r}_{layer_tag}_{timestamp_str}"

                    cfg = {
                        "model_init": {
                            **regime["model_init"],
                            "lr": lr,
                            "weight_decay": wd,
                            "lora_config": {
                                layers: {"rank": r, "alpha": r, "modules": ["qkv", "proj"]}
                            },
                        },
                        "trainer": {"devices": 1},
                        "data_init": {
                            "path": data_path,
                            "img_size": [512, 512],
                        },
                    }
                    runs.append((run_name, cfg))
            else:
                run_name = f"{regime_name}_lr{lr}_wd{wd}_{timestamp_str}"
                cfg = {
                    "model_init": {
                        **regime["model_init"],
                        "lr": lr,
                        "weight_decay": wd,
                    },
                    "trainer": {"devices": 1},
                    "data_init": {
                        "path": data_path,
                        "img_size": [512, 512],
                    },
                }
                runs.append((run_name, cfg))

    summary = []

    for run_name, cfg in runs:
        # If prewarm produced a checkpoint, reuse it for this run to avoid repeated heavy initialization
        if prewarm and prewarm_ckpt:
            cfg.setdefault("model_init", {})["ckpt_path"] = prewarm_ckpt
        print(f"Starting run: {run_name}")
        # pass pilot_logs_root so each run writes under the sweep's pilot_logs folder
        cmd = build_cmd(run_name, cfg, pilot_logs=str(pilot_logs_root))
        print(" ", " ".join(cmd))

        log_file = out / f"{run_name}.log"
        if dry:
            print(f"Dry run: log would go to {log_file}")
            continue

        # snapshot existing pilot_logs dirs so we can identify which new folder
        # was created by this run (avoids racing with other concurrent runs)
        before_dirs = {p.resolve() for p in pilot_logs_root.glob("**/*") if p.is_dir()}

        with open(log_file, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            ret = proc.wait()

        after_dirs = sorted([p.resolve() for p in pilot_logs_root.glob("**/*") if p.is_dir()],
                            key=lambda p: p.stat().st_mtime, reverse=True)
        new_dirs = [d for d in after_dirs if d not in before_dirs]

        metrics = None
        candidate = new_dirs[0] if new_dirs else (after_dirs[0] if after_dirs else None)

        if candidate is not None:
            metrics_path = candidate / "validation_metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None

        summary.append({
            "run_name": run_name,
            "cmd": cmd,
            "return_code": ret,
            "metrics": metrics,
            "log": str(log_file),
            "run_dir": str(candidate) if candidate is not None else None
        })

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
    parser.add_argument("--sweep-name", default=None, help="Optional sweep name to create sweeps/<name>/pilot_logs and sweeps/<name>/runs")
    parser.add_argument("--prewarm", action="store_true", help="Run a short pre-warm job to create a shared checkpoint before the sweep")
    parser.add_argument("--prewarm-epochs", type=int, default=1, help="Number of epochs for the pre-warm job")
    parser.add_argument("--prewarm-limit-train-batches", type=int, default=1, help="Limit train batches for the pre-warm job (smoke)")
    args = parser.parse_args()
    main(args.out, dry=args.dry, sweep_name=args.sweep_name, prewarm=args.prewarm, prewarm_epochs=args.prewarm_epochs, prewarm_limit_train_batches=args.prewarm_limit_train_batches)
