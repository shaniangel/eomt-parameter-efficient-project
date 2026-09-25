"""Checks that the smoke runs learned, and updated exactly the weights they should.

For each regime:
  - the training loss went down from the first to the last epoch;
  - the validation mIoU went up from the first to the last epoch;
  - compared with the pretrained DINOv2 weights the run started from:
      full:   the backbone weights changed
      frozen: the backbone weights are identical to the pretrained ones
      lora:   the pretrained backbone weights are identical, and every LoRA ``lora_B``
              (which starts at zero) is non-zero, i.e. the adapters learned

Run it on the smoke runs (the default), whose short warmup gives the backbone and LoRA a
non-zero learning rate. With the paper's warmup the backbone learning rate stays at 0 for
the first 500 steps, so a short run would fail the full and lora checks. The predictions
on fixed validation images per epoch are in each run's progress/evolution.png.

Usage:
    python scripts/check_smoke.py [--logs logs_smoke] [--run lora=logs_smoke/lora/<folder>]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import timm
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_results import find_run_dir, parse_run_args  # noqa: E402

BACKBONE = "network.encoder.backbone."


def pretrained_backbone(config: dict) -> dict[str, torch.Tensor]:
    # Created exactly as models/vit.py does, so resized weights match the trained model
    encoder = config["model"]["init_args"]["network"]["init_args"]["encoder"]["init_args"]
    return timm.create_model(
        encoder.get("backbone_name", "vit_large_patch14_reg4_dinov2"),
        pretrained=True,
        img_size=tuple(config["data"]["init_args"]["img_size"]),
        patch_size=encoder.get("patch_size", 16),
        num_classes=0,
    ).state_dict()


def trained_key(name: str, state_dict: dict) -> str:
    """Key of a pretrained weight in the checkpoint; LoRA moves it to <layer>.base.<param>."""
    key = BACKBONE + name
    if key not in state_dict:
        layer, param = key.rsplit(".", 1)
        key = f"{layer}.base.{param}"
    return key


def learning_curves(metrics_csv: Path) -> tuple[list[float], list[float]]:
    """Mean training loss per epoch, and validation mIoU per epoch."""
    losses, mious = defaultdict(list), {}
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("losses/train_loss_total"):
                losses[int(row["epoch"])].append(float(row["losses/train_loss_total"]))
            if row.get("metrics/val_iou_all"):
                mious[int(row["epoch"])] = float(row["metrics/val_iou_all"])
    return (
        [sum(losses[e]) / len(losses[e]) for e in sorted(losses)],
        [mious[e] for e in sorted(mious)],
    )


def check_run(regime: str, run_dir: Path) -> bool:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt = sorted(run_dir.glob("checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    reference = pretrained_backbone(config)

    changed = [
        name for name, weight in reference.items()
        if not torch.equal(state_dict[trained_key(name, state_dict)], weight)
    ]
    lora_b = {k: v for k, v in state_dict.items() if k.endswith("lora_B")}
    zero_lora_b = [k for k, v in lora_b.items() if not v.any()]

    losses, mious = learning_curves(run_dir / "metrics.csv")

    print(f"{regime}: {run_dir}")
    print("  training loss per epoch: " + ", ".join(f"{v:.2f}" for v in losses))
    print("  val mIoU per epoch:      " + ", ".join(f"{100 * v:.2f}%" for v in mious))
    print(f"  pretrained backbone tensors changed: {len(changed)} of {len(reference)}")
    if lora_b:
        print(f"  LoRA lora_B tensors still zero: {len(zero_lora_b)} of {len(lora_b)}")

    checks = {
        "full": [("backbone weights changed", len(changed) > 0)],
        "frozen": [("backbone weights unchanged", not changed)],
        "lora": [
            ("pretrained backbone weights unchanged", not changed),
            ("LoRA adapters present", len(lora_b) > 0),
            ("LoRA adapters learned (lora_B non-zero)", len(lora_b) > 0 and not zero_lora_b),
        ],
    }[regime]
    checks = [
        ("training loss decreased", len(losses) >= 2 and losses[-1] < losses[0]),
        ("validation mIoU increased", len(mious) >= 2 and mious[-1] > mious[0]),
    ] + checks
    for description, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")
    return all(passed for _, passed in checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs", type=Path, default=Path("logs_smoke"))
    parser.add_argument("--regimes", nargs="+", default=["full", "frozen", "lora"])
    parser.add_argument("--run", action="append", default=[], metavar="REGIME=FOLDER")
    args = parser.parse_args()
    chosen = parse_run_args(args.run)

    all_passed = True
    for regime in args.regimes:
        run_dir = find_run_dir(args.logs, regime, chosen)
        if run_dir is None:
            print(f"{regime}: no finished run found under {args.logs / regime}, skipped")
            continue
        all_passed &= check_run(regime, run_dir)
        print(f"  predictions over the epochs: {run_dir / 'progress' / 'evolution.png'}")

    print("All checks passed" if all_passed else "Some checks FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
