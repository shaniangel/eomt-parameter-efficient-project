"""Unified experiment runner for EoMT experiments.

Usage examples:
  # Synthetic smoke run (fast)
  python scripts/experiment_runner.py --mode full --dataset synthetic

  # ADE20K run (requires ADEChallengeData2016.zip under DATA_ROOT)
  python scripts/experiment_runner.py --mode lora --dataset ade20k --data-root C:\data\ade20k --no-fast-dev-run --batch-size 4

The runner supports three modes: full, frozen, lora. It will build the model,
optionally apply LoRA, freeze backbone when needed, create dataloaders, wrap
the model in the project's LightningModule and run training with deterministic
logging and checkpointing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import time
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from models.vit import ViT
from models.eomt import EoMT
from models.lora import apply_lora_to_backbone
from training.mask_classification_semantic import MaskClassificationSemantic
from datasets.ade20k_semantic import ADE20KSemantic


class SyntheticSemanticDataset(Dataset):
    def __init__(self, length=32, img_size=(128, 128), num_classes=150, max_objects=3):
        self.length = length
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        C = 3
        H, W = self.img_size
        img = torch.randint(0, 256, (C, H, W), dtype=torch.uint8)

        num_objs = 1
        masks = torch.zeros((num_objs, H, W), dtype=torch.bool)
        labels = torch.zeros((num_objs,), dtype=torch.long)
        masks[0, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3] = True
        labels[0] = 0

        target = {"masks": masks, "labels": labels}
        return img, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: EoMT):
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False


def build_model(img_size: Tuple[int, int], backbone_name: str, use_pretrained: bool, num_classes: int, num_q: int, num_blocks: int) -> EoMT:
    ckpt_path = None if use_pretrained else "no_pretrained"
    vit = ViT(img_size=img_size, patch_size=16, backbone_name=backbone_name, ckpt_path=ckpt_path)
    model = EoMT(encoder=vit, num_classes=num_classes, num_q=num_q, num_blocks=num_blocks)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "frozen", "lora"], required=True)
    parser.add_argument("--dataset", choices=["synthetic", "ade20k"], default="synthetic")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    parser.add_argument("--use-pretrained", action="store_true", default=False, help="Use pretrained backbone weights if available")
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--num-q", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--fast-dev-run", dest="fast_dev_run", action="store_true")
    parser.add_argument("--no-fast-dev-run", dest="fast_dev_run", action="store_false")
    parser.set_defaults(fast_dev_run=True)

    # LoRA options
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-num-blocks", type=int, default=4)
    parser.add_argument("--lora-target", choices=["q","qkv","proj"], default="qkv", help="Which projection to adapt with LoRA (q: query only, qkv: full qkv, proj: output projection)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")

    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"{args.mode}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility: set seeds if provided
    if args.seed is not None:
        import random as _random
        import numpy as _np

        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.backends.cudnn.deterministic = True

    print(f"Building model (mode={args.mode})...")
    model = build_model(tuple(args.img_size), args.backbone, args.use_pretrained, args.num_classes, args.num_q, args.num_blocks)

    # Apply regime-specific modifications
    if args.mode == "frozen":
        freeze_backbone(model)
    elif args.mode == "lora":
        # Freeze backbone then apply LoRA adapters to final blocks
        freeze_backbone(model)
        lora_stats = apply_lora_to_backbone(
            model.encoder.backbone,
            num_blocks=args.lora_num_blocks,
            r=args.lora_r,
            alpha=args.lora_alpha,
            target=args.lora_target,
        )
        print("LoRA applied:", lora_stats)

    total, trainable = count_params(model)
    print(f"Model params: total={total:,}, trainable={trainable:,} ({100.0*trainable/total:.4f}%)")

    # Data
    if args.dataset == "ade20k":
        if args.data_root is None:
            raise ValueError("--data-root must be provided for ADE20K dataset")
        dm = ADE20KSemantic(path=args.data_root, num_workers=args.num_workers, batch_size=args.batch_size, img_size=tuple(args.img_size), num_classes=args.num_classes)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    else:
        train_ds = SyntheticSemanticDataset(length=64, img_size=tuple(args.img_size), num_classes=args.num_classes)
        val_ds = SyntheticSemanticDataset(length=16, img_size=tuple(args.img_size), num_classes=args.num_classes)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    lm = MaskClassificationSemantic(network=model, img_size=tuple(args.img_size), num_classes=args.num_classes, attn_mask_annealing_enabled=False)

    # Logging and checkpointing
    logger = WandbLogger(project="eomt_experiments", mode="offline", save_dir=str(out_dir))
    # Use save_top_k=0 (no top-k monitoring) and keep the last checkpoint.
    ckpt_cb = ModelCheckpoint(dirpath=str(out_dir), filename="checkpoint-{epoch}", save_top_k=0, save_last=True)

    trainer = lightning.Trainer(logger=logger, callbacks=[ckpt_cb], max_epochs=args.max_epochs, fast_dev_run=args.fast_dev_run)

    # Run training
    print("Starting training...")
    trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from)

    # Save run summary
    total, trainable = count_params(model)
    # Attempt to get git commit
    try:
        import subprocess

        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None

    summary = {
        "mode": args.mode,
        "backbone": args.backbone,
        "use_pretrained": args.use_pretrained,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percentage": 100.0 * trainable / total if total > 0 else 0.0,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "git_commit": git_commit,
        "cmdline": " ".join(sys.argv),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Run finished. Summary written to", out_dir / "summary.json")


if __name__ == "__main__":
    main()

