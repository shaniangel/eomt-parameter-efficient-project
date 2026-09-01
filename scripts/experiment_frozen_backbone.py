"""Experiment: Frozen backbone baseline for EoMT.

This script builds an EoMT model with a ViT backbone, freezes the backbone
parameters, and prints parameter counts. It also prepares a LightningModule
ready to be trained where only the segmentation queries and heads are
trainable.

This is a lightweight helper — to actually run training use the project's
training harness (create a Trainer and pass an instance of
`training.mask_classification_semantic.MaskClassificationSemantic`).

Run locally for a quick sanity check:
    python scripts/experiment_frozen_backbone.py

"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import the package modules when
# running this as a script: `python scripts/experiment_frozen_backbone.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from typing import Tuple

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic
from datasets.ade20k_semantic import ADE20KSemantic
from torch.utils.data import DataLoader


def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: EoMT):
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False


def print_trainable_modules(model: EoMT):
    print("Trainable parameter groups (name, shape, requires_grad):")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name} | {tuple(p.shape)} | {p.numel()} params")


def build_model(img_size=(128, 128), backbone_name="vit_base_patch16_224", num_classes=150, num_q=100, num_blocks=4):
    # Pass a non-None ckpt_path to avoid downloading pretrained weights in timm
    vit = ViT(img_size=img_size, patch_size=16, backbone_name=backbone_name, ckpt_path="no_pretrained")
    model = EoMT(encoder=vit, num_classes=num_classes, num_q=num_q, num_blocks=num_blocks)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--num-q", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dataset", type=str, choices=["synthetic", "ade20k"], default="synthetic")
    parser.add_argument("--data-root", type=str, default=None, help="Path to ADE20K root (contains ADEChallengeData2016.zip)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print("Building model...")
    model = build_model(img_size=tuple(args.img_size), backbone_name=args.backbone, num_classes=args.num_classes, num_q=args.num_q, num_blocks=args.num_blocks)

    total, trainable = count_params(model)
    print(f"Before freezing: total params={total:,}, trainable params={trainable:,} ({100.0*trainable/total:.4f}%)")

    # Freeze backbone
    freeze_backbone(model)

    total, trainable = count_params(model)
    print(f"After freezing backbone: total params={total:,}, trainable params={trainable:,} ({100.0*trainable/total:.4f}%)")

    print_trainable_modules(model)

    # Build a LightningModule wrapper (no training here) to show optimizer groups
    lm = MaskClassificationSemantic(
        network=model,
        img_size=tuple(args.img_size),
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=False,
    )

    # If ADE20K requested, create dataloaders and show an example batch shape
    if args.dataset == "ade20k":
        if args.data_root is None:
            raise ValueError("--data-root must be set when --dataset ade20k is used")
        dm = ADE20KSemantic(path=args.data_root, num_workers=args.num_workers, batch_size=2, img_size=tuple(args.img_size), num_classes=args.num_classes)
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        imgs, targets = batch
        print(f"Example ADE20K batch: imgs.shape={imgs.shape}, num_targets={len(targets)}")

    # Try to configure optimizers to show parameter grouping logic. This
    # requires the LightningModule to be attached to a Trainer; when running
    # this script standalone (no Trainer) configure_optimizers will raise —
    # catch that and skip.
    try:
        opt_dict = lm.configure_optimizers()
        optim = opt_dict["optimizer"]
        print(f"Configured optimizer with {len(optim.param_groups)} parameter groups")
    except Exception as e:
        print(f"configure_optimizers skipped: {e}")

    print("Frozen-backbone experiment setup complete. To run training, create a Trainer and fit the LightningModule instance `lm`.")


if __name__ == "__main__":
    main()

