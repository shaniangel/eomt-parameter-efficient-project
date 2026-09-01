"""Experiment: Full fine-tuning baseline for EoMT.

Builds an EoMT model with a ViT backbone and runs a short smoke training
session with a synthetic dataset to verify end-to-end training works when the
backbone is trainable.

Run:
    python scripts/experiment_full_finetune.py

By default this performs a fast_dev_run (one train + one val batch). For full
training, pass appropriate Trainer args or remove fast_dev_run.
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import random

# allow running as a script from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic
from datasets.ade20k_semantic import ADE20KSemantic
import lightning


class SyntheticSemanticDataset(Dataset):
    """Produces synthetic images and simple per-image semantic targets.

    Each sample returns (img_tensor, target_dict) where target_dict is a dict
    with keys 'masks' (Tensor[num_objects, H, W], bool) and 'labels'
    (Tensor[num_objects]). This matches what MaskClassificationLoss expects.
    """

    def __init__(self, length=8, img_size=(128, 128), num_classes=150, max_objects=3):
        self.length = length
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        C = 3
        H, W = self.img_size
        # images are expected in 0..255 (LightningModule divides by 255 in forward)
        img = torch.randint(0, 256, (C, H, W), dtype=torch.uint8)

        num_objs = random.randint(1, self.max_objects)
        masks = torch.zeros((num_objs, H, W), dtype=torch.bool)
        labels = torch.zeros((num_objs,), dtype=torch.long)

        for i in range(num_objs):
            # simple random rectangle mask
            y0 = random.randint(0, H - 1)
            x0 = random.randint(0, W - 1)
            y1 = random.randint(y0, H - 1)
            x1 = random.randint(x0, W - 1)
            masks[i, y0:y1+1, x0:x1+1] = True
            labels[i] = random.randint(0, self.num_classes - 1)

        target = {"masks": masks, "labels": labels}
        return img, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


def build_model(img_size=(128, 128), backbone_name="vit_base_patch16_224", num_classes=150, num_q=100, num_blocks=4):
    vit = ViT(img_size=img_size, patch_size=16, backbone_name=backbone_name, ckpt_path="no_pretrained")
    model = EoMT(encoder=vit, num_classes=num_classes, num_q=num_q, num_blocks=num_blocks)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fast-dev-run", dest="fast_dev_run", action="store_true")
    parser.add_argument("--no-fast-dev-run", dest="fast_dev_run", action="store_false")
    parser.set_defaults(fast_dev_run=True)
    parser.add_argument("--dataset", type=str, choices=["synthetic", "ade20k"], default="synthetic")
    parser.add_argument("--data-root", type=str, default=None, help="Path to ADE20K root (contains ADEChallengeData2016.zip)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    img_size = tuple(args.img_size)

    print("Building model (full fine-tuning)...")
    model = build_model(img_size=img_size, backbone_name=args.backbone, num_classes=args.num_classes)

    # Ensure backbone is trainable (default) — explicit set for clarity
    for p in model.encoder.backbone.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total:,}, trainable={trainable:,} ({100.0*trainable/total:.2f}%)")

    # LightningModule wrapper
    lm = MaskClassificationSemantic(
        network=model,
        img_size=img_size,
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=False,
    )

    # Dataloaders: ADE20K or synthetic for debugging
    if args.dataset == "ade20k":
        if args.data_root is None:
            raise ValueError("--data-root must be set when --dataset ade20k is used")
        dm = ADE20KSemantic(path=args.data_root, num_workers=args.num_workers, batch_size=args.batch_size, img_size=img_size, num_classes=args.num_classes)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    else:
        # Synthetic dataloaders
        train_ds = SyntheticSemanticDataset(length=8, img_size=img_size, num_classes=args.num_classes)
        val_ds = SyntheticSemanticDataset(length=4, img_size=img_size, num_classes=args.num_classes)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Trainer — use fast_dev_run for a quick smoke test
    # Use a real WandB logger in offline mode so that `trainer.logger.experiment`
    # supports `.log(...)` used by the project's plotting utilities.
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(project="eomt_experiment", mode="offline", save_dir=str(PROJECT_ROOT / "logs"))
    trainer = lightning.Trainer(fast_dev_run=args.fast_dev_run, logger=wandb_logger, enable_checkpointing=False)

    print("Starting Trainer.fit (this will run a short smoke run if fast_dev_run=True)")
    trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Trainer finished")


if __name__ == "__main__":
    main()

