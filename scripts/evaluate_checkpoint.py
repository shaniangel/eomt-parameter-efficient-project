"""Evaluate a saved Lightning checkpoint on the validation set and print metrics.

Usage:
    python scripts\evaluate_checkpoint.py --ckpt <path-to-ckpt> --data <path-to-data>

If --ckpt is omitted, the script will attempt to find the most-recent checkpoint under
pilot_logs/smoke_lora_run/checkpoints or lightning_logs. It instantiates the data module
and runs trainer.validate() to print metrics.
"""
import argparse
import sys
from pathlib import Path

import torch
from lightning.pytorch import Trainer

# Import project classes
from training.mask_classification_semantic import MaskClassificationSemantic
from datasets.ade20k_semantic import ADE20KSemantic


def find_latest_checkpoint(search_dirs):
    cands = []
    for d in search_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob('*.ckpt'):
            cands.append(f)
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint (.ckpt)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None or not ckpt_path.exists():
        # Look for checkpoints in common locations
        candidate = find_latest_checkpoint([
            Path('pilot_logs') / 'smoke_lora_run' / 'checkpoints',
            Path('lightning_logs'),
            Path('pilot_logs') / 'smoke_lora_run',
        ])
        if candidate is None:
            print('No checkpoint provided and none found in default locations.', file=sys.stderr)
            sys.exit(2)
        ckpt_path = candidate

    print(f'Using checkpoint: {ckpt_path}')

    # Load checkpoint (map to CPU by default)
    map_location = 'cpu'
    model = MaskClassificationSemantic.load_from_checkpoint(str(ckpt_path), map_location=map_location)

    # Create data module
    dm = ADE20KSemantic(path=args.data, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()

    # Create trainer for validation only (no logger to avoid extra output)
    trainer = Trainer(accelerator='auto', devices=1 if torch.cuda.is_available() or True else 1, logger=False)

    # Run validation
    results = trainer.validate(model, datamodule=dm)
    print('Validation results:')
    print(results)


if __name__ == '__main__':
    main()
