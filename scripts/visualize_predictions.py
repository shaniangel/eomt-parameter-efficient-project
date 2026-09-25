"""Plots side-by-side predictions of the trained regimes on the same validation images.

Each regime's model is rebuilt from the project configs and loaded from the latest
checkpoint in ``<logs>/<regime>/version_*/checkpoints``. Inference uses the same sliding
window path as validation. The grid (image | ground truth | one column per regime) is
written to ``<out>/qualitative.png``.

Usage:
    python scripts/visualize_predictions.py [--logs logs] [--data data/ade20k] [--indices 0 10 20]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import LightningCLI  # noqa: E402
from datasets.lightning_data_module import LightningDataModule  # noqa: E402
from training.lightning_module import LightningModule  # noqa: E402

BASE_CONFIG = ROOT / "configs" / "project" / "base_ade20k_eomt_small_512.yaml"


def latest_checkpoint(logs: Path, regime: str) -> Path:
    ckpts = sorted((logs / regime).glob("version_*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {logs / regime}")
    return ckpts[-1]


def build(regime: str, data_path: str):
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=[
            "-c", str(BASE_CONFIG),
            "-c", str(ROOT / "configs" / "project" / f"{regime}.yaml"),
            "--data.init_args.path", data_path,
            "--data.init_args.num_workers", "0",
            "--trainer.logger", "false",
        ],
    )
    return cli.model, cli.datamodule


@torch.no_grad()
def predict(model, img: torch.Tensor) -> np.ndarray:
    crops, origins = model.window_imgs_semantic([img])
    mask_logits_per_layer, class_logits_per_layer = model(crops)
    mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
    crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
    logits = model.revert_window_logits_semantic(crop_logits, origins, [img.shape[-2:]])
    return logits[0].argmax(0).cpu().numpy()


def colorize(label_map: np.ndarray, palette: np.ndarray, ignore_idx: int) -> np.ndarray:
    rgb = palette[np.clip(label_map, 0, len(palette) - 1)]
    rgb[label_map == ignore_idx] = 0
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--data", default="data/ade20k")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--regimes", nargs="+", default=["full", "frozen", "lora"])
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 100, 200, 300, 400, 500])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette = np.random.default_rng(0).integers(0, 256, size=(256, 3), dtype=np.uint8)

    samples, gts, preds = None, None, {}
    for regime in args.regimes:
        ckpt_path = latest_checkpoint(args.logs, regime)
        print(f"{regime}: {ckpt_path}")
        model, datamodule = build(regime, args.data)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device).eval()

        if samples is None:
            datamodule.setup()
            dataset = datamodule.val_dataset
            indices = [i for i in args.indices if i < len(dataset)]
            samples = [dataset[i] for i in indices]
            gts = [
                model.to_per_pixel_targets_semantic([target], model.ignore_idx)[0].numpy()
                for _, target in samples
            ]

        preds[regime] = [predict(model, img.to(device)) for img, _ in samples]
        del model

    columns = ["image", "ground truth"] + args.regimes
    fig, axes = plt.subplots(
        len(samples), len(columns), figsize=(3 * len(columns), 3 * len(samples)), squeeze=False
    )
    for row, (img, _) in enumerate(samples):
        panels = [img.permute(1, 2, 0).numpy(), colorize(gts[row], palette, 255)]
        panels += [colorize(preds[r][row], palette, 255) for r in args.regimes]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(columns[col])

    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "qualitative.png", dpi=150)
    print(f"Wrote {args.out / 'qualitative.png'}")


if __name__ == "__main__":
    main()
