"""Plots side-by-side predictions of the trained regimes on the same validation images.

Each regime's model is rebuilt from the ``config.yaml`` saved in its run folder (so any
settings overridden at training time are used too) and loaded from that run's checkpoint.
The run is the one summarize_results.py picks (the latest finished run, or the folder given
with ``--run <regime>=<folder>``). Inference uses the same sliding window path as
validation. The grid (image | ground truth | one column per regime) is written to
``<out>/qualitative.png``.

Usage:
    python scripts/visualize_predictions.py [--logs logs] [--data data/ade20k] [--indices 0 10 20]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import LightningCLI  # noqa: E402
from training.progress import colorize  # noqa: E402
from scripts.summarize_results import find_run_dir, parse_run_args  # noqa: E402
from datasets.lightning_data_module import LightningDataModule  # noqa: E402
from training.lightning_module import LightningModule  # noqa: E402

def find_run(logs: Path, regime: str, chosen: dict[str, Path]) -> tuple[Path, Path]:
    run_dir = find_run_dir(logs, regime, chosen)
    if run_dir is None or not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"No finished run with a config.yaml for {regime} under {logs / regime}")
    ckpts = sorted(run_dir.glob("checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")
    return run_dir, ckpts[-1]


def build(run_dir: Path, data_path: str):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    config.pop("ckpt_path", None)  # a fit-only option, unknown without a subcommand

    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
        yaml.safe_dump(config, config_file)
        config_file.flush()
        cli = LightningCLI(
            LightningModule,
            LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=None,
            run=False,
            args=[
                "-c", config_file.name,
                "--data.init_args.path", data_path,
                "--data.init_args.num_workers", "0",
                "--trainer.logger", "false",
                "--trainer.accelerator", "auto",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--data", default="data/ade20k")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--regimes", nargs="+", default=["full", "frozen", "lora"])
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 100, 200, 300, 400, 500])
    parser.add_argument("--run", action="append", default=[], metavar="REGIME=FOLDER")
    args = parser.parse_args()
    chosen = parse_run_args(args.run)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, gts, preds = None, None, {}
    for regime in args.regimes:
        run_dir, ckpt_path = find_run(args.logs, regime, chosen)
        print(f"{regime}: {ckpt_path}")
        model, datamodule = build(run_dir, args.data)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
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
        panels = [img.permute(1, 2, 0).numpy(), colorize(gts[row])]
        panels += [colorize(preds[r][row]) for r in args.regimes]
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
