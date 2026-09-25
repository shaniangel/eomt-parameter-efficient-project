"""Progress files written to <run folder>/progress/ after every validation epoch.

- epoch_<n>.png: fixed validation images | ground truth | prediction at that epoch
- evolution.png: the same images' predictions across (up to 6) epochs so far
- curves.png:    training loss and validation mIoU per epoch
- progress.txt:  one line per epoch with loss, mIoU, elapsed time and estimated time left
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn.functional as F

IGNORE_IDX = 255
MAX_SIDE = 320  # stored and plotted images are downscaled to this size
PALETTE = np.random.default_rng(0).integers(0, 256, size=(256, 3), dtype=np.uint8)


def colorize(label_map, ignore_idx: int = IGNORE_IDX) -> np.ndarray:
    """RGB image of a class map, with a fixed color per class and black for ignored pixels."""
    label_map = np.asarray(label_map)
    rgb = PALETTE[np.clip(label_map, 0, len(PALETTE) - 1)]
    rgb[label_map == ignore_idx] = 0
    return rgb


def downscale(img: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrinks a (3, H, W) uint8 image and an (H, W) class map to at most MAX_SIDE pixels.

    Returns CPU uint8 tensors, an (h, w, 3) image and an (h, w) class map."""
    scale = min(1.0, MAX_SIDE / max(img.shape[-2:]))
    size = [max(1, round(s * scale)) for s in img.shape[-2:]]
    small_img = F.interpolate(img[None].float(), size, mode="bilinear")[0]
    small_labels = F.interpolate(labels[None, None].float(), size, mode="nearest")[0, 0]
    return (
        small_img.round().clamp(0, 255).byte().permute(1, 2, 0).cpu(),
        small_labels.to(torch.uint8).cpu(),
    )


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _save_grid(rows: list[list[np.ndarray]], titles: list[str], path: Path, suptitle: str):
    fig, axes = plt.subplots(
        len(rows), len(titles), figsize=(2.6 * len(titles), 2.2 * len(rows)), squeeze=False
    )
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            axes[r, c].imshow(panel)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_progress(out_dir: Path, samples, preds_by_epoch: dict, history: list[dict], max_epochs: int):
    """Writes all progress files. ``samples`` are (image, ground truth) pairs and
    ``preds_by_epoch`` maps each validated epoch to the predictions for those samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = history[-1]
    epochs = sorted(preds_by_epoch)

    if samples:
        # This epoch: image | ground truth | prediction
        rows = [
            [np.asarray(img), colorize(gt), colorize(pred)]
            for (img, gt), pred in zip(samples, preds_by_epoch[latest["epoch"]])
        ]
        _save_grid(
            rows,
            ["image", "ground truth", "prediction"],
            out_dir / f"epoch_{latest['epoch'] + 1:03d}.png",
            f"Epoch {latest['epoch'] + 1} · val mIoU {100 * latest['val_miou']:.1f}%",
        )

        # Evolution: the same images at up to 6 evenly spaced epochs, always incl. first and last
        shown = sorted({epochs[round(i)] for i in np.linspace(0, len(epochs) - 1, min(6, len(epochs)))})
        rows = [
            [np.asarray(img), colorize(gt)] + [colorize(preds_by_epoch[e][i]) for e in shown]
            for i, (img, gt) in enumerate(samples)
        ]
        _save_grid(
            rows,
            ["image", "ground truth"] + [f"epoch {e + 1}" for e in shown],
            out_dir / "evolution.png",
            "Predictions on fixed validation images during training",
        )

    # Curves
    x = [h["epoch"] + 1 for h in history]
    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax_loss.plot(x, [h["train_loss"] for h in history], marker="o")
    ax_loss.set(xlabel="Epoch", ylabel="Training loss (epoch mean)")
    ax_miou.plot(x, [100 * h["val_miou"] for h in history], marker="o", label="mIoU")
    ax_miou.plot(x, [100 * h["val_miou_present"] for h in history], marker=".", ls="--",
                 label="mIoU over present classes")
    ax_miou.set(xlabel="Epoch", ylabel="Validation mIoU (%)")
    ax_miou.legend(fontsize=8)
    for ax in (ax_loss, ax_miou):
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=110)
    plt.close(fig)

    # Text log
    lines = ["epoch | step   | train loss | val mIoU | mIoU present | elapsed | est. left"]
    for h in history:
        done = h["epoch"] + 1
        left = h["elapsed_sec"] / done * max(0, max_epochs - done)
        lines.append(
            f"{done:>2}/{max_epochs:<2} | {h['step']:>6} | {h['train_loss']:>10.3f} "
            f"| {100 * h['val_miou']:>7.2f}% | {100 * h['val_miou_present']:>11.2f}% "
            f"| {format_duration(h['elapsed_sec']):>7} | {format_duration(left):>9}"
        )
    (out_dir / "progress.txt").write_text("\n".join(lines) + "\n")
