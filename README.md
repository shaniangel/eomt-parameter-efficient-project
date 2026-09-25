# Parameter-Efficient Optimization of Encoder-Only Vision Transformers for Segmentation

*Bar Muller, Shani Angel, Guy Shloush, Yehonatan Pelleg, Daniel Ishai Aharonovitz*

This project builds on the official EoMT implementation (the original README follows below).
EoMT simplifies the segmentation architecture but still fine-tunes the whole ViT backbone. We ask
whether that is necessary, by training EoMT-S (DINOv2 ViT-S/14) on ADE20K semantic segmentation
at 512×512 under three regimes that share the same pretrained initialization, data, schedule and
evaluation:

| Regime | Trained parameters | Config |
|---|---|---|
| Full fine-tuning | backbone, queries, heads | `configs/project/full.yaml` |
| Frozen backbone | queries, heads | `configs/project/frozen.yaml` |
| Local LoRA | queries, heads, rank-8 LoRA on `qkv`/`proj` of the last `num_blocks` (= 3) blocks, where the queries are processed | `configs/project/lora.yaml` |

We report mIoU, the number and share of trainable parameters, training time, peak GPU memory,
convergence curves and qualitative examples. [IMPLEMENTATION.md](IMPLEMENTATION.md) explains how
the code works.

### What we added to the upstream code

- `models/lora.py`: `LoRALinear` and `apply_lora`, which wrap the attention projections of chosen ViT blocks.
- `training/mask_classification_semantic.py`: `freeze_backbone`, `lora_rank`, `lora_alpha` and `lora_modules` options.
- `training/lightning_module.py`:
  - writes `efficiency_stats.json` per run (parameter counts, GFLOPs, training time, peak memory);
  - derives the mask-annealing steps from the schedule length when they are not given;
  - logs a present-classes mIoU (`metrics/val_iou_present`) next to the standard one;
  - saves validation prediction plots as PNGs.
- `training/csv_logger.py`: names each run folder after its start time.
- `configs/project/`: a shared base config plus one overlay per regime and a `smoke.yaml` overlay.
- `scripts/`: `run_experiments.sh`, `summarize_results.py`, `visualize_predictions.py`.
- `tests/test_lora.py`: CPU unit tests for the LoRA wrapper and freezing.

### Reproducing

Install the requirements as in [Installation](#installation), then place `ADEChallengeData2016.zip`
in `data/ade20k/` (see [Data preparation](#data-preparation), no unzipping needed).

```bash
python -m pytest tests                              # unit tests (CPU)
bash scripts/run_experiments.sh --smoke             # quick check of all three regimes
bash scripts/run_experiments.sh                     # full runs of all three regimes, one after another
python scripts/summarize_results.py                 # results/results.md, results/results.csv, curves
python scripts/visualize_predictions.py             # results/qualitative.png
```

With two machines, split the regimes: run `bash scripts/run_experiments.sh full` on one and
`bash scripts/run_experiments.sh frozen lora` on the other. Then copy the `logs/<regime>/` folders
into `logs/` on one machine and run the two result scripts there.

Extra arguments go to `main.py`, e.g. `--data.path /path/to/ade20k` or `--trainer.max_epochs 8`.
A single regime can also be trained with:

```bash
python main.py fit -c configs/project/base_ade20k_eomt_small_512.yaml -c configs/project/lora.yaml
```

Each run writes to its own folder, named after its start time, e.g. `logs/lora/2026-09-26_14-03-12/`.
The folder holds `metrics.csv`, `efficiency_stats.json`, `checkpoints/` and `predictions/`. The
console output goes to `logs/<regime>_<start time>.out`. The result scripts use the latest
*finished* run of each regime; pick another with `--run lora=logs/lora/<folder>`.

To re-evaluate a checkpoint:

```bash
python main.py validate -c configs/project/base_ade20k_eomt_small_512.yaml -c configs/project/lora.yaml --ckpt_path logs/lora/<folder>/checkpoints/<file>.ckpt
```

### Choosing the batch size and number of epochs

1. Run the smoke test. It trains each regime for 20 steps with the real batch size and image
   size, so it shows whether memory fits and how fast training is.
2. **Batch size:** we keep the paper's batch size of 16. If the smoke test runs out of GPU memory,
   add `--data.init_args.batch_size 8 --trainer.accumulate_grad_batches 2`. This keeps the
   effective batch at 16, so the learning rate and schedule stay the same.
3. **Epochs:** read `sec_per_step` from `logs/full/<folder>/efficiency_stats.json`. One epoch is
   20,210 / 16 ≈ 1,263 steps, so `max_epochs ≈ hours available × 3600 / (sec_per_step × 1263)`.
   Use about 90% of that to leave room for validation. Use the same value for all three regimes
   (full fine-tuning is the slowest), and set it in the base config or with `--trainer.max_epochs`.

### Setup and deviations from the paper

- **Same as the paper:** AdamW with lr 1e-4, layer-wise lr decay 0.8, weight decay 0.05, poly decay
  0.9, two-stage warmup (500 steps for the new parameters, then 1000 for the backbone), mask annealing
  (steps scaled to our schedule) and L2 = 3 query blocks for ViT-S.
- **Reduced to fit our compute:** fewer epochs than the paper's 31 (12 by default, chosen as
  described above). All regimes use the same hyperparameters; none were tuned per regime.
- **LoRA learning rate:** the LoRA parameters sit inside the backbone blocks, so they share the
  backbone's learning rate and warmup.

### Results

To be filled in from `results/results.md`.

---

# Your ViT is Secretly an Image Segmentation Model  
[![Papers with Code: SOTA on BRAVO (OOD)](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=640&live=1)](https://paperswithcode.co/benchmark/bravo-ood?task=image-segmentation&eval=640)
[![Papers with Code: SOTA on COCO 2017 Panoptic Segmentation](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=6260&live=1)](https://paperswithcode.co/benchmark/coco-2017-panoptic-segmentation?task=image-segmentation&eval=6260)


**CVPR 2025 ✨ Highlight** · [📄 Paper](https://arxiv.org/abs/2503.19108)

**[Tommie Kerssies](https://tommiekerssies.com)<sup>1</sup>, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)<sup>2,*</sup>, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)<sup>3</sup>, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)<sup>1</sup>, [Giuseppe Averta](https://www.giuseppeaverta.me/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)<sup>3</sup>, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)<sup>1</sup>, [Daan de Geus](https://ddegeus.github.io)<sup>1,3</sup>**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

## Overview

We present the **Encoder-only Mask Transformer (EoMT)**, a minimalist image segmentation model that repurposes a plain Vision Transformer (ViT) to jointly encode image patches and segmentation queries as tokens. No adapters. No decoders. Just the ViT.

Leveraging large-scale pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, for example up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT shows that architectural complexity isn't necessary. For segmentation, a plain Transformer is all you need.

## 🚀 NEW: PMT 

Presenting our latest model, [PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders](https://arxiv.org/abs/2603.25398).

PMT reconciles EoMT minimal philosophy with the need of preserving the features of frozen Foundation Models, by mimicking the last layers of EoMT and VidEoMT with a simple and fast decoder.

Take a [look](https://github.com/tue-mps/pmt)!

## 🚀 NEW: VidEoMT 

🔥 We're pleased to present our latest CVPR 2026 paper, [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](https://arxiv.org/abs/2602.17807).

VidEoMT extends EoMT philosophy to the temporal domain, introducing an encoder-only video segmentation model that is up to 10x faster than competitors.

Go check it [out](https://github.com/tue-mps/videomt)! 


## 🚀 NEW: DINOv3 Support

🔥 We're excited to announce support for **DINOv3** backbones! Our new DINOv3-based EoMT models deliver improved performance across all segmentation tasks:

- **Panoptic Segmentation**: Up to 58.9 PQ on COCO with EoMT-L at 1280×1280
- **Instance Segmentation**: Up to 49.9 mAP on COCO with EoMT-L at 1280×1280  
- **Semantic Segmentation**: Up to 59.5 mIoU on ADE20K with EoMT-L at 512×512

All of this, at the impressive speed of EoMT!

Check out our [DINOv3 Model Zoo](model_zoo/dinov3.md) for all available EoMT configurations and performance benchmarks.

Thanks to the [DINOv3](https://github.com/facebookresearch/dinov3) team for providing these powerful foundation models!

## 🤗 Transformers

EoMT with DINOv2 is also available on [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/model_doc/eomt). See available models [here](https://huggingface.co/models?library=transformers&other=eomt&sort=trending).

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch, run:

```bash
python3 main.py fit \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16. Switch to ```dinov3``` in the configuration path to enable the corresponding DINOv3 model.

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

## Model Zoo

We provide pre-trained weights for both DINOv2- and DINOv3-based EoMT models.

- **[DINOv2 Models](model_zoo/dinov2.md)** - Original published results and pre-trained weights.
- **[DINOv3 Models](model_zoo/dinov3.md)** - New DINOv3-based models and pre-trained weights.

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
