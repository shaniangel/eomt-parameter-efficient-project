# How the code works

This document explains what we added to the official EoMT code and how the pieces fit together.
Everything not mentioned here is the unchanged upstream code.

## The big picture

```text
  configs/project/
  ┌───────────────────────────────────────────────┐
  │ base_ade20k_eomt_small_512.yaml               │  model, data, schedule (shared)
  │ + full.yaml | frozen.yaml | lora.yaml         │  what is trained
  │ + smoke.yaml (optional)                       │  short learning test
  └───────────────────────┬───────────────────────┘
                          │ read by
                          ▼
  scripts/run_experiments.sh ──► main.py (upstream LightningCLI)
                                    │
          ┌─────────────────────────┼──────────────────────────────┐
          ▼                         ▼                              ▼
  datasets/ade20k_semantic.py   training/mask_classification_    training/csv_logger.py
  reads ADEChallengeData2016    semantic.py                      one run folder per start
  .zip                          builds EoMT, then freezes the    time
                                backbone and/or adds LoRA
                                (models/lora.py)
                                    │
                                    ▼
                                training/lightning_module.py
                                training loop, metrics, efficiency
                                stats, progress (training/progress.py)
                                    │
                                    ▼
                  logs/<regime>/<start time>/     (logs_smoke/ for smoke runs)
                  ├── config.yaml             every setting of the run
                  ├── metrics.csv
                  ├── progress/               pictures, curves, text log per epoch
                  ├── checkpoints/
                  └── efficiency_stats.json   written when training ends
                          │
          ┌───────────────┼────────────────────────────────┐
          ▼               ▼                                ▼
  scripts/check_smoke.py  scripts/summarize_results.py     scripts/visualize_predictions.py
  PASS/FAIL per regime    results/results.md, .csv,        results/qualitative.png
  (smoke runs)            results/curves_*.png
```

In words:

1. **Configs** describe a run. The base config is shared by all regimes, and the regime config
   only says what is trained. They are stacked with `-c base.yaml -c lora.yaml`; later files
   override earlier ones.
2. **`main.py`** (unchanged upstream code) reads the configs, builds the dataset, the model, the
   logger and the PyTorch Lightning trainer, and starts training.
3. **The model** is built in `mask_classification_semantic.py`. It creates EoMT as upstream
   does, then freezes the backbone and/or adds LoRA layers, depending on the regime.
4. **During training**, `lightning_module.py` runs the upstream training loop and writes our
   extra outputs: parameter counts, time and memory, and prediction images.
5. **After training**, two scripts turn the run folders into the results table, the curves and
   the qualitative figure.

## What each regime trains

EoMT-S uses a ViT-S backbone with 12 transformer blocks. The 100 segmentation queries are added
before the last 3 blocks (`num_blocks: 3`, called L2 in the paper). In those blocks the queries
and the image patches attend to each other, and the heads turn the query tokens into masks and
classes.

```text
  image patches ──► blocks 0-8 ──► blocks 9-11 ──────► class head, mask head, upscale layers
                    (image only)   (image + queries)
                                        ▲   LoRA goes on attn.qkv and attn.proj here
                   100 queries ─────────┘
```

Inside each of the 12 blocks (sizes for ViT-S, 384 features per token):

| Component | What it does | Weights | LoRA? |
|---|---|---|---|
| `norm1`, `norm2` | normalize each token | small | no |
| `attn.qkv` | turns every token into a query, key and value (384 → 3×384) | 443 K | **yes**, in blocks 9-11 |
| attention itself | each token mixes in information from the tokens it attends to | none | – |
| `attn.proj` | combines the attention heads' outputs (384 → 384) | 148 K | **yes**, in blocks 9-11 |
| `ls1`, `ls2` | LayerScale: learned per-feature scaling of each branch | tiny | no |
| `mlp.fc1`, `mlp.fc2` | per-token feed-forward network (384 → 1536 → 384) | 1.18 M | no |

Attention is the only place where tokens exchange information, so in blocks 9-11 it is where the
queries collect information from the image patches. The MLP works on every token separately.
LoRA on `qkv` and `proj` therefore adapts exactly the query–image interaction, with 18,432
parameters per block (qkv: 8×384 + 1152×8; proj: 8×384 + 384×8).

| Regime | Backbone blocks 0-8 | Backbone blocks 9-11 | Queries + heads |
|---|---|---|---|
| `full` | trained | trained | trained |
| `frozen` | frozen | frozen | trained |
| `lora` | frozen | frozen, plus trainable LoRA on `qkv` and `proj` | trained |

"Frozen" means the weights get no gradient and never change. Gradients still flow *through*
frozen blocks, so the queries in blocks 9-11 can still learn.

Parameter counts at 512×512 (from `efficiency_stats.json`):

| Regime | Trainable parameters | Share of the backbone that is trained |
|---|---|---|
| `full` | 23.7 M | 100% |
| `frozen` | 1.73 M (queries and heads) | 0% |
| `lora` | 1.78 M (queries, heads and 55,296 LoRA parameters) | 0.25% |

## The files we added or changed

### `models/lora.py`: the LoRA layer

A normal linear layer computes `y = W x`. LoRA keeps `W` frozen and adds a small trainable
correction made of two thin matrices:

```
y = W x  +  (alpha / rank) · B (A x)
```

- `A` has shape `rank × in` and `B` has shape `out × rank`. With rank 8 and ViT-S sizes, that is
  a few thousand numbers instead of the hundreds of thousands in `W`.
- `B` starts at zero, so at the start of training the correction is zero and the model behaves
  exactly like the pretrained one. `A` starts random (Kaiming init). Otherwise both matrices
  would get zero gradients and never learn.
- `alpha / rank` is a fixed scale factor. We use `alpha = rank = 8`, so the scale is 1.

`apply_lora(backbone, block_indices, rank, alpha, modules)` goes to each chosen block, takes
`block.attn.qkv` and `block.attn.proj`, and replaces each with a `LoRALinear` that wraps the
original layer. It returns how many parameters were added. If a requested layer does not exist
or is not a linear layer, it raises an error instead of quietly skipping it.

EoMT's own attention code calls `module.qkv(x)` and `module.proj(...)`, so the wrapped layers
are used automatically. No other model code needed to change.

### `training/mask_classification_semantic.py`: freezing and LoRA options

Four new settings, which can be set in the config files:

| Setting | Default | Meaning |
|---|---|---|
| `freeze_backbone` | `false` | stop all ViT backbone weights from training |
| `lora_rank` | `0` | LoRA rank; 0 means no LoRA |
| `lora_alpha` | = rank | LoRA scale numerator |
| `lora_modules` | `[qkv, proj]` | which attention layers get LoRA |

After the upstream model is built (including loading the pretrained DINOv2 weights), the code:

1. freezes the backbone if `freeze_backbone` is set;
2. if `lora_rank > 0`, adds LoRA to the last `num_blocks` blocks, so LoRA always follows the
   blocks that process the queries.

The order matters: freezing first and adding LoRA second means the original weights are frozen
while the new LoRA weights stay trainable.

### `training/lightning_module.py`: measurements and outputs

- **At the start of training:** counts total, trainable and backbone parameters, works out the
  mask-annealing steps (see below), then starts a timer and resets the GPU peak-memory counter.
- **At the end of training** (`on_train_end`): adds the training time, seconds per step and peak
  GPU memory, and writes everything to `efficiency_stats.json` in the run folder. This file is
  only written when training finishes, which is how the result scripts tell finished runs apart
  from crashed ones.
- **Resumed runs:** each checkpoint also stores the training time so far, the peak memory and the
  progress history. A resumed run adds to them, so its training time covers all sessions, and
  `sec_per_step` is measured over the current session only. (The part of an interrupted epoch
  after its last checkpoint is redone, and not counted twice.)
- **Progress after every validation epoch.** Six validation batches spread over the validation
  set are chosen; the first image of each is tracked. Validation isn't shuffled, so these are
  the same 6 images every epoch. After validation, `training/progress.py` rewrites the run's
  `progress/` folder: a picture of the images, ground truth and current prediction
  (`epoch_<n>.png`), the same images across up to 6 epochs so far (`evolution.png`), loss and
  mIoU curves (`curves.png`), and a text log with the estimated time left (`progress.txt`).
  Images are stored at most 320 pixels wide, so this costs almost no memory or time.
- **Mask annealing.** EoMT trains with masked attention and slowly switches it off, one query
  block after another, so the final model does not need it. Upstream hard-codes the steps where
  this happens for its own schedule length. When the config does not give them, we compute them
  from our total number of steps, using the same fractions of training as upstream. For our
  3 blocks, annealing starts at 1/6, 5/12 and 2/3 of training and each takes 1/6 of training.
- **A second mIoU** (`metrics/val_iou_present`). The standard mIoU (`metrics/val_iou_all`)
  averages the IoU of all 150 classes, and a class that appears in neither the ground truth
  nor the predictions counts as 0. On the full validation set every class appears, so this is
  the paper's metric and the one we report. On the few images of a smoke test, most classes
  never appear, so the standard mIoU is close to 0 even when the model is doing well on what it
  sees. The second mIoU averages only over the classes that do appear. It is a debugging aid.
- **Prediction images.** Upstream uploads a picture of the first validation image to wandb
  (`plot_semantic`). We don't use wandb, so validation no longer calls it; the progress files
  above replace it.

### `training/csv_logger.py`: one folder per run

The standard Lightning CSV logger names run folders `version_0`, `version_1`, and so on. Ours
names them after the start time, e.g. `logs/lora/2026-09-26_14-03-12/`, so repeated attempts
never overwrite each other and are easy to tell apart. Lightning's logger also deletes an
existing `metrics.csv` when it opens a folder; ours keeps it, so a resumed run continues its
curves in the same file.

### Other small changes

- `main.py` saves each run's fully resolved configuration, including command-line overrides, as
  `config.yaml` in the run folder. The result scripts rebuild models from it and compare runs'
  settings with it.
- `datasets/ade20k_semantic.py` has a separate `val_batch_size` (4 in our base config).
  Validation cuts every image into about 2 crops of 512×512, so it needs more memory per image
  than training.

### `configs/project/`

- `base_ade20k_eomt_small_512.yaml`: the shared setup. It covers:
  - the model: EoMT-S with DINOv2 ViT-S/14 weights and the patch embedding resized to 16×16 as
    in the paper, 100 queries, 3 query blocks;
  - the data: ADE20K at 512×512, batch 16, validation batch 4;
  - the paper's optimizer settings: lr 1e-4, layer-wise decay 0.8, weight decay 0.05;
  - 31 epochs, and CSV logging to `logs/`.
- `full.yaml`, `frozen.yaml`, `lora.yaml`: only the regime settings and the run name.
- `lora_r2.yaml`, `lora_r4.yaml`, `lora_r16.yaml`, `lora_r32.yaml`: the LoRA regime with another
  rank, for a rank ablation (alpha = rank, so the scale stays 1).
- `smoke.yaml`: a short learning test. 5 epochs of 20 training batches, 5 validation batches after
  each, a warmup shortened to [5, 5] steps so that the backbone and LoRA already train, and output
  to `logs_smoke/` so smoke runs never mix with real runs.

### `scripts/`

- `run_experiments.sh [regime...] [--smoke] [--resume <run folder>] [extra args]`: a regime is
  any config in `configs/project/` except the base and smoke ones. It trains the given regimes one after another on the current machine (give each machine one
  regime to run them in parallel) and saves each console output to
  `logs/<regime>_<start time>.out`. It stops if a run fails. `--resume` continues an interrupted
  run from its latest checkpoint, in the same folder.
- `check_smoke.py`: for each regime's latest smoke run (the kind of regime is read from the
  run's `config.yaml`, so rank ablations get the LoRA checks), checks that the training loss went down,
  the validation mIoU went up, and exactly the right weights changed compared with the pretrained
  DINOv2 weights (full: backbone changed; frozen: identical; LoRA: backbone identical and every
  LoRA `lora_B` non-zero). Prints PASS/FAIL and exits with an error code if anything fails.
- `summarize_results.py`:
  - By default covers every regime folder in `logs/`, rank ablations included.
  - For each regime, picks the latest finished run (or the one given with `--run`) and reads
    `metrics.csv` and `efficiency_stats.json`.
  - Compares the runs' `config.yaml` files and stops if they differ in anything besides the
    regime, run naming and machine-specific paths (`--allow-mismatch` overrides this).
  - Writes the results table (`results.md`, `results.csv`), with the final mIoU and the
    difference from full fine-tuning.
  - Writes the validation mIoU and training loss curves.
- `visualize_predictions.py`: rebuilds each regime's model from its run's `config.yaml`, loads
  its checkpoint, predicts a few fixed validation images with the same sliding-window method as
  validation, and draws image | ground truth | full | frozen | LoRA side by side.

### `tests/test_lora.py`

Five quick CPU tests on a tiny fake ViT. They check that:

1. a LoRA layer gives exactly the same output as the original layer at the start;
2. LoRA is added to the right blocks with the right number of parameters;
3. after freezing and adding LoRA, training changes only the LoRA weights;
4. a wrong block index raises an error;
5. a wrong layer name raises an error.

Run them with `python -m pytest tests`.

## Things to keep in mind when reading results

- In the real runs, for the first 500 steps full and frozen training behave the same, and the
  LoRA weights do not change yet (the smoke test shortens this warmup). The paper's warmup trains
  only the new parameters (queries and heads) first, then warms up the backbone learning rate
  over the next 1,000 steps. The regimes only start to differ after that.
- LoRA parameters live inside the backbone blocks, so they get the backbone's learning rate and
  warmup, the same as the weights they adapt.
- All regimes use the same hyperparameters. We did not tune the learning rate per regime, which
  might favour one regime over another.
