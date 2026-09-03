# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # New LoRA / freezing options (accepted from config)
        freeze_backbone: bool = False,
        lora_enabled: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_target_blocks: Optional[List[int]] = None,
        lora_config: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

        # -----------------------------
        # LoRA integration and freezing
        # -----------------------------
        # These options are parsed from the config and enable two behaviors:
        # 1) freeze_backbone: set requires_grad=False for all backbone params
        # 2) lora_enabled: attach LoRA adapters to specified backbone blocks
        # The chosen order is: freeze backbone first, then attach LoRA adapters so
        # that the original backbone weights remain frozen while LoRA params are trainable.
        try:
            # local import to avoid circular import at module load time
            from models.lora import apply_lora_to_backbone
        except Exception:
            apply_lora_to_backbone = None

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for p in self.network.encoder.backbone.parameters():
                p.requires_grad = False

        # Apply LoRA adapters if requested
        if lora_enabled and apply_lora_to_backbone is not None:
            # Try parsing lora_config (JSON string or Python literal) if provided
            lora_spec = None
            if lora_config:
                try:
                    import json

                    if isinstance(lora_config, str):
                        lora_spec = json.loads(lora_config)
                    elif isinstance(lora_config, dict):
                        lora_spec = lora_config
                except Exception:
                    try:
                        lora_spec = eval(lora_config)
                    except Exception:
                        raise ValueError("Could not parse lora_config; provide a JSON string or Python dict literal")

            if lora_spec is not None:
                added = apply_lora_to_backbone(
                    self.network.encoder.backbone,
                    target_blocks=lora_spec,
                    modules=("qkv", "proj"),
                )
                import logging
                logging.info(f"Applied LoRA adapters via lora_config: approx params added={added}")
            else:
                # Fallback to legacy args
                target_blocks = lora_target_blocks or [-3, -2, -1]
                # allow comma-separated string form for the target blocks
                if isinstance(target_blocks, str):
                    target_blocks = [int(x) for x in target_blocks.split(",") if x.strip()]

                # allow lora_rank/alpha be csv lists or scalars (handled inside function)
                added = apply_lora_to_backbone(
                    self.network.encoder.backbone,
                    target_blocks=target_blocks,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    modules=("qkv", "proj"),
                )
                import logging
                logging.info(f"Applied LoRA adapters to blocks {target_blocks}: approx params added={added}")

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
