"""Lightweight LoRA utilities for the EoMT project.

This module provides a small LoRA wrapper for nn.Linear layers and a helper
to insert LoRA adapters into the attention projections of the final ViT blocks
used by `EoMT`.

Usage example:
    from models.lora import apply_lora_to_backbone
    apply_lora_to_backbone(model.encoder.backbone, num_blocks=4, r=8, alpha=16)

The function will freeze existing backbone parameters and leave only the newly
added LoRA parameters trainable.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear and add a low-rank trainable update.

    The original linear is kept frozen (its params' requires_grad are set to
    False by the caller). The LoRA update has rank `r` and is scaled by
    `alpha / r`.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.lora_B = nn.Parameter(torch.zeros(base.in_features, r))
            self.lora_A = nn.Parameter(torch.zeros(r, base.out_features))
            # scaling factor applied in forward
            self.scaling = alpha / max(1, r)
            # initialize A with normal and B with zeros (common LoRA init)
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)
        else:
            # no-op
            self.lora_B = None
            self.lora_A = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base forward (use existing weights)
        out = self.base(x)
        if self.r > 0 and self.lora_A is not None:
            # LoRA update: (x @ B) @ A  => shape (batch, *, out_features)
            # handle 2D or 3D inputs
            orig_shape = x.shape
            if x.dim() == 3:
                # (B, N, C) -> (B*N, C)
                x_2d = x.reshape(-1, x.shape[-1])
                lora_part = (x_2d @ self.lora_B) @ self.lora_A
                lora_part = lora_part.reshape(orig_shape[0], orig_shape[1], -1)
            else:
                lora_part = (x @ self.lora_B) @ self.lora_A

            out = out + lora_part * self.scaling
        return out


class LoRAQKV(nn.Module):
    """Specialized wrapper for a qkv linear that applies LoRA only to the
    query slice (first third of the output features).

    The base linear projects to 3*embed_dim. We keep the base linear and add
    a low-rank update that only affects the query output.
    """

    def __init__(self, base: nn.Linear, embed_dim: int, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.embed_dim = embed_dim
        self.r = r
        self.alpha = alpha
        if r > 0:
            # LoRA parameters target only the query output (first embed_dim columns)
            self.lora_B = nn.Parameter(torch.zeros(base.in_features, r))
            self.lora_A = nn.Parameter(torch.zeros(r, embed_dim))
            self.scaling = alpha / max(1, r)
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_B = None
            self.lora_A = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base out: (..., 3*embed_dim)
        out = self.base(x)

        if self.r > 0 and self.lora_A is not None:
            orig_shape = out.shape
            # compute LoRA update for query slice
            # x may be (B,N,C) or (B*N, C)
            if x.dim() == 3:
                x_2d = x.reshape(-1, x.shape[-1])
                lora_q = (x_2d @ self.lora_B) @ self.lora_A  # (B*N, embed_dim)
                lora_q = lora_q.reshape(orig_shape[0], orig_shape[1], -1)
                # reshape out to (B, N, 3*embed_dim)
                out = out.reshape(orig_shape[0], orig_shape[1], -1)
                out[:, :, : self.embed_dim] = out[:, :, : self.embed_dim] + lora_q * self.scaling
                out = out.reshape(orig_shape)
            else:
                lora_q = (x @ self.lora_B) @ self.lora_A
                out[:, : self.embed_dim] = out[:, : self.embed_dim] + lora_q * self.scaling

        return out


def _replace_linear_with_lora(module: nn.Module, attr: str, r: int, alpha: float):
    """Replace attribute `attr` on `module` if it's an nn.Linear with a
    LoRALinear wrapper. Returns True if replacement happened.
    """
    orig = getattr(module, attr, None)
    if orig is None:
        return False
    if isinstance(orig, nn.Linear):
        lora = LoRALinear(orig, r=r, alpha=alpha)
        setattr(module, attr, lora)
        return True
    return False


def apply_lora_to_backbone(
    backbone: nn.Module,
    num_blocks: int = 4,
    r: int = 4,
    alpha: float = 1.0,
    target: str = "qkv",
):
    """Insert LoRA adapters into attention projections of the last `num_blocks`
    Transformer blocks of a timm/transformers ViT backbone used by this repo.

    This function:
    - freezes all existing backbone parameters (requires_grad=False)
    - replaces `qkv` and `proj` linear layers in the attention modules of the
      last `num_blocks` with LoRALinear wrappers (which keep base linear but
      expose trainable low-rank updates)

    Returns a small summary dict with counts of trainable / total params.
    """
    # freeze all backbone params first
    for p in backbone.parameters():
        p.requires_grad = False

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("Backbone has no attribute 'blocks' to apply LoRA to")

    target_blocks = list(blocks[-num_blocks:])

    replaced = 0
    for block in target_blocks:
        # attn may be named 'attn' or 'attention'
        attn = getattr(block, "attn", None) or getattr(block, "attention", None)
        if attn is None:
            continue

        # Replace according to requested target
        if target == "q":
            # Replace the combined qkv projection with LoRA only on the q slice
            orig = getattr(attn, "qkv", None)
            if isinstance(orig, nn.Linear):
                out_f = orig.out_features
                if out_f % 3 == 0:
                    embed_dim = out_f // 3
                    lora_mod = LoRAQKV(orig, embed_dim=embed_dim, r=r, alpha=alpha)
                    setattr(attn, "qkv", lora_mod)
                    replaced += 1
                else:
                    # fallback to full qkv LoRA if shapes unexpected
                    if _replace_linear_with_lora(attn, "qkv", r, alpha):
                        replaced += 1

        elif target == "qkv":
            if _replace_linear_with_lora(attn, "qkv", r, alpha):
                replaced += 1

        elif target == "proj":
            if _replace_linear_with_lora(attn, "proj", r, alpha):
                replaced += 1

        else:
            # unknown target: try replacing both qkv and proj with LoRA
            if _replace_linear_with_lora(attn, "qkv", r, alpha):
                replaced += 1
            if _replace_linear_with_lora(attn, "proj", r, alpha):
                replaced += 1

    # Count params
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    return {
        "replaced_modules": replaced,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }

