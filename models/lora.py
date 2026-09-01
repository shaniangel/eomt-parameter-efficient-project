# LoRA module and utilities for applying Low-Rank Adapters to ViT backbones
# This file adds a lightweight LoRA wrapper for nn.Linear layers and
# provides a helper to attach LoRA adapters to attention projection
# layers in timm/transformer-style ViT backbones used by this project.
#
# Changes made:
# - Implemented LoRALinear: wraps an existing nn.Linear and adds a
#   trainable low-rank update (B @ A) that is added to the original
#   linear output in the forward pass.
# - Provided apply_lora_to_backbone(backbone, target_blocks, rank, alpha, modules)
#   which replaces matching Linear layers (qkv, proj) in the specified
#   blocks with LoRALinear wrappers so only the LoRA parameters are
#   trainable when the backbone is frozen.

from typing import List, Sequence, Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear and adds a low-rank adapter: out = orig(x) + alpha/rank * (A @ (B @ x)).
    The adapter is initialized to zeros so behavior is identical to the
    original network before training.

    This supports replacing combined qkv projections as well as separate
    projection layers. The wrapped original linear remains and its
    parameters are preserved (and can be frozen externally).
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        # keep the original linear layer (weights & bias)
        self.orig = orig_linear
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        # LoRA parameters: down (B) and up (A)
        if rank > 0:
            self.rank = rank
            self.lora_A = nn.Parameter(
                torch.zeros((self.out_features, rank)), requires_grad=True
            )
            self.lora_B = nn.Parameter(
                torch.zeros((rank, self.in_features)), requires_grad=True
            )
            # scaling as in LoRA: alpha / r
            self.alpha = alpha
            # initialize A with zeros and B with zeros so adapter is inactive at init
            # (keeps original behavior)
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        else:
            self.rank = 0
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original linear forward
        out = self.orig(x)
        if self.rank > 0:
            # LoRA forward: A @ (B @ x.T) -> (out_feat, batch)
            # compute B @ x^T efficiently by x @ B^T then * A^T
            # x: (B, N, in_features) or (B, in_features)
            # support both batched and 2D inputs
            if x.dim() == 3:
                # treat last dim as features
                b, n, _ = x.shape
                x_2d = x.reshape(-1, x.shape[-1])  # (B*N, in)
                lora_down = torch.matmul(x_2d, self.lora_B.t())  # (B*N, r)
                lora_up = torch.matmul(lora_down, self.lora_A.t())  # (B*N, out)
                lora_up = lora_up.reshape(b, n, self.out_features)
            else:
                lora_down = torch.matmul(x, self.lora_B.t())
                lora_up = torch.matmul(lora_down, self.lora_A.t())

            out = out + (self.alpha / max(1, self.rank)) * lora_up

        return out


def _iter_target_blocks(backbone, target_blocks: Sequence[int]):
    """Yield (block_idx, block_module) for requested indices; supports negative indices."""
    blocks = list(backbone.blocks)
    n = len(blocks)
    resolved = []
    for idx in target_blocks:
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise IndexError(f"Block index {idx} out of range for backbone with {n} blocks")
        if idx in resolved:
            continue
        resolved.append(idx)
        yield idx, blocks[idx]


def apply_lora_to_backbone(
    backbone: nn.Module,
    target_blocks: Sequence[int] = (-3, -2, -1),
    rank: int = 8,
    alpha: float = 1.0,
    modules: Optional[Sequence[str]] = ("qkv", "proj"),
) -> int:
    """
    Replace target Linear layers in the backbone's attention modules with LoRALinear wrappers.

    - backbone: ViT backbone adapted by models.vit (has .blocks)
    - target_blocks: indices (can be negative) of blocks to modify
    - rank, alpha: LoRA hyperparameters
    - modules: names of Linear attributes to replace inside attention module

    Returns number of LoRA parameters added (approx).
    """
    added = 0
    for idx, block in _iter_target_blocks(backbone, target_blocks):
        # attention may be stored as block.attn or block.attention
        attn = getattr(block, "attn", None) or getattr(block, "attention", None)
        if attn is None:
            continue

        for name in modules:
            if hasattr(attn, name):
                linear = getattr(attn, name)
                # if it's a fused qkv (nn.Linear) or proj (nn.Linear), wrap it
                if isinstance(linear, nn.Linear):
                    wrapped = LoRALinear(linear, rank=rank, alpha=alpha)
                    setattr(attn, name, wrapped)
                    added += rank * (linear.in_features + linear.out_features)
                else:
                    # If not a Linear, skip (could be a fused kernel or custom op)
                    continue
    return added
