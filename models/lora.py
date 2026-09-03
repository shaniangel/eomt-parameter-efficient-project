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

from typing import List, Sequence, Optional, Union, Mapping, Any
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


def _broadcast_param(value: Union[int, float, Sequence], length: int):
    """Broadcast a scalar or verify a sequence matches `length`.

    Returns a list of length `length` containing the values.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected length {length}, got {len(value)}")
        return list(value)
    else:
        return [value] * length


def apply_lora_to_backbone(
    backbone: nn.Module,
    target_blocks: Union[Sequence[int], Mapping[Any, Any]] = (-3, -2, -1),
    rank: Union[int, Sequence[int]] = 8,
    alpha: Union[float, Sequence[float]] = 1.0,
    modules: Optional[Sequence[str]] = ("qkv", "proj"),
) -> int:
    """
    Replace target Linear layers in the backbone's attention modules with LoRALinear wrappers.

    target_blocks may be either:
      - a sequence of int indices (legacy behavior), or
      - a Mapping where each key specifies one or more block indices and the
        corresponding value gives the LoRA spec for those blocks. Keys may be:
          - int (e.g. -3)
          - tuple/list (e.g. (-4, -3))
          - string forms like "-3", "-4,-3", or "(-4,-3)"
        Values may be:
          - scalar (interpreted as rank)
          - sequence [rank, alpha]
          - mapping {"rank": int, "alpha": float, "modules": [..]}

    Returns approximate number of adapter parameters added.
    """
    blocks = list(backbone.blocks)
    n = len(blocks)

    added = 0
    if isinstance(target_blocks, Mapping):
        # Build per-block spec: idx -> (rank, alpha, modules)
        per_block: dict[int, tuple[int, float, Optional[Sequence[str]]]] = {}
        for key, val in target_blocks.items():
            # parse key into list of ints
            if isinstance(key, (list, tuple)):
                keys = list(key)
            elif isinstance(key, int):
                keys = [key]
            elif isinstance(key, str):
                s = key.strip()
                try:
                    if (s.startswith("(") and s.endswith(")")) or (
                        s.startswith("[") and s.endswith("]")
                    ):
                        ks = eval(s)
                        keys = list(ks) if isinstance(ks, (list, tuple)) else [int(ks)]
                    elif "," in s:
                        keys = [int(x) for x in s.split(",") if x.strip()]
                    else:
                        keys = [int(s)]
                except Exception:
                    raise ValueError(f"Could not parse LoRA target key: {key}")
            else:
                raise ValueError(f"Unsupported LoRA target key type: {type(key)}")

            # parse value into rank/alpha/modules
            if isinstance(val, Mapping):
                r = val.get("rank", rank)
                a = val.get("alpha", alpha)
                mods = val.get("modules", modules)
            elif isinstance(val, (list, tuple)):
                if len(val) >= 2:
                    r, a = val[0], val[1]
                else:
                    r = val[0]
                    a = alpha
                mods = modules
            else:
                # scalar interpreted as rank
                r = val
                a = alpha
                mods = modules

            for k in keys:
                per_block[int(k)] = (int(r), float(a), tuple(mods) if mods is not None else modules)

        # Apply per-block adapters
        for idx, (r, a, mods) in per_block.items():
            resolved_idx = idx if idx >= 0 else n + idx
            if resolved_idx < 0 or resolved_idx >= n:
                raise IndexError(f"Block index {idx} out of range for backbone with {n} blocks")
            block = blocks[resolved_idx]
            attn = getattr(block, "attn", None) or getattr(block, "attention", None)
            if attn is None:
                continue
            for name in mods:
                if hasattr(attn, name):
                    linear = getattr(attn, name)
                    if isinstance(linear, nn.Linear):
                        wrapped = LoRALinear(linear, rank=int(r), alpha=float(a))
                        setattr(attn, name, wrapped)
                        added += int(r) * (linear.in_features + linear.out_features)
        return added

    # Fallback: sequence of indices with optional broadcasted rank/alpha (legacy behavior)
    target_blocks_list = list(target_blocks)
    num_targets = len(target_blocks_list)

    ranks = _broadcast_param(rank, num_targets)
    alphas = _broadcast_param(alpha, num_targets)

    for idx, r, a in zip(target_blocks_list, ranks, alphas):
        # resolve negative indices
        resolved_idx = idx if idx >= 0 else n + idx
        if resolved_idx < 0 or resolved_idx >= n:
            raise IndexError(f"Block index {idx} out of range for backbone with {n} blocks")

        block = blocks[resolved_idx]
        attn = getattr(block, "attn", None) or getattr(block, "attention", None)
        if attn is None:
            continue

        for name in modules:
            if hasattr(attn, name):
                linear = getattr(attn, name)
                if isinstance(linear, nn.Linear):
                    wrapped = LoRALinear(linear, rank=int(r), alpha=float(a))
                    setattr(attn, name, wrapped)
                    added += int(r) * (linear.in_features + linear.out_features)
                else:
                    # skip non-linear modules (custom/fused ops)
                    continue

    return added
