# ---------------------------------------------------------------
# Low-Rank Adaptation (LoRA, Hu et al., 2021) for the attention
# projections of the EoMT ViT backbone.
# ---------------------------------------------------------------


import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a linear layer as ``y = base(x) + (alpha / rank) * B(A(x))``.

    ``A`` (rank x in) gets a Kaiming init and ``B`` (out x rank) starts at zero, so the
    wrapped layer initially computes exactly the same function as ``base``.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * F.linear(
            F.linear(x, self.lora_A), self.lora_B
        )


def apply_lora(
    backbone: nn.Module,
    block_indices: Iterable[int],
    rank: int,
    alpha: float,
    modules: Sequence[str] = ("qkv", "proj"),
) -> int:
    """Wraps ``backbone.blocks[i].attn.<module>`` with LoRA for every given block.

    Negative block indices count from the end. Returns the number of added parameters.
    """
    blocks = backbone.blocks
    num_added = 0

    for idx in block_indices:
        if not -len(blocks) <= idx < len(blocks):
            raise IndexError(f"Block {idx} out of range for {len(blocks)} blocks")

        attn = blocks[idx].attn
        for name in modules:
            layer = getattr(attn, name, None)
            if not isinstance(layer, nn.Linear):
                raise ValueError(f"blocks[{idx}].attn.{name} is not an nn.Linear")

            lora_layer = LoRALinear(layer, rank, alpha)
            setattr(attn, name, lora_layer)
            num_added += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    return num_added
