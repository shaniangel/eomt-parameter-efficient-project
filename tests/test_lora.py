import pytest
import torch
import torch.nn as nn

from models.lora import LoRALinear, apply_lora

DIM = 16


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        return self.proj(self.qkv(x)[..., :DIM])


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()

    def forward(self, x):
        return x + self.attn(x)


class TinyViT(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


def test_lora_linear_matches_base_at_init():
    base = nn.Linear(DIM, 2 * DIM)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 5, DIM)

    torch.testing.assert_close(lora(x), base(x), rtol=0, atol=0)


def test_apply_lora_wraps_last_blocks_and_counts_params():
    backbone = TinyViT()
    rank = 4

    num_added = apply_lora(backbone, [-3, -2, -1], rank=rank, alpha=4.0)

    for i, block in enumerate(backbone.blocks):
        expected = LoRALinear if i >= 3 else nn.Linear
        assert isinstance(block.attn.qkv, expected)
        assert isinstance(block.attn.proj, expected)
    assert num_added == 3 * (rank * (DIM + 3 * DIM) + rank * (DIM + DIM))


def test_frozen_backbone_only_updates_lora_params():
    torch.manual_seed(0)
    backbone = TinyViT()
    backbone.requires_grad_(False)
    apply_lora(backbone, [-2, -1], rank=4, alpha=4.0)

    trainable = {n for n, p in backbone.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    before = {n: p.detach().clone() for n, p in backbone.named_parameters()}
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(2):  # lora_A only gets a gradient once lora_B is non-zero
        optimizer.zero_grad()
        backbone(torch.randn(4, 3, DIM)).pow(2).mean().backward()
        optimizer.step()

    for name, param in backbone.named_parameters():
        changed = not torch.equal(before[name], param)
        assert changed == (name in trainable), name


def test_apply_lora_rejects_bad_block_index():
    with pytest.raises(IndexError):
        apply_lora(TinyViT(num_blocks=2), [5], rank=4, alpha=4.0)


def test_apply_lora_rejects_missing_module():
    with pytest.raises(ValueError):
        apply_lora(TinyViT(), [-1], rank=4, alpha=4.0, modules=("fc1",))
