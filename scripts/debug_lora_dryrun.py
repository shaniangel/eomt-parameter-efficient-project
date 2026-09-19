import logging
import torch
import torch.nn as nn
from models.lora import apply_lora_to_backbone

from scripts.debug_param_summary import print_parameter_summary


logging.basicConfig(level=logging.INFO, format="%(message)s")


class FakeAttn(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        # simple qkv and proj as linear layers
        self.qkv = nn.Linear(in_features, out_features)
        self.proj = nn.Linear(out_features, out_features)


class Block(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.attn = FakeAttn(in_features, out_features)


class FakeBackbone(nn.Module):
    def __init__(self, num_blocks=6, in_features=16, out_features=16):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_features, out_features) for _ in range(num_blocks)])


class Container(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = backbone


def snapshot_params(module, watch_substrings):
    snap = {}
    for name, p in module.named_parameters():
        if any(s in name for s in watch_substrings):
            snap[name] = p.detach().cpu().clone()
    return snap


def compare_snaps(before, after):
    rows = []
    for name, b in before.items():
        a = after.get(name)
        if a is None:
            continue
        diff = (a - b).float().norm().item()
        rows.append((name, diff))
    return rows


def run():
    torch.manual_seed(0)
    backbone = FakeBackbone(num_blocks=6, in_features=16, out_features=16)
    model = Container(backbone)

    # Freeze backbone
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False

    # Apply LoRA to last block
    added = apply_lora_to_backbone(model.encoder.backbone, target_blocks=[-1], rank=2, alpha=1.0, modules=("qkv", "proj"))
    logging.info(f"Applied LoRA adapters: approx params added={added}")

    # Print parameter summary
    print_parameter_summary(model.encoder.backbone)

    # Build optimizer over trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Optimizer will update {sum(p.numel() for p in trainable)} params across {len(trainable)} tensors")
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    watch = ["lora_A", "lora_B", "orig.weight"]
    before = snapshot_params(model, watch)

    # Simple forward/backward on the LoRA-wrapped projection
    model.train()
    x = torch.randn(4, 16)
    out = model.encoder.backbone.blocks[-1].attn.qkv(x)
    loss = out.abs().mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    after = snapshot_params(model, watch)
    rows = compare_snaps(before, after)

    logging.info("\n" + "=" * 80)
    logging.info("LoRA / Freeze dry-run parameter change norms:")
    for name, diff in rows:
        status = "CHANGED" if diff > 0.0 else "UNCHANGED"
        logging.info(f" {status:8s} | {diff:12.6e} | {name}")
    logging.info("=" * 80 + "\n")


if __name__ == "__main__":
    run()

