from lightning.pytorch.callbacks import Callback
import torch


class LoRAFreezeChecker(Callback):
    """
    Callback that snapshots selected parameters before the optimizer step and
    reports which parameters changed after the optimizer step. Useful to verify
    that frozen backbone weights remain unchanged while LoRA adapter weights
    (e.g. `lora_A` / `lora_B`) are updated.

    Parameters
    - watch_patterns: list of substrings; any parameter name containing one of
      these substrings will be included in the check.
    - max_items: maximum number of reported items.
    """

    def __init__(self, watch_patterns=None, max_items: int = 200):
        super().__init__()
        self.watch_patterns = watch_patterns or ["network.encoder.backbone", "lora_A", "lora_B", "orig.weight"]
        self.before = {}
        self.max_items = max_items

    def _should_watch(self, name: str) -> bool:
        for p in self.watch_patterns:
            if p in name:
                return True
        return False

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Snapshot selected parameter tensors to CPU
        self.before.clear()
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if self._should_watch(name):
                    try:
                        self.before[name] = p.detach().cpu().clone()
                    except Exception:
                        # ignore any parameter that can't be cloned
                        continue

    def on_after_optimizer_step(self, trainer, pl_module, optimizer):
        # Compare and report changes
        changes = []
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if name in self.before:
                    before = self.before.pop(name)
                    cur = p.detach().cpu()
                    try:
                        diff_norm = (cur - before).float().norm().item()
                    except Exception:
                        diff_norm = float("nan")
                    changes.append((name, diff_norm))

        if not changes:
            return

        # Print a compact table sorted by magnitude of change
        changes_sorted = sorted(changes, key=lambda x: (0.0 if x[1] is None else -x[1]))
        print("\n" + "=" * 80)
        print("LoRA / Freeze check after optimizer.step():")
        for name, diff in changes_sorted[: self.max_items]:
            status = "CHANGED" if (isinstance(diff, float) and diff > 0.0) else "UNCHANGED"
            print(f" {status:8s} | {diff:12.6e} | {name}")
        print("=" * 80 + "\n")

