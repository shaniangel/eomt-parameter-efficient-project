def print_parameter_summary(module, max_list: int = 200):
    """
    Print a compact summary of trainable vs frozen parameters for `module`.

    Shows total counts and a short listing of parameter names/shape for each
    category (trainable / frozen).
    """
    total = 0
    trainable = 0
    frozen = 0
    trainable_list = []
    frozen_list = []

    for name, p in module.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if len(trainable_list) < max_list:
                trainable_list.append((name, tuple(p.shape)))
        else:
            frozen += n
            if len(frozen_list) < max_list:
                frozen_list.append((name, tuple(p.shape)))

    print("=" * 80)
    print(f"Module: {module.__class__.__name__}")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({100.0 * trainable / max(1, total):.2f}%)")
    print(f"Frozen params: {frozen:,} ({100.0 * frozen / max(1, total):.2f}%)")
    print("-" * 80)
    print("Sample trainable parameters:")
    for n, s in trainable_list[:max_list]:
        print(f"  [TRAIN] {n:120s} shape={s}")
    print("-" * 80)
    print("Sample frozen parameters:")
    for n, s in frozen_list[:max_list]:
        print(f"  [FROZEN] {n:120s} shape={s}")
    print("=" * 80)

