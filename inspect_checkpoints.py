import torch
from pathlib import Path

files = [
    Path("results/checkpoints/attention_unet_pilot_best.pth"),
    Path("results/checkpoints/hybrid_quantum_unet_ring_pilot.pth"),
    Path("results/checkpoints/hybrid_quantum_unet_ring_extended.pth"),
]

for path in files:
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("NOT FOUND")
        continue

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("Checkpoint type:", type(ckpt).__name__)
        continue

    print("Metadata keys:", [
        k for k in ckpt.keys() if k != "model_state_dict"
    ])
    for key in (
        "epoch", "best_val_loss", "best_val_dice",
        "strategy", "model_config", "configuration"
    ):
        if key in ckpt:
            print(f"{key}: {ckpt[key]}")

    state = ckpt.get("model_state_dict", {})
    if isinstance(state, dict):
        print("Parameter tensors:", len(state))
        print("Total parameters:", sum(
            v.numel() for v in state.values()
            if isinstance(v, torch.Tensor)
        ))
