import json
from pathlib import Path

p = Path("results/comparison/baseline_vs_hybrid_40_full_volume.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("Evaluation method:", data.get("method", "Not specified"))

def walk(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ("wt", "tc", "et") and isinstance(v, dict):
                print(f"{path}.{k}: Dice={v.get('dice', 'N/A')}")
            elif k.lower() in ("model", "mean_dice", "mean_foreground_dice"):
                print(f"{path}.{k}: {v}")
            else:
                walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")

walk(data)
