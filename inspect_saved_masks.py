from pathlib import Path
import numpy as np

p = Path("results/visualizations/baseline_vs_hybrid_v2/BraTS-GLI-00036-001_predictions.npz")

with np.load(p) as data:
    print("Available arrays:", data.files)

    for name in data.files:
        arr = data[name]
        print(f"\n{name}:")
        print("  Shape:", arr.shape)
        print("  Dtype:", arr.dtype)

        if arr.ndim == 3:
            labels, counts = np.unique(arr, return_counts=True)
            print("  Label counts:", dict(zip(labels.tolist(), counts.tolist())))
