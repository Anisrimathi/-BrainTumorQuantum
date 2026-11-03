
import json

with open("results/comparison/baseline_vs_hybrid_full_volume.json", encoding="utf-8") as f:
    data = json.load(f)

regions = ["WT", "TC", "ET"]

print("REGION-WISE DICE COMPARISON")
print("-" * 95)
print("{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Patient", "Base WT", "Hybrid WT", "Base TC", "Hybrid TC", "Base ET", "Hybrid ET"
))

for patient, result in data["patients"].items():
    values = [patient]
    for region in regions:
        for model in ["baseline", "hybrid"]:
            score = result[model][region]["dice"]
            values.append("{:.4f}".format(score) if score is not None else "N/A")
    print("{:<22} {}".format(
        values[0], " ".join("{:>10}".format(x) for x in values[1:])
    ))

print("\nMODEL WINS BY REGION")
for region in regions:
    baseline_wins = hybrid_wins = ties = 0
    for result in data["patients"].values():
        b = result["baseline"][region]["dice"]
        h = result["hybrid"][region]["dice"]
        if b is None or h is None:
            continue
        if abs(b - h) < 1e-12:
            ties += 1
        elif b > h:
            baseline_wins += 1
        else:
            hybrid_wins += 1
    print("{}: Baseline wins={}, Hybrid wins={}, Ties={}".format(
        region, baseline_wins, hybrid_wins, ties
    ))
