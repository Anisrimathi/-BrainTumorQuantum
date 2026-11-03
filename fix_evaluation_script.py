
from pathlib import Path

path = Path("src/evaluate_heldout_models.py")
lines = path.read_text(encoding="utf-8-sig").splitlines()

start = next(
    i for i, line in enumerate(lines)
    if line.strip() == 'test_data = json.loads(TEST_PATIENTS_PATH.read_text(encoding="utf-8"))'
)
end = next(
    i for i, line in enumerate(lines)
    if line.strip().startswith('models = {"baseline_attention_unet"')
)

replacement = '''    test_data = json.loads(TEST_PATIENTS_PATH.read_text(encoding="utf-8"))
    test_ids = test_data["patient_ids"]

    used_ids = (
        set(cfg["train_patient_ids"])
        | set(cfg["validation_patient_ids"])
        | set(baseline_ckpt.get("train_case_ids", []))
        | set(baseline_ckpt.get("val_case_ids", []))
    )

    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Duplicate patient IDs in held-out test list.")

    overlap = set(test_ids) & used_ids
    if overlap:
        raise RuntimeError(f"Test patients overlap existing splits: {sorted(overlap)}")

    all_cases = {case["id"]: case for case in discover_cases()}
    missing = [pid for pid in test_ids if pid not in all_cases]
    if missing:
        raise RuntimeError(f"Missing test patients: {missing}")

    test_ids = sorted(test_ids)
    cases = [all_cases[pid] for pid in test_ids]

    # Build the patch list once. Both models consume the same tensors.
    dataset = BraTSPatchDataset(
        cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=2027,
    )
    fixed_patches = [dataset[i] for i in range(len(dataset))]
    print(f"Prepared {len(fixed_patches)} fixed patches for both models.")

    device = select_device()
    print("Inference device:", device)

    baseline = AttentionUNet3D(
        in_channels=4, num_classes=4, base_channels=8
    )
    baseline.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline = baseline.to(device).eval()

    hybrid = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy=hybrid_ckpt["strategy"],
    )
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    hybrid = hybrid.to(device).eval()
'''

lines[start:end] = replacement.rstrip().splitlines()
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Replaced the damaged section.")
