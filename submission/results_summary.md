# BrainTumorQuantum ? Held-out Evaluation

## Evaluation setup
- Patients: 20
- Patches: 80
- Patch size: [32, 32, 32]
- Device: xpu
- Evaluation: held-out patient patch-level evaluation; exploratory comparison
- Training cohorts match: False

## Region Dice scores

| Region | Baseline Attention U-Net | Hybrid Quantum U-Net | Difference |
|---|---:|---:|---:|
| WT | 0.7011 | 0.6527 | -0.0484 |
| TC | 0.3045 | 0.3406 | +0.0360 |
| ET | 0.3250 | 0.3640 | +0.0391 |

## Interpretation

The hybrid model achieved higher Dice scores for TC and ET. The baseline achieved a higher WT Dice score. The hybrid model also showed higher recall for all three regions in this evaluation, which may be associated with more false-positive predictions.

## Limitations

This is an exploratory patch-level evaluation of 20 held-out patients using 80 patches. The models were trained on different cohorts, so the comparison does not isolate the effect of the hybrid quantum component. Patch-level metrics are not equivalent to full-volume patient-level evaluation, and these results do not establish quantum advantage.
