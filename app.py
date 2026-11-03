
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

BASELINE = CHECKPOINTS / "attention_unet_pilot_best.pth"
HYBRID = CHECKPOINTS / "hybrid_quantum_unet_ring_extended.pth"

HISTORY = RESULTS / "hybrid_ring_extended_history.json"
EVALUATION = RESULTS / "hybrid_ring_extended_evaluation.json"
COMPARISON = RESULTS / "comparison" / "baseline_vs_hybrid_extended.json"

st.set_page_config(
    page_title="BrainTumorQuantum | Neuro AI",
    page_icon="ðŸ§ ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #080d17;
  --panel: #101a2a;
  --border: #24344a;
  --muted: #91a2ba;
  --teal: #5eead4;
  --text: #edf5ff;
}

.stApp {
  background:
    radial-gradient(ellipse at 80% 0%, #102b3b 0%, transparent 38%),
    var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.6px;
}
.block-container {
  padding: 1.5rem 2rem 2rem;
  max-width: 100%;
  width: 100%;
}
html, body, [data-testid="stAppViewContainer"] {
  overflow-x: hidden;
}

[data-testid="stSidebar"] {
  min-width: 250px;
  max-width: 250px;
}

[data-testid="stMainBlockContainer"] {
  min-width: 0;
}

.hero {
  overflow-wrap: anywhere;
  box-sizing: border-box;
}
[data-testid="stSidebar"] {
  background: #0b1321;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {font-family: 'DM Sans', sans-serif;}
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, #132237, #0e1827);
  border: 1px solid #263b51;
  border-radius: 15px;
  padding: 18px 20px;
}
div[data-testid="stMetricLabel"] {color: #a7b8ce;}
div[data-testid="stMetricValue"] {color: #f2f8ff;}
div[data-testid="stAlert"] {border-radius: 12px;}
.stButton > button {
  background: #5eead4;
  color: #062b2a;
  border: 0;
  border-radius: 10px;
  font-weight: 700;
  min-height: 42px;
}
.stButton > button:hover {
  background: #99f6e4;
  color: #062b2a;
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
hr {border-color: #26374d;}
.hero {
  padding: 28px 30px;
  border-radius: 20px;
  border: 1px solid #284458;
  background: linear-gradient(115deg, #12263a 0%, #10202f 58%, #103a3b 100%);
  margin-bottom: 22px;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 2px;
  font-size: 11px;
  color: #5eead4;
  font-weight: 700;
}
.hero h1 {margin: 8px 0 10px; font-size: 36px;}
.hero p {color: #b5c7da; max-width: 800px; margin-bottom: 0;}
.panel {
  border: 1px solid var(--border);
  background: linear-gradient(145deg, #111d2d, #0d1624);
  border-radius: 15px;
  padding: 18px 20px;
}
.small-muted {color: #91a2ba; font-size: 13px;}
.tag {
  display: inline-block;
  color: #5eead4;
  border: 1px solid #285b5c;
  background: #102c31;
  padding: 5px 10px;
  border-radius: 20px;
  font-size: 12px;
  margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- DATA HELPERS --------------------
def read_json(path):
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.warning(f"Could not read {path.name}.")
    return None


def file_size(path):
    return f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.is_file() else "Missing"


def first_number(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


history = read_json(HISTORY) or {}
evaluation = read_json(EVALUATION) or {}
comparison = read_json(COMPARISON) or {}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## ðŸ§  BrainTumorQuantum")
    st.caption("NEUROIMAGING RESEARCH PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "MRI Viewer",
            "Model Comparison",
            "Training Analytics",
            "Evaluation Metrics",
            "Reports & Files",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown(
        f"{'ðŸŸ¢' if BASELINE.is_file() else 'ðŸ”´'} Attention U-Net"
    )
    st.markdown(
        f"{'ðŸŸ¢' if HYBRID.is_file() else 'ðŸ”´'} Hybrid Quantum U-Net"
    )
    st.markdown(
        f"{'ðŸŸ¢' if evaluation else 'ðŸŸ¡'} Extended evaluation"
    )
    st.divider()
    st.caption("Academic research prototype Â· Not for clinical diagnosis")


# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI research Â· 3D medical imaging</div>
  <h1>Brain MRI Intelligence</h1>
  <p>Explore 3D brain-tumor segmentation experiments using Attention U-Net
  and a hybrid quantum-classical neural network. Review model results,
  training behaviour, and saved experiment artefacts in one workspace.</p>
  <br>
  <span class="tag">BraTS research</span>
  <span class="tag">3D segmentation</span>
  <span class="tag">Quantum-classical ML</span>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype only. These experimental models are not validated "
    "for clinical use and must not be used to diagnose or guide treatment."
)

# -------------------- OVERVIEW --------------------
if page == "Overview":
    st.subheader("Research overview")
    st.caption("Project status from local checkpoint and result files.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model checkpoints", f"{int(BASELINE.is_file()) + int(HYBRID.is_file())} / 2")
    c2.metric("Hybrid epochs", len(history.get("epochs", [])))
    c3.metric("Evaluation classes", len(evaluation.get("aggregate_patch_metrics", {})))
    c4.metric("Comparison report", "Ready" if COMPARISON.is_file() else "Unavailable")

    st.markdown("### Experiment workflow")
    steps = st.columns(4)
    workflow = [
        ("01", "MRI preprocessing", "Four modalities: T1, T1ce, T2 and FLAIR."),
        ("02", "Model training", "Attention U-Net baseline and hybrid model."),
        ("03", "Evaluation", "Patch-level class metrics and comparisons."),
        ("04", "Research reporting", "Charts, saved metrics and experiment files."),
    ]
    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.markdown(
                f'<div class="panel"><div class="eyebrow">{number}</div>'
                f'<h3>{title}</h3><div class="small-muted">{description}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Available artefacts")
    files = [
        ("Attention U-Net checkpoint", BASELINE),
        ("Hybrid extended checkpoint", HYBRID),
        ("Hybrid training history", HISTORY),
        ("Extended evaluation", EVALUATION),
        ("Baseline vs hybrid comparison", COMPARISON),
    ]
    rows = [
        {"Artefact": name, "Status": "Available" if path.is_file() else "Missing",
         "Size": file_size(path)}
        for name, path in files
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "Checkpoint availability confirms that saved weights exist; it does "
        "not prove clinical accuracy or generalisation to unseen patients."
    )



# -------------------- MRI VIEWER --------------------
elif page == "MRI Viewer":
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from src.preprocessing.patch_dataset import (
        discover_cases,
        load_volume,
    )
    from src.models.attention_unet import AttentionUNet3D
    from src.models.hybrid_quantum_unet import HybridQuantumUNet

    st.subheader("3D MRI Segmentation Lab")
    st.caption(
        "Run a saved research model on one 32×32×32 patch "
        "and compare its prediction with the reference segmentation."
    )

    CLASS_NAMES = {
        0: "Background",
        1: "Label 1",
        2: "Label 2",
        3: "Label 3",
    }
    MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

    @st.cache_resource(show_spinner="Loading model checkpoint...")
    def load_inference_model(model_name):
        if model_name == "Attention U-Net":
            checkpoint_path = BASELINE
            model = AttentionUNet3D(
                in_channels=4, num_classes=4, base_channels=8
            )
        else:
            checkpoint_path = HYBRID
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model = HybridQuantumUNet(
                in_channels=4,
                num_classes=4,
                base_channels=8,
                strategy=checkpoint.get("strategy", "ring"),
            )

        if model_name == "Attention U-Net":
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to("cpu")
        model.eval()
        return model

    @st.cache_data(show_spinner=False)
    def get_case_list():
        return discover_cases()

    cases = get_case_list()

    if not cases:
        st.error("No complete BraTS cases were found. Check the dataset path.")
        st.stop()

    with st.container(border=True):
        st.markdown("#### Inference configuration")
        c1, c2, c3 = st.columns(3)

        with c1:
            model_name = st.selectbox(
                "Segmentation model",
                ["Attention U-Net", "Hybrid Quantum U-Net"],
            )

        with c2:
            case_ids = [case["id"] for case in cases]
            patient_id = st.selectbox(
                "BraTS patient",
                case_ids,
                index=0,
            )

        with c3:
            modality_name = st.selectbox(
                "MRI modality",
                MODALITY_NAMES,
                index=3,
            )

        st.caption(
            "Inference runs on CPU for compatibility. The hybrid model "
            "includes a quantum-circuit simulator and may take longer."
        )

        run_inference = st.button(
            "Run 3D patch inference",
            type="primary",
            use_container_width=True,
        )

    if run_inference:
        if model_name == "Attention U-Net" and not BASELINE.is_file():
            st.error(f"Checkpoint not found: {BASELINE}")
            st.stop()
        if model_name == "Hybrid Quantum U-Net" and not HYBRID.is_file():
            st.error(f"Checkpoint not found: {HYBRID}")
            st.stop()

        case = next(item for item in cases if item["id"] == patient_id)

        try:
            with st.spinner("Loading and normalizing MRI volumes..."):
                image_volume, label_volume = load_volume(case)

            # image_volume: [4, D, H, W]
            # label_volume: [D, H, W]
            _, depth, height, width = image_volume.shape

            # Find the nonzero brain region and center a 32-cubed patch
            # around its bounding-box center.
            brain_mask = np.any(np.abs(image_volume) > 1e-6, axis=0)
            coords = np.where(brain_mask)

            if len(coords[0]) == 0:
                st.error("No nonzero MRI region found in this patient.")
                st.stop()

            center = [
                (int(axis.min()) + int(axis.max())) // 2
                for axis in coords
            ]

            patch_size = 32
            starts = [
                max(0, min(center[0] - 16, depth - patch_size)),
                max(0, min(center[1] - 16, height - patch_size)),
                max(0, min(center[2] - 16, width - patch_size)),
            ]

            # Handle unusually small dimensions by padding.
            image_volume = np.pad(
                image_volume,
                (
                    (0, 0),
                    (0, max(0, patch_size - depth)),
                    (0, max(0, patch_size - height)),
                    (0, max(0, patch_size - width)),
                ),
            )
            label_volume = np.pad(
                label_volume,
                (
                    (0, max(0, patch_size - depth)),
                    (0, max(0, patch_size - height)),
                    (0, max(0, patch_size - width)),
                ),
            )

            d0, h0, w0 = starts
            image_patch = image_volume[
                :, d0:d0+32, h0:h0+32, w0:w0+32
            ].copy()
            label_patch = label_volume[
                d0:d0+32, h0:h0+32, w0:w0+32
            ].copy()

            if image_patch.shape != (4, 32, 32, 32):
                st.error(f"Unexpected MRI patch shape: {image_patch.shape}")
                st.stop()

            model = load_inference_model(model_name)
            input_tensor = torch.from_numpy(
                image_patch.astype(np.float32)
            ).unsqueeze(0)

            with st.spinner("Running model inference..."):
                start_time = time.perf_counter()
                with torch.inference_mode():
                    logits = model(input_tensor)
                    prediction = torch.argmax(logits, dim=1)[0].numpy()
                inference_seconds = time.perf_counter() - start_time

            if prediction.shape != (32, 32, 32):
                st.error(
                    f"Unexpected prediction shape: {prediction.shape}"
                )
                st.stop()

            # Save results for display and downloads across Streamlit reruns.
            
            # Calculate patch-level segmentation metrics.
            class_metrics = {}

            for class_id in (1, 2, 3):
                pred_mask = prediction == class_id
                true_mask = label_patch == class_id

                intersection = int(np.logical_and(
                    pred_mask, true_mask
                ).sum())

                pred_count = int(pred_mask.sum())
                true_count = int(true_mask.sum())

                dice = (
                    2.0 * intersection / (pred_count + true_count)
                    if pred_count + true_count > 0
                    else 1.0
                )

                union = int(np.logical_or(
                    pred_mask, true_mask
                ).sum())

                iou = (
                    intersection / union
                    if union > 0
                    else 1.0
                )

                class_metrics[class_id] = {
                    "dice": dice,
                    "iou": iou,
                    "predicted_voxels": pred_count,
                    "ground_truth_voxels": true_count,
                }

            st.session_state["mri_inference_result"] = {
                "patient_id": patient_id,
                "model_name": model_name,
                "modality_name": modality_name,
                "image_patch": image_patch,
                "label_patch": label_patch,
                "prediction": prediction,
                "class_metrics": class_metrics,
                "inference_seconds": inference_seconds,
                "patch_start": starts,
            }

        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("mri_inference_result")

    if result:
        if (
            result["patient_id"] != patient_id
            or result["model_name"] != model_name
            or result["modality_name"] != modality_name
        ):
            st.info("Configuration changed. Run inference to refresh results.")
            st.stop()

        image_patch = result["image_patch"]
        label_patch = result["label_patch"]
        prediction = result["prediction"]

        st.divider()
        st.markdown("#### Prediction results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Model", result["model_name"])
        m2.metric("Inference time", f'{result["inference_seconds"]:.3f} s')
        m3.metric("Patch dimensions", "32 × 32 × 32")

        slice_index = st.slider(
            "Axial slice within patch",
            min_value=0,
            max_value=31,
            value=16,
        )

        modality_index = MODALITY_NAMES.index(result["modality_name"])
        mri_slice = image_patch[modality_index, slice_index]
        prediction_slice = prediction[slice_index]
        truth_slice = label_patch[slice_index]

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**MRI input**")
            st.image(
                mri_slice,
                clamp=True,
                use_container_width=True,
            )

        with p2:
            st.markdown("**Predicted segmentation**")
            st.image(
                prediction_slice,
                clamp=True,
                use_container_width=True,
            )

        with p3:
            st.markdown("**Ground-truth segmentation**")
            st.image(
                truth_slice,
                clamp=True,
                use_container_width=True,
            )

        st.markdown("#### Predicted voxel distribution")
        
        st.markdown("#### Ground-truth comparison")

        metric_rows = []
        for class_id, metrics in result["class_metrics"].items():
            metric_rows.append({
                "Class": f"Label {class_id}",
                "Dice": round(metrics["dice"], 4),
                "IoU": round(metrics["iou"], 4),
                "Predicted voxels": metrics["predicted_voxels"],
                "Ground-truth voxels": metrics["ground_truth_voxels"],
            })

        st.dataframe(
            pd.DataFrame(metric_rows),
            hide_index=True,
            use_container_width=True,
        )


        voxel_counts = np.bincount(
            prediction.astype(np.int64).ravel(), minlength=4
        )
        total_voxels = int(prediction.size)

        metric_columns = st.columns(4)
        for class_id, column in enumerate(metric_columns):
            count = int(voxel_counts[class_id])
            column.metric(
                CLASS_NAMES[class_id],
                f"{count:,}",
                f"{100 * count / total_voxels:.1f}%",
            )

        st.caption(
            f'Patch origin (D, H, W): {result["patch_start"]}. '
            "Class names are generic label IDs; their clinical meaning "
            "depends on the dataset's annotation convention."
        )

        st.download_button(
            "Download predicted patch (NumPy)",
            data=__import__("io").BytesIO(
                __import__("numpy").asarray(prediction, dtype=np.uint8)
            ).getvalue(),
            file_name=f'{patient_id}_{model_name.replace(" ", "_")}_prediction.npy',
            mime="application/octet-stream",
        )

        st.warning(
            "Research prototype only. These patch-level predictions are "
            "not clinically validated and must not be used for diagnosis "
            "or treatment decisions."
        )
    else:
        st.info(
            "Choose a patient and model, then select **Run 3D patch "
            "inference** to generate a prediction."
        )

    


# -------------------- MODEL COMPARISON --------------------
elif page == "Model Comparison":
    st.subheader("Baseline vs hybrid model")
    st.caption("Exploratory comparison from the saved experiment report.")

    if comparison:
        st.json(comparison, expanded=False)

        def find_metric_rows(obj):
            rows = []
            if isinstance(obj, dict):
                # Handle common per-class metric structures.
                for key, value in obj.items():
                    if isinstance(value, dict):
                        numeric = {
                            k: v for k, v in value.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric and any(
                            term in str(key).lower()
                            for term in ["label", "class", "whole", "core", "enhancing"]
                        ):
                            rows.append({"Class": str(key), **numeric})
                        rows.extend(find_metric_rows(value))
                    elif isinstance(value, list):
                        rows.extend(find_metric_rows(value))
            elif isinstance(obj, list):
                for item in obj:
                    rows.extend(find_metric_rows(item))
            return rows

        rows = find_metric_rows(comparison)
        if rows:
            frame = pd.DataFrame(rows).drop_duplicates()
            st.markdown("#### Detected class-level metrics")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            if "dice" in frame.columns:
                st.plotly_chart(
                    px.bar(frame, x="Class", y="dice",
                           title="Dice score by class",
                           template="plotly_dark", color="Class"),
                    use_container_width=True,
                )
    else:
        st.info("Comparison report not found. Check results/comparison.")

    st.warning(
        "The saved comparison uses a shared patch sample, but the baseline "
        "and hybrid models were trained with different splits. Some comparison "
        "patients overlap the baseline training set, so this is not a strict "
        "held-out generalisation test. Do not interpret it as evidence of "
        "quantum advantage."
    )


# -------------------- TRAINING ANALYTICS --------------------
elif page == "Training Analytics":
    st.subheader("Hybrid training analytics")
    epochs = history.get("epochs", [])

    if not epochs:
        st.info("Extended training history was not found.")
    else:
        rows = []
        for item in epochs:
            train = item.get("train", {})
            val = item.get("validation", {})
            rows.append({
                "Epoch": item.get("epoch"),
                "Training loss": train.get("loss"),
                "Validation loss": val.get("loss"),
                "Training Dice": train.get("mean_foreground_dice"),
                "Validation Dice": val.get("mean_foreground_dice"),
                "Seconds": item.get("seconds"),
            })

        frame = pd.DataFrame(rows)
        a, b = st.columns(2)
        with a:
            st.markdown("#### Loss curves")
            cols = [c for c in ["Training loss", "Validation loss"] if frame[c].notna().any()]
            if cols:
                st.plotly_chart(
                    px.line(frame, x="Epoch", y=cols, markers=True,
                            template="plotly_dark"),
                    use_container_width=True,
                )
        with b:
            st.markdown("#### Foreground Dice")
            cols = [c for c in ["Training Dice", "Validation Dice"] if frame[c].notna().any()]
            if cols:
                st.plotly_chart(
                    px.line(frame, x="Epoch", y=cols, markers=True,
                            template="plotly_dark"),
                    use_container_width=True,
                )

        st.markdown("#### Epoch details")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download training history CSV",
            frame.to_csv(index=False).encode("utf-8"),
            "hybrid_training_history.csv",
            "text/csv",
        )


# -------------------- EVALUATION METRICS --------------------
elif page == "Evaluation Metrics":
    st.subheader("Extended hybrid evaluation")
    st.caption("Saved patch-level evaluation results.")

    if not evaluation:
        st.info("Extended evaluation JSON not found.")
    else:
        st.write(
            f"Evaluation type: {evaluation.get('evaluation_type', 'See report details')}"
        )

        classes = evaluation.get("aggregate_patch_metrics", {})
        if classes:
            frame = pd.DataFrame.from_dict(classes, orient="index")
            frame.index.name = "Class"
            frame = frame.reset_index()

            metric_cols = [c for c in ["dice", "iou", "precision", "recall"] if c in frame.columns]
            metric_cols = [c for c in metric_cols if pd.to_numeric(frame[c], errors="coerce").notna().any()]

            if metric_cols:
                st.plotly_chart(
                    px.bar(
                        frame, x="Class", y=metric_cols, barmode="group",
                        template="plotly_dark", title="Segmentation metrics by class",
                    ),
                    use_container_width=True,
                )

            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download class metrics CSV",
                frame.to_csv(index=False).encode("utf-8"),
                "hybrid_evaluation_metrics.csv",
                "text/csv",
            )
        else:
            st.json(evaluation)

        st.warning(
            "These metrics are based on sampled 32Ã—32Ã—32 patches from eight "
            "validation patients. They are not full-volume evaluation results."
        )

    st.markdown("#### Raw evaluation report")
    st.download_button(
        "Download evaluation JSON",
        json.dumps(evaluation, indent=2).encode("utf-8"),
        "hybrid_ring_extended_evaluation.json",
        "application/json",
        disabled=not bool(evaluation),
    )


# -------------------- REPORTS & FILES --------------------
elif page == "Reports & Files":
    st.subheader("Reports & experiment artefacts")

    paths = [
        ("Baseline checkpoint", BASELINE),
        ("Hybrid extended checkpoint", HYBRID),
        ("Training history JSON", HISTORY),
        ("Extended evaluation JSON", EVALUATION),
        ("Model comparison JSON", COMPARISON),
    ]

    for label, path in paths:
        with st.container(border=True):
            left, middle, right = st.columns([2.2, 1, 1])
            left.markdown(f"**{label}**")
            left.caption(str(path.relative_to(ROOT)))
            middle.write(file_size(path))
            if path.is_file():
                right.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=path.name,
                    key=f"download_{path.name}",
                )
            else:
                right.caption("Unavailable")

    st.markdown("#### Export project summary")
    summary = {
        "project": "BrainTumorQuantum",
        "purpose": "Academic brain MRI segmentation research prototype",
        "baseline_checkpoint_available": BASELINE.is_file(),
        "hybrid_extended_checkpoint_available": HYBRID.is_file(),
        "training_epochs_recorded": len(history.get("epochs", [])),
        "evaluation": evaluation,
        "limitations": [
            "Patch-level metrics are not full-volume metrics.",
            "Experimental models are not clinically validated.",
            "The image upload viewer does not run inference.",
            "The baseline comparison is exploratory and has split overlap.",
        ],
    }
    st.download_button(
        "Download project summary JSON",
        json.dumps(summary, indent=2).encode("utf-8"),
        "brain_tumor_quantum_project_summary.json",
        "application/json",
    )


# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "BRAIN TUMOR QUANTUM Â· Research dashboard Â· "
    "Experimental results only â€” not for diagnosis or treatment decisions."
)

