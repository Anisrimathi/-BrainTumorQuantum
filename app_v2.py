
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

def explain_region(label, voxel_count, volume_ml=None):
    explanations = {
        1: (
            "Necrotic/non-enhancing tumor-core region",
            "The model marked this region as Label 1 in the project's "
            "segmentation scheme. This is an image-based prediction, "
            "not confirmation of tissue type or diagnosis."
        ),
        2: (
            "Edema-associated region",
            "The model marked this region as Label 2. Edema refers to "
            "swelling associated with fluid in tissue. The prediction "
            "must be checked against the MRI by a qualified clinician."
        ),
        3: (
            "Enhancing tumor region",
            "The model marked this region as Label 3. Enhancement is an "
            "MRI imaging characteristic; this result alone cannot "
            "establish tumor grade, cancer stage, or severity."
        ),
    }

    name, explanation = explanations.get(
        int(label),
        ("Background", "No specific tumor-region interpretation.")
    )

    result = f"{name}\n{explanation}\nPredicted voxel count: {int(voxel_count):,}"

    if volume_ml is not None:
        result += f"\nEstimated volume: {float(volume_ml):.2f} mL"

    return result


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

BASELINE = CHECKPOINTS / "attention_unet_pilot_best.pth"
HYBRID = CHECKPOINTS / "hybrid_quantum_unet_ring_final_v2.pth"

HISTORY = RESULTS / "hybrid_ring_extended_history.json"
EVALUATION = RESULTS / "hybrid_ring_extended_evaluation.json"
COMPARISON = RESULTS / "comparison" / "baseline_vs_hybrid_full_volume_v2.json"

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
            "Full-Volume Analysis",
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

    



# -------------------- FULL-VOLUME ANALYSIS --------------------
elif page == "Full-Volume Analysis":
    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("Full-volume segmentation analysis")
    st.caption(
        "Saved predictions for one pilot patient. "
        "Results are for research use and are not clinically validated."
    )

    pred_dir = RESULTS / "visualizations" / "baseline_vs_hybrid_v2"
    npz_path = pred_dir / "BraTS-GLI-00036-001_predictions.npz"
    png_path = pred_dir / "BraTS-GLI-00036-001_slice_81_comparison.png"

    if not npz_path.exists():
        st.error(f"Prediction file not found: {npz_path}")
    else:
        with np.load(npz_path, allow_pickle=False) as saved:
            gt = saved["ground_truth"].copy()
            baseline = saved["baseline_prediction"].copy()
            hybrid = saved["hybrid_prediction"].copy()
            default_slice = int(saved["flair_slice_index"].item())

        if gt.shape != baseline.shape or gt.shape != hybrid.shape:
            st.error("The saved masks have inconsistent dimensions.")
        else:
            st.write(
                f"**Patient:** BraTS-GLI-00036-001  |  "
                f"**Volume shape:** {gt.shape}  |  "
                f"**Slices:** {gt.shape[2]}"
            )

            if png_path.exists():
                with st.expander("Original four-panel MRI comparison", expanded=True):
                    st.image(str(png_path), caption="Saved FLAIR, ground truth, baseline and hybrid comparison.")

            slice_idx = st.slider(
                "Axial slice index",
                min_value=0,
                max_value=gt.shape[2] - 1,
                value=min(default_slice, gt.shape[2] - 1),
                key="full_volume_slice",
            )

            labels = [1, 2, 3]
            label_names = {
                1: "Label 1",
                2: "Label 2",
                3: "Label 3",
            }

            # Load the original FLAIR MRI for this comparison patient.
            import nibabel as nib
            from matplotlib.colors import ListedColormap

            flair_matches = list(
                (ROOT / "dataset").rglob(
                    "BraTS-GLI-00036-001-flair.nii.gz"
                )
            )
            flair_matches += list(
                (ROOT / "dataset").rglob(
                    "BraTS-GLI-00036-001-flair.nii"
                )
            )

            if not flair_matches:
                st.error(
                    "Original FLAIR file not found. Showing mask-only views."
                )
                flair_slice = None
            else:
                flair_img = nib.load(str(flair_matches[0]))
                flair_data = flair_img.get_fdata(dtype="float32")

                if tuple(flair_data.shape[:3]) != tuple(gt.shape):
                    st.error(
                        "FLAIR dimensions do not match the saved masks. "
                        "Showing mask-only views."
                    )
                    flair_slice = None
                else:
                    flair_slice = flair_data[:, :, slice_idx].T
                    low, high = np.percentile(
                        flair_slice[np.isfinite(flair_slice)], [1, 99]
                    )
                    if high > low:
                        flair_slice = np.clip(
                            (flair_slice - low) / (high - low), 0, 1
                        )

            # Consistent colors for segmentation labels across all panels.
            label_cmap = ListedColormap([
                (0.0, 0.0, 0.0, 0.0),
                (0.1, 0.9, 0.3, 0.65),
                (0.1, 0.45, 1.0, 0.65),
                (1.0, 0.15, 0.15, 0.70),
            ])

            cols = st.columns(3)
            panels = [
                (gt, "Ground truth"),
                (baseline, "Baseline prediction"),
                (hybrid, "Hybrid prediction"),
            ]

            for col, (mask, title) in zip(cols, panels):
                fig, ax = plt.subplots(figsize=(5, 5))

                if flair_slice is not None:
                    ax.imshow(
                        flair_slice,
                        cmap="gray",
                        origin="lower",
                        interpolation="nearest",
                    )

                ax.imshow(
                    mask[:, :, slice_idx].T,
                    cmap=label_cmap,
                    vmin=0,
                    vmax=3,
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_title(title)
                ax.set_axis_off()
                col.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.caption(
                "Overlay legend: Label 1 = green, Label 2 = blue, "
                "Label 3 = red. Colors identify segmentation classes, "
                "not tumor severity."
            )

            st.markdown("### Full-volume voxel counts")
            st.caption(
                "Counts include every voxel in the saved 3D mask. "
                "Physical volume in mL is not reported because voxel-spacing metadata "
                "has not been verified for this saved prediction file."
            )

            # Read physical voxel spacing for this exact patient.
            import nibabel as nib

            seg_matches = list(
                (ROOT / "dataset").rglob(
                    "BraTS-GLI-00036-001-seg.nii.gz"
                )
            )
            seg_matches += list(
                (ROOT / "dataset").rglob(
                    "BraTS-GLI-00036-001-seg.nii"
                )
            )

            if not seg_matches:
                st.error(
                    "Cannot calculate physical volumes: the comparison "
                    "patient's NIfTI segmentation file was not found."
                )
                voxel_ml = None
            else:
                seg_img = nib.load(str(seg_matches[0]))
                spacing = seg_img.header.get_zooms()[:3]
                if tuple(seg_img.shape[:3]) != tuple(gt.shape):
                    st.error(
                        "NIfTI segmentation dimensions do not match the "
                        "saved masks. Physical volumes are disabled."
                    )
                    voxel_ml = None
                else:
                    voxel_ml = (
                        float(spacing[0])
                        * float(spacing[1])
                        * float(spacing[2])
                        / 1000.0
                    )
                    st.caption(
                        f"Verified voxel spacing: "
                        f"{spacing[0]:g} ? {spacing[1]:g} ? "
                        f"{spacing[2]:g} mm; "
                        f"voxel volume: {voxel_ml:.6f} mL."
                    )

            rows = []
            for label in labels:
                gt_count = int(np.count_nonzero(gt == label))
                base_count = int(np.count_nonzero(baseline == label))
                hybrid_count = int(np.count_nonzero(hybrid == label))

                def dice(a, b):
                    a_count = int(np.count_nonzero(a == label))
                    b_count = int(np.count_nonzero(b == label))
                    if a_count + b_count == 0:
                        return 1.0
                    overlap = int(np.count_nonzero((a == label) & (b == label)))
                    return 2.0 * overlap / (a_count + b_count)

                row = {
                    "Class": label_names[label],
                    "Ground-truth voxels": gt_count,
                    "Baseline voxels": base_count,
                    "Hybrid voxels": hybrid_count,
                    "Baseline Dice": round(dice(gt, baseline), 4),
                    "Hybrid Dice": round(dice(gt, hybrid), 4),
                }

                if voxel_ml is not None:
                    gt_ml = gt_count * voxel_ml
                    base_ml = base_count * voxel_ml
                    hybrid_ml = hybrid_count * voxel_ml
                    row.update({
                        "Ground-truth volume (mL)": round(gt_ml, 4),
                        "Baseline volume (mL)": round(base_ml, 4),
                        "Hybrid volume (mL)": round(hybrid_ml, 4),
                        "Baseline absolute volume difference (mL)": round(
                            abs(base_ml - gt_ml), 4
                        ),
                        "Hybrid absolute volume difference (mL)": round(
                            abs(hybrid_ml - gt_ml), 4
                        ),
                    })

                rows.append(row)

            st.dataframe(rows, use_container_width=True, hide_index=True)

            # Visual comparison of segmentation performance.
            import pandas as pd
            import plotly.express as px

            metrics_df = pd.DataFrame(rows)

            st.markdown("### Baseline vs hybrid Dice comparison")
            dice_df = metrics_df.melt(
                id_vars=["Class"],
                value_vars=["Baseline Dice", "Hybrid Dice"],
                var_name="Model",
                value_name="Dice score",
            )
            dice_fig = px.bar(
                dice_df,
                x="Class",
                y="Dice score",
                color="Model",
                barmode="group",
                range_y=[0, 1],
                title="Per-class Dice score (higher is better)",
                hover_data={"Dice score": ":.4f"},
            )
            st.plotly_chart(dice_fig, use_container_width=True)

            volume_columns = [
                "Ground-truth volume (mL)",
                "Baseline volume (mL)",
                "Hybrid volume (mL)",
            ]

            if all(column in metrics_df.columns for column in volume_columns):
                st.markdown("### Ground-truth vs predicted volume")
                volume_df = metrics_df.melt(
                    id_vars=["Class"],
                    value_vars=volume_columns,
                    var_name="Segmentation",
                    value_name="Volume (mL)",
                )
                volume_fig = px.bar(
                    volume_df,
                    x="Class",
                    y="Volume (mL)",
                    color="Segmentation",
                    barmode="group",
                    title="Predicted and reference class volumes",
                    hover_data={"Volume (mL)": ":.4f"},
                )
                st.plotly_chart(volume_fig, use_container_width=True)

            st.caption(
                "Interpretation: Dice measures overlap with the reference mask. "
                "Higher is better. Volume agreement alone does not establish "
                "spatial accuracy. Results shown here are from one pilot patient."
            )

            if voxel_ml is not None:
                csv_data = __import__("pandas").DataFrame(rows).to_csv(index=False)
                st.download_button(
                    "Download volume and Dice report (.csv)",
                    data=csv_data.encode("utf-8"),
                    file_name="BraTS-GLI-00036-001_volume_metrics.csv",
                    mime="text/csv",
                )

                # Build a downloadable PDF research report.
                import io
                from datetime import datetime
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import landscape, A4
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import mm
                from reportlab.platypus import (
                    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                )

                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(
                    pdf_buffer,
                    pagesize=landscape(A4),
                    rightMargin=15 * mm,
                    leftMargin=15 * mm,
                    topMargin=15 * mm,
                    bottomMargin=15 * mm,
                )
                styles = getSampleStyleSheet()
                story = [
                    Paragraph("BrainTumorQuantum", styles["Title"]),
                    Paragraph(
                        "Full-Volume Segmentation Research Report",
                        styles["Heading2"],
                    ),
                    Spacer(1, 5 * mm),
                    Paragraph(
                        "<b>Patient:</b> BraTS-GLI-00036-001",
                        styles["BodyText"],
                    ),
                    Paragraph(
                        f"<b>Generated:</b> "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        styles["BodyText"],
                    ),
                    Paragraph(
                        f"<b>Mask dimensions:</b> {gt.shape}",
                        styles["BodyText"],
                    ),
                    Spacer(1, 5 * mm),
                    Paragraph("Segmentation metrics", styles["Heading2"]),
                ]

                headers = list(rows[0].keys()) if rows else []
                table_data = []

                # Patient-Friendly Results Explanation.
                story.append(Spacer(1, 12))
                story.append(
                    Paragraph("Patient-Friendly Results Explanation", styles["Heading1"])
                )
                story.append(
                    Paragraph(
                        "The following regions were predicted by the experimental MRI "
                        "segmentation model. These are estimates from image analysis "
                        "and are not a confirmed diagnosis.",
                        styles["BodyText"],
                    )
                )
                story.append(Spacer(1, 8))

                # Dynamic patient-friendly interpretation from this case's actual rows.
                patient_label_explanations = {
                    1: ("Label 1 — necrotic/non-enhancing tumor-core region",
                        "In this project's label mapping, Label 1 marks a region annotated as necrotic or non-enhancing tumor core. This image-based prediction does not confirm tissue type."),
                    2: ("Label 2 — edema-associated region",
                        "In this project's label mapping, Label 2 marks an edema-associated region. Edema means swelling related to fluid in tissue; the model alone cannot confirm its cause."),
                    3: ("Label 3 — enhancing tumor region",
                        "In this project's label mapping, Label 3 marks an enhancing-tumor region. Enhancement is an MRI imaging feature and does not establish tumor grade, cancer stage, or severity."),
                }
                story.append(Paragraph("What the region measurements mean", styles["Heading2"]))
                story.append(Paragraph(
                    "The following values compare the saved reference mask with the baseline and hybrid predictions for this pilot case. A voxel is a 3D image element. Volume is estimated only when image spacing is available. These are segmentation measurements, not a clinical tumor level.",
                    styles["BodyText"]))
                story.append(Spacer(1, 3 * mm))

                for label_id in (1, 2, 3):
                    row = next((item for item in rows if item.get("Class") == label_names[label_id]), None)
                    heading, explanation = patient_label_explanations[label_id]
                    story.append(Paragraph(heading, styles["Heading3"]))
                    story.append(Paragraph(explanation, styles["BodyText"]))
                    if row:
                        story.append(Paragraph(
                            f"Reference-mask count: {int(row.get('Ground-truth voxels', 0)):,} voxels. "
                            f"Baseline prediction: {int(row.get('Baseline voxels', 0)):,} voxels. "
                            f"Hybrid prediction: {int(row.get('Hybrid voxels', 0)):,} voxels.",
                            styles["BodyText"]))
                        if "Ground-truth volume (mL)" in row:
                            story.append(Paragraph(
                                f"Estimated volumes: reference {float(row['Ground-truth volume (mL)']):.3f} mL; "
                                f"baseline {float(row['Baseline volume (mL)']):.3f} mL; "
                                f"hybrid {float(row['Hybrid volume (mL)']):.3f} mL.",
                                styles["BodyText"]))
                        if row.get("Baseline Dice") is not None and row.get("Hybrid Dice") is not None:
                            story.append(Paragraph(
                                f"Dice overlap with the reference mask: baseline {float(row['Baseline Dice']):.3f}; "
                                f"hybrid {float(row['Hybrid Dice']):.3f}. Dice ranges from 0 (no overlap) to 1 "
                                f"(perfect overlap). It measures agreement with the reference annotation, not medical correctness.",
                                styles["BodyText"]))
                    story.append(Spacer(1, 2 * mm))

                story.append(Paragraph("How to understand these results", styles["Heading2"]))
                story.append(Paragraph(
                    "Differences between the reference, baseline, and hybrid counts mean the masks do not mark exactly the same number of voxels. A larger predicted region is not proof of a more severe tumor, and a smaller region is not proof of a less severe condition. These results cannot be classified as low, medium, or high tumor severity.",
                    styles["BodyText"]))
                story.append(Paragraph(
                    "<b>Suggested next step:</b> Ask a qualified radiologist or treating clinician to interpret the original MRI and official radiology report. This experimental project is not clinically validated and must not be used alone for diagnosis or treatment decisions.",
                    styles["BodyText"]))
                story.append(Spacer(1, 5 * mm))

                # Landscape A4 printable width: 297 mm minus 30 mm margins.
                available_width = landscape(A4)[0] - 30 * mm

                from reportlab.lib.styles import ParagraphStyle
                from reportlab.platypus import KeepTogether

                cell_style = ParagraphStyle(
                    "MetricCell",
                    parent=styles["BodyText"],
                    fontName="Helvetica",
                    fontSize=6.5,
                    leading=8,
                    spaceAfter=0,
                )
                header_style = ParagraphStyle(
                    "MetricHeader",
                    parent=cell_style,
                    fontName="Helvetica-Bold",
                    fontSize=6,
                    leading=7,
                    textColor=colors.white,
                )

                # Shorten visual labels while preserving the full CSV column names.
                display_headers = {
                    "Class": "Class",
                    "Ground-truth voxels": "GT voxels",
                    "Baseline voxels": "Baseline voxels",
                    "Hybrid voxels": "Hybrid voxels",
                    "Baseline Dice": "Baseline Dice",
                    "Hybrid Dice": "Hybrid Dice",
                    "Ground-truth volume (mL)": "GT volume (mL)",
                    "Baseline volume (mL)": "Baseline volume (mL)",
                    "Hybrid volume (mL)": "Hybrid volume (mL)",
                    "Baseline absolute volume difference (mL)": "Baseline abs. diff. (mL)",
                    "Hybrid absolute volume difference (mL)": "Hybrid abs. diff. (mL)",
                }

                if headers:
                    table_data.append([
                        Paragraph(
                            display_headers.get(h, h).replace("&", "&amp;"),
                            header_style,
                        )
                        for h in headers
                    ])

                    for row in rows:
                        table_data.append([
                            Paragraph(
                                str(row.get(h, "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                                cell_style,
                            )
                            for h in headers
                        ])

                if table_data and headers:
                    # Allocate widths based on content type, then scale to page width.
                    weights = []
                    for header in headers:
                        name = header.lower()
                        if "class" in name:
                            weights.append(0.75)
                        elif "dice" in name:
                            weights.append(0.85)
                        elif "voxels" in name:
                            weights.append(1.05)
                        elif "volume" in name or "difference" in name:
                            weights.append(1.25)
                        else:
                            weights.append(1.0)

                    total_weight = sum(weights)
                    col_widths = [
                        available_width * weight / total_weight
                        for weight in weights
                    ]

                    report_table = Table(
                        table_data,
                        colWidths=col_widths,
                        repeatRows=1,
                        hAlign="LEFT",
                    )
                    report_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23395B")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#EEF2F7")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.extend([report_table, Spacer(1, 5 * mm)])

                story.extend([
                    Paragraph("Interpretation and limitations", styles["Heading2"]),
                    Paragraph(
                        "This report summarizes saved segmentation masks for one "
                        "pilot patient. Dice scores compare each predicted class "
                        "with the saved ground-truth mask. Volume estimates are "
                        "included only when the patient's voxel spacing and mask "
                        "dimensions are verified. These outputs are experimental, "
                        "have not been clinically validated, and must not be used "
                        "as a standalone diagnostic or treatment decision.",
                        styles["BodyText"],
                    ),
                ])

                doc.build(story)
                pdf_buffer.seek(0)

                st.download_button(
                    "Generate / Download PDF Research Report",
                    data=pdf_buffer.getvalue(),
                    file_name="BrainTumorQuantum_full_volume_report.pdf",
                    mime="application/pdf",
                    key="download_full_volume_pdf",
                )

            st.markdown("### Download prediction masks")
            import io

            for mask_name, mask in [
                ("ground_truth", gt),
                ("baseline_prediction", baseline),
                ("hybrid_prediction", hybrid),
            ]:
                buffer = io.BytesIO()
                np.save(buffer, mask, allow_pickle=False)
                st.download_button(
                    label=f"Download {mask_name}.npy",
                    data=buffer.getvalue(),
                    file_name=f"{mask_name}.npy",
                    mime="application/octet-stream",
                    key=f"download_{mask_name}",
                )

            # Download the original complete NPZ archive.
            st.download_button(
                "Download all saved 3D masks (.npz)",
                data=npz_path.read_bytes(),
                file_name=npz_path.name,
                mime="application/octet-stream",
            )

            st.info(
                "Dice scores are computed against the saved ground-truth mask. "
                "This is a single-patient pilot comparison, not a clinical performance claim."
            )


# -------------------- MODEL COMPARISON --------------------
elif page == "Model Comparison":
    import pandas as pd
    import plotly.express as px

    st.subheader("Baseline vs hybrid model")
    st.caption("Whole-volume segmentation comparison using the saved pilot evaluation.")

    if comparison:
        summary = comparison.get("summary", {})
        baseline = summary.get("baseline_patient_macro_average", {})
        hybrid = summary.get("hybrid_patient_macro_average", {})

        st.info(
            f"Evaluated patients: {comparison.get('evaluated_count', 0)} / "
            f"{comparison.get('heldout_count', 0)} held-out-list patients. "
            "These are preliminary pilot results, not clinical validation."
        )

        if baseline and hybrid:
            metric_rows = []
            regions = list(dict.fromkeys(list(baseline.keys()) + list(hybrid.keys())))

            for region in regions:
                b = baseline.get(region, {})
                h = hybrid.get(region, {})
                row = {"Region": region}
                for metric in ("dice", "iou", "precision", "recall"):
                    row[f"Baseline {metric.title()}"] = b.get(metric)
                    row[f"Hybrid {metric.title()}"] = h.get(metric)
                metric_rows.append(row)

            frame = pd.DataFrame(metric_rows)
            st.markdown("### Five-patient macro-average metrics")
            st.dataframe(
                frame.round(4),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Dice score comparison")
            dice_frame = frame.melt(
                id_vars=["Region"],
                value_vars=["Baseline Dice", "Hybrid Dice"],
                var_name="Model",
                value_name="Dice",
            )
            dice_frame = dice_frame.dropna(subset=["Dice"])
            st.plotly_chart(
                px.bar(
                    dice_frame,
                    x="Region",
                    y="Dice",
                    color="Model",
                    barmode="group",
                    range_y=[0, 1],
                    title="Patient macro-average Dice (higher is better)",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

            st.markdown("### Precision vs recall")
            pr_frame = frame.melt(
                id_vars=["Region"],
                value_vars=[
                    "Baseline Precision", "Hybrid Precision",
                    "Baseline Recall", "Hybrid Recall",
                ],
                var_name="Metric",
                value_name="Score",
            )
            pr_frame = pr_frame.dropna(subset=["Score"])
            pr_frame["Model"] = pr_frame["Metric"].str.split().str[0]
            pr_frame["Measure"] = pr_frame["Metric"].str.split().str[1]
            st.plotly_chart(
                px.bar(
                    pr_frame,
                    x="Region",
                    y="Score",
                    color="Model",
                    barmode="group",
                    facet_col="Measure",
                    range_y=[0, 1],
                    title="Precision and recall by region",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

            st.caption(
                "Interpretation: higher recall means more reference-positive voxels "
                "were detected; lower precision indicates more false-positive "
                "predictions. High recall alone does not mean better segmentation."
            )
        else:
            st.error(
                "The saved comparison does not contain the expected macro-average "
                "keys. Check results/comparison/baseline_vs_hybrid_full_volume_v2.json."
            )
            st.json(summary, expanded=True)
    else:
        st.error("Comparison report not found. Check results/comparison/.")

    st.warning(
        "Research limitation: only a pilot subset has been evaluated. The baseline "
        "and hybrid models used different training splits, and patient overlap "
        "with baseline training may remain. Do not interpret these results as "
        "evidence of quantum advantage or clinical performance."
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

