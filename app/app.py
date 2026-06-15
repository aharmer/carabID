"""
CarabID — Streamlit app (calibrated).

A two-stage YOLO pipeline for identifying New Zealand carabid beetles
to genus from photographs.

  Stage 1 — Detection model (YOLOv11n):
      Locates the beetle (thorax + elytra) in the uploaded image.

  Stage 2 — Classification model (YOLOv11n-cls):
      Classifies the crop to genus (76 genera).

Post-training calibration:
  - Temperature scaling corrects overconfident softmax probabilities.
  - Mahalanobis distance OOD detection flags specimens that do not
    resemble any trained genus.

Run from the carabID root:
    streamlit run app/app.py
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st
from ultralytics import YOLO

# Suppress noisy framework logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Static assets live alongside this file: app/static/
STATIC_DIR = (Path(__file__).parent / "static").resolve()

# Calibration utilities live in calibrated/
sys.path.insert(0, str(Path(__file__).parent.parent / "calibrated"))
from calibration_utils import MahalanobisOOD, TemperatureScaler
from calibrate_yolo_classifier import get_linear_layer


# --------------------------------------------------------------------------- #
#  Image helpers                                                               #
# --------------------------------------------------------------------------- #

def contrast_stretch(img):
    """Robust contrast stretch (2nd – 98th percentile)."""
    arr = np.asarray(img, dtype=np.float32)
    p2  = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 - p2 == 0:
        return img
    arr = np.clip((arr - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_img, imgsz: int = 640) -> torch.Tensor:
    """Convert a PIL image to a normalised YOLO-compatible (1, 3, H, W) tensor."""
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),       # [0,255] → [0.0,1.0]
    ])
    return tf(pil_img).unsqueeze(0)  # (1, 3, H, W)


def format_genus_name(cls_name: str) -> str:
    parts    = cls_name.split("_")
    parts[0] = parts[0].capitalize()
    return " ".join(parts)


def crop_beetle(image: Image.Image, detection_results) -> Image.Image | None:
    """Return the largest detected crop, or None."""
    boxes = detection_results[0].boxes
    if not boxes or len(boxes) == 0:
        return None
    best_box, max_area = None, 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area, best_box = area, (x1, y1, x2, y2)
    return image.crop(best_box) if best_box else None


# --------------------------------------------------------------------------- #
#  Calibrated inference                                                        #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def calibrated_predict(classification_model, tensor: torch.Tensor,
                       scaler, ood_detector):
    """
    Run the YOLO backbone directly to obtain raw logits and embeddings,
    then apply temperature scaling and OOD detection.

    Returns
    -------
    probs      : np.ndarray (C,)  temperature-scaled softmax probabilities
    ood_score  : float            Mahalanobis distance (None if no detector)
    is_novel   : bool             True when ood_score > threshold
    """
    linear       = get_linear_layer(classification_model)
    emb_buffer   = []
    logit_buffer = []

    def _hook(m, inp, out):
        emb_buffer.append(inp[0].detach().cpu())
        logit_buffer.append(out.detach().cpu())

    hook = linear.register_forward_hook(_hook)
    classification_model.model(tensor)   # forward pass; outputs captured via hook
    hook.remove()

    logits    = logit_buffer[0]                                 # (1, C) true logits
    embedding = F.normalize(emb_buffer[0], dim=-1).numpy()[0]  # (D,)

    # Temperature scaling
    if scaler is not None:
        probs = scaler.calibrate(logits.cpu())[0].numpy()
    else:
        probs = torch.softmax(logits.cpu(), dim=-1)[0].numpy()

    # OOD detection
    ood_score = None
    is_novel  = False
    if ood_detector is not None:
        ood_score = float(ood_detector.score(embedding))
        is_novel  = ood_score > ood_detector.threshold

    return probs, ood_score, is_novel


def ood_familiarity_text(ood_score: float, threshold: float) -> str:
    """Plain-language description of how familiar a specimen looks to the model."""
    ratio = ood_score / threshold
    if ratio < 0.5:
        return "Looks very similar to training specimens"
    elif ratio < 0.8:
        return "Looks reasonably similar to training specimens"
    elif ratio < 1.0:
        return "Somewhat unusual — verify identification carefully"
    else:
        return "Looks unlike any trained genus"


def get_top_predictions(probs: np.ndarray, class_names: list[str],
                        top_k: int, conf_threshold: float) -> list[tuple]:
    """Return [(formatted_name, confidence_pct), ...] sorted by confidence."""
    predictions = [
        (format_genus_name(class_names[i]), round(float(p) * 100, 1))
        for i, p in enumerate(probs)
        if p >= conf_threshold
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]


# --------------------------------------------------------------------------- #
#  Model loading                                                               #
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_models():
    """Load YOLO models and calibration artefacts once at startup."""
    detection_path      = STATIC_DIR / "detection.pt"
    classification_path = STATIC_DIR / "classification.pt"
    temperature_path    = STATIC_DIR / "temperature.pt"
    ood_path            = STATIC_DIR / "ood_model.pkl"

    detection_model      = YOLO(str(detection_path))
    classification_model = YOLO(str(classification_path))
    classification_model.model.eval()

    scaler, ood_detector, calibration_loaded = None, None, False
    if temperature_path.exists() and ood_path.exists():
        scaler             = TemperatureScaler.load(temperature_path)
        ood_detector       = MahalanobisOOD.load(ood_path)
        calibration_loaded = True

    return detection_model, classification_model, scaler, ood_detector, calibration_loaded


@st.cache_data
def load_class_table():
    return pd.read_csv(STATIC_DIR / "class_counts.csv")


# --------------------------------------------------------------------------- #
#  App                                                                         #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="CarabID",
    page_icon=str(STATIC_DIR / "carabid_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

detection_model, classification_model, scaler, ood_detector, calibration_loaded = load_models()
df_classes  = load_class_table()
class_names = classification_model.names   # dict {int: str}

tab1, tab2 = st.tabs(["App", "About"])

with tab1:
    with st.sidebar:
        st.image(str(STATIC_DIR / "carabid_icon.png"))
        st.header("Identify a ground beetle")

        if calibration_loaded:
            st.success("✓ Calibrated mode active")
            T = scaler.temperature.item()
            st.caption(
                f"Temperature T = {T:.2f} "
                f"({'overconfidence corrected' if T > 1 else 'calibrated'})"
            )
        else:
            st.warning(
                "Calibration artefacts not found in static/.  "
                "Run `calibrated/calibrate_yolo_classifier.py` to enable "
                "calibration and novelty detection."
            )

        source_imgs = st.file_uploader(
            "Upload images...",
            type=("jpg", "jpeg", "png", "bmp", "webp"),
            accept_multiple_files=True,
        )

        detection_confidence      = st.slider(
            "Detection confidence threshold", 25, 100, 50) / 100
        classification_confidence = st.slider(
            "Classification confidence threshold", 1, 100, 25) / 100
        top_k = st.slider("Top predictions to show", 1, 5, 3)

    st.title("CarabID")
    st.caption("Upload photos of ground beetles.")
    st.caption("Then click :blue[Identify] and check the results.")

    if st.sidebar.button("Identify"):
        if not source_imgs:
            st.error("Please upload at least one image first!")
        else:
            with st.spinner("Analysing images..."):
                COLS_PER_ROW = 4
                cols = None

                for i, source_img in enumerate(source_imgs):
                    if i % COLS_PER_ROW == 0:
                        cols = st.columns(COLS_PER_ROW)
                    with cols[i % COLS_PER_ROW]:
                        with st.container(border=True):
                            st.caption(source_img.name)
                            try:
                                # 1. Load & preprocess
                                uploaded = PIL.Image.open(source_img)
                                if uploaded.mode != "RGB":
                                    uploaded = uploaded.convert("RGB")
                                prepped = contrast_stretch(
                                    uploaded.resize((640, 640))
                                )

                                # 2. Detection
                                det_results = detection_model.predict(
                                    prepped,
                                    conf=detection_confidence,
                                    verbose=False,
                                )

                                if (not det_results[0].boxes
                                        or len(det_results[0].boxes) == 0):
                                    st.image(uploaded, use_container_width=True)
                                    st.warning("No beetle detected")
                                    continue

                                det_plot = det_results[0].plot()[:, :, ::-1]
                                cropped  = crop_beetle(uploaded, det_results)

                                if cropped is None:
                                    st.image(det_plot, use_container_width=True)
                                    st.error("Crop failed")
                                    continue

                                # 3. Calibrated classification
                                tensor = pil_to_tensor(
                                    cropped.resize((640, 640))
                                )
                                probs, ood_score, is_novel = calibrated_predict(
                                    classification_model, tensor,
                                    scaler, ood_detector,
                                )

                                top_preds = get_top_predictions(
                                    probs, class_names, top_k,
                                    classification_confidence,
                                )

                                # 4. Display
                                st.image(det_plot, use_container_width=True)

                                if is_novel:
                                    st.error(
                                        "⚠ **Novel genus** — this specimen does "
                                        "not resemble any trained class."
                                    )
                                    if ood_score is not None:
                                        st.caption(
                                            f"Familiarity: {ood_familiarity_text(ood_score, ood_detector.threshold)}  \n"
                                            f"_(novelty score {ood_score:.0f} exceeds limit of {ood_detector.threshold:.0f})_"
                                        )
                                    with st.expander("Best guess anyway"):
                                        if top_preds:
                                            for rank, (genus, conf) in enumerate(top_preds, 1):
                                                st.text(f"{rank}. {genus} ({conf}%)")
                                        else:
                                            st.text("No predictions above threshold")

                                elif top_preds:
                                    top_genus, top_conf = top_preds[0]
                                    st.markdown(f"**{top_genus}**")
                                    st.progress(top_conf / 100)
                                    st.caption(f"{top_conf}% confidence (calibrated)")
                                    if ood_score is not None:
                                        st.caption(
                                            f"Familiarity: {ood_familiarity_text(ood_score, ood_detector.threshold)}  \n"
                                            f"_(novelty score {ood_score:.0f} / limit {ood_detector.threshold:.0f})_"
                                        )
                                    with st.expander("Details"):
                                        for rank, (genus, conf) in enumerate(top_preds, 1):
                                            st.text(f"{rank}. {genus} ({conf}%)")

                                else:
                                    st.warning("Low confidence — no predictions above threshold")

                            except Exception as ex:
                                st.error(f"Error: {ex}")

with tab2:
    st.header("About the project")
    st.markdown(
        "This tool was developed using YOLOv11 models from the Ultralytics "
        "Python library (Jocher et al., 2023). A two-stage detection and "
        "classification pipeline is used. The first stage detects the beetle "
        "thorax and elytra; the second classifies it to genus. "
        "Full details can be found in the associated publication: "
        "Gong, Harmer & Ward (in review)."
    )
    st.image(str(STATIC_DIR / "example_image.jpg"), width=350)

    st.subheader("Interpreting results")

    st.markdown("#### Detection confidence")
    st.markdown(
        "Before the beetle can be identified, the app must first *find* it in "
        "the photograph. This is done by the detection model, which scans the "
        "image and draws a bounding box around the beetle's thorax and elytra.\n\n"
        "**Detection confidence** reflects how certain the detection model is "
        "that a real beetle is present in a given bounding box — not what genus "
        "it belongs to. A detection confidence of 80% means the model is 80% "
        "sure it has located a beetle, regardless of species.\n\n"
        "The **Detection confidence threshold** slider in the sidebar controls "
        "the minimum detection confidence required for a bounding box to be "
        "accepted. Raising the threshold reduces false detections (e.g. shadows "
        "or debris mistaken for a beetle) but may miss real beetles in poor-quality "
        "images. Lowering it catches more detections but may introduce false "
        "positives. The default of 50% works well for clear, well-lit photographs."
    )

    st.markdown("#### Classification confidence")
    st.markdown(
        "Once a beetle has been detected and cropped, the classification model "
        "compares the crop against all 76 genera it has been trained on.\n\n"
        "**Classification confidence** answers a different question from "
        "detection confidence: *'Of the 76 genera the model knows, which one "
        "does this look most like, and by how much?'* It is a relative comparison "
        "— the model scores every known genus and reports how much of its "
        "'vote' went to the top candidate. A high classification confidence "
        "means the model strongly prefers one genus over all others, but it "
        "does not on its own tell you whether the specimen actually belongs to "
        "any of those 76 genera.\n\n"
        "Confidence values are adjusted using *temperature scaling* (see below) "
        "so that, for example, an 80% confidence should correspond to the model "
        "being correct roughly 80% of the time.\n\n"
        "The **Classification confidence threshold** slider controls the minimum "
        "classification confidence required for a prediction to be displayed. "
        "Raising it hides uncertain predictions; lowering it shows more candidates, "
        "which can be useful when the top prediction is genuinely ambiguous."
    )

    st.markdown("#### Familiarity score")
    st.markdown(
        "The familiarity check answers a different question: *'Does this "
        "specimen look as close to the training photographs as a typical "
        "specimen should?'* It measures how far the specimen's visual features "
        "sit from the cloud of training examples for the nearest genus — in "
        "absolute terms, not relative to other genera.\n\n"
        "This means confidence and familiarity can appear to give conflicting "
        "signals, and that is intentional:\n\n"
        "- **High confidence + high familiarity** — the model recognises this "
        "specimen clearly and it looks like a typical training example. "
        "The identification is likely reliable.\n"
        "- **High confidence + lower familiarity** — the model still strongly "
        "favours one genus over all others, but the specimen sits somewhat "
        "further from the training examples than usual. This can happen with "
        "unusual viewing angles, damaged specimens, unusual lighting, or "
        "natural variation within the genus. Treat the result as a strong lead "
        "but verify it against reference material before finalising.\n"
        "- **Novel genus warning** — the specimen falls so far outside the "
        "training distribution that no reliable identification can be made. "
        "The model will still show a best guess, but this should be treated "
        "with caution."
    )

    st.subheader("Technical notes")
    st.markdown(
        "#### Confidence calibration (temperature scaling)\n"
        "Raw neural network classifiers tend to be overconfident — they often "
        "report very high probabilities (e.g. 99.9%) even when they are wrong. "
        "CarabID corrects for this using *temperature scaling*: the raw scores "
        "are divided by a constant (T = {T:.2f}) fitted on a held-out "
        "validation set, which spreads the probabilities to better reflect the "
        "model's true accuracy. Calibration does not change which genus is "
        "ranked first — it only makes the confidence percentages more "
        "trustworthy.\n\n"
        "#### Familiarity / novelty detection (Mahalanobis distance)\n"
        "Each image is converted to a compact feature vector by the model's "
        "backbone. During training, the average feature vector (*centroid*) for "
        "each genus is recorded, along with how spread out the vectors are "
        "across all genera. At inference time, the distance from the specimen's "
        "feature vector to the nearest centroid is computed — this is the "
        "*novelty score*. If the score exceeds a threshold (set so that only "
        "5% of genuine training-distribution specimens are incorrectly flagged), "
        "the specimen is considered outside the known distribution and a novelty "
        "warning is shown.".format(T=scaler.temperature.item() if scaler else 1.0)
    )

    st.subheader("Trained genera")
    st.markdown(
        "The classification model was trained on the genera listed below. "
        "Image counts are for the original dataset before augmentation."
    )
    st.dataframe(df_classes, height=600)
