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

# The novelty threshold lives on the calibration artefact, so every script
# agrees on one value.  It is set to 2000 rather than the 5 % FPR fit of 1784:
# validated against genuinely novel genera (imgs/ood_test/, see
# scripts/evaluate_novelty_detection.py), 2000 holds the same novel-genus
# recall (93 %) while halving false alarms on known specimens (8 % -> 4 %).
#
# It cannot be raised much further to accommodate low-magnification field
# photos: recall collapses to 73 % at 2331 and 13 % at 3067.  That gap has to
# be closed in training (resolution-jitter augmentation), not at the threshold.

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


def detect_best_orientation(image: Image.Image, model, conf: float):
    """Detect on the image and on a 90°-rotated copy, keeping whichever view
    the detector is more confident about.

    Both models were trained only on landscape specimens (no rotation
    augmentation), so an uncorrected portrait photo loses about a quarter of
    detections and most of the classification accuracy.  The rotation must be
    chosen *before* trusting any box: on a portrait image the box is often
    missing altogether, and poorly localised when present, so its shape cannot
    be used to infer the orientation.

    Returns (view, det_results, rotation).  Boxes are measured against `view`,
    so any crop must be taken from `view` — not from the original image.
    """
    best = None
    for rot in (0, 90):
        view    = image.rotate(rot, expand=True) if rot else image
        results = model.predict(contrast_stretch(view), conf=conf, verbose=False)
        boxes   = results[0].boxes
        if not boxes or len(boxes) == 0:
            continue
        top_conf = float(max(b.conf[0] for b in boxes))
        if best is None or top_conf > best[0]:
            best = (top_conf, view, results, rot)
    if best is None:
        return image, None, 0
    return best[1], best[2], best[3]


def crop_beetle(image: Image.Image, detection_results) -> Image.Image | None:
    """Return the largest detected crop, or None.

    Box coordinates are in the coordinate space of the image detection ran on,
    so `image` must be that same image (not a differently sized copy).
    """
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
        return "Unusual for the training set — verify carefully"


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


def render_guidance():
    """Photography guidance.

    The figures quoted here are measured from the training set and the
    deployed model, not rules of thumb: every training image is landscape
    with the beetle body spanning ~300 px and filling ~17 % of the frame.
    Portrait photos are corrected automatically by detect_best_orientation
    (92 % top-1 vs 98 % for a landscape original), so orientation is stated
    as a preference rather than a requirement.
    """
    with st.container(border=True):
        st.markdown("#### How to photograph your specimen")
        st.caption(
            "The model was trained on museum specimen photographs — the closer "
            "your photo matches them, the more reliable the identification."
        )
        st.image(str(STATIC_DIR / "guidance_examples.png"),
                 use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown(
                "**View** — dorsal (straight down onto the beetle's back), "
                "specimen flat and square to the camera.\n\n"
                "**Orientation** — **landscape** is best, long axis "
                "horizontal, head either way. Portrait photos are rotated "
                "automatically, but a landscape original is still a little "
                "more reliable.\n\n"
                "**Framing** — whole beetle in frame with a small margin, the "
                "body (thorax + elytra) should fill roughly a fifth of "
                "the image."
            )
        with right:
            st.markdown(
                "**Resolution** — width at least **1200 px**, with the beetle's "
                "body spanning **600 px or more**. Larger is better.\n\n"
                "**Background** — plain and uncluttered (white or pale), no "
                "leaf litter, soil, or fingers in shot.\n\n"
                "**Focus & lighting** — sharp and evenly lit, surface details visible, with minimal "
                "shadows and glare."
            )
        st.caption(
            "Field photos of live beetles are identified far less reliably and "
            "will raise the *unusual image or novel genus* warning — treat those "
            "results as tentative and verify them against reference material."
        )


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

# Streamlit defaults the popover panel to max-width 704px and right-aligns it
# to its trigger, so a popover in a left-hand grid column renders off the page.
# Constraining it to roughly card width keeps it on screen.  The triggers are
# all warnings, so they get an amber treatment to read as such at a glance.
st.markdown(
    """
    <style>
      [data-testid="stPopoverBody"] { max-width: 300px !important; }
      button[data-testid="stPopoverButton"] {
          background-color: #fff4e5;
          border: 1px solid #ffb74d;
          color: #8a4b00;
      }
      button[data-testid="stPopoverButton"]:hover {
          background-color: #ffe8cc;
          border-color: #fb8c00;
          color: #6d3b00;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

detection_model, classification_model, scaler, ood_detector, calibration_loaded = load_models()
df_classes  = load_class_table()
class_names = classification_model.names   # dict {int: str}

LOW_SAMPLE_THRESHOLD = 20

# Result cards are pinned to a fixed height so the grid stays aligned across
# uploads of differing aspect ratio.  Anything taller scrolls within the card.
CARD_HEIGHT = 540
training_counts = {
    row["Class Name"].lower(): int(row["Image Count"])
    for _, row in df_classes.iterrows()
}

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

        # A file_uploader cannot be emptied by assigning to its session state;
        # rebuilding it under a fresh key is the supported way to clear it.
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        source_imgs = st.file_uploader(
            "Upload images...",
            type=("jpg", "jpeg", "png", "bmp", "webp"),
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if source_imgs:
            if st.button(f"Clear all ({len(source_imgs)})",
                         use_container_width=True):
                st.session_state.uploader_key += 1
                st.rerun()

        detection_confidence      = st.slider(
            "Detection confidence threshold", 10, 100, 50) / 100
        classification_confidence = st.slider(
            "Classification confidence threshold", 1, 100, 25) / 100
        top_k = st.slider("Top predictions to show", 1, 5, 3)

    st.title("CarabID")
    st.caption("Upload photos of ground beetles.")
    st.caption("Then click :blue[Identify] and check the results.")

    identify     = st.sidebar.button("Identify")
    show_results = bool(identify and source_imgs)

    # The guidance sits up front until there are results to look at, then moves
    # below them so the identifications stay in view.
    if not show_results:
        render_guidance()

    if identify:
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
                        # Fixed height keeps the grid aligned: uploads vary in
                        # aspect ratio, so cards would otherwise stagger.
                        with st.container(border=True, height=CARD_HEIGHT):
                            st.caption(source_img.name)
                            try:
                                # 1. Load
                                uploaded = PIL.Image.open(source_img)
                                if uploaded.mode != "RGB":
                                    uploaded = uploaded.convert("RGB")

                                # 2. Detection, orientation-corrected.  Runs at
                                # native resolution (YOLO letterboxes
                                # internally, so non-square photos are not
                                # distorted).  The crop must come from `view`,
                                # the image the boxes were measured against.
                                view, det_results, rotation = detect_best_orientation(
                                    uploaded, detection_model, detection_confidence
                                )

                                if det_results is None:
                                    st.image(uploaded, use_container_width=True)
                                    st.warning("No beetle detected")
                                    continue

                                det_plot = det_results[0].plot()[:, :, ::-1]
                                cropped  = crop_beetle(view, det_results)

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
                                if rotation:
                                    st.caption(
                                        f"↻ Rotated {rotation}° to landscape — "
                                        "the models expect horizontal specimens."
                                    )

                                ood_limit = (
                                    ood_detector.threshold
                                    if ood_detector is not None else None
                                )

                                if top_preds:
                                    top_genus, top_conf = top_preds[0]
                                    st.markdown(f"**{top_genus}**")
                                    st.progress(top_conf / 100)
                                    st.caption(f"{top_conf}% confidence (calibrated)")

                                    # Domain / familiarity check (Mahalanobis OOD).
                                    # Flags are compact triggers so every card in
                                    # the grid stays the same height; a popover
                                    # floats its detail rather than growing the
                                    # card the way an inline warning would.
                                    if is_novel:
                                        with st.popover("⚠ Unusual image or novel genus",
                                                        use_container_width=True):
                                            st.markdown(
                                                "This specimen sits far from everything the "
                                                "model was trained on. That can mean one of "
                                                "two quite different things:\n\n"
                                                "- **The photograph is unusual** — the training "
                                                "images are sharp, high-magnification "
                                                "**dorsal** views on a **plain background**, "
                                                "so a field photo or a distant shot lands "
                                                "far from them.\n"
                                                "- **The genus may not be one the model "
                                                "knows** — only 76 New Zealand carabid genera "
                                                "are covered, and anything outside that set "
                                                "can still only be matched to the closest of "
                                                "them.\n\n"
                                                "The app cannot tell these apart, so treat the "
                                                "identification with caution: check **All "
                                                "candidates** and verify against reference "
                                                "material."
                                            )
                                            if ood_score is not None:
                                                st.caption(
                                                    f"Novelty score {ood_score:.0f} "
                                                    f"(limit {ood_limit:.0f})"
                                                )
                                    elif ood_score is not None:
                                        st.caption(
                                            f"Familiarity: "
                                            f"{ood_familiarity_text(ood_score, ood_limit)}"
                                        )

                                    n_train = training_counts.get(top_genus.lower())
                                    if n_train is not None and n_train < LOW_SAMPLE_THRESHOLD:
                                        with st.popover("⚠ Limited training data",
                                                        use_container_width=True):
                                            st.markdown(
                                                f"*{top_genus}* was represented by only "
                                                f"**{n_train}** original images during training. "
                                                "Performance estimates for this genus are less "
                                                "reliable; verify this identification against "
                                                "reference material before finalising."
                                            )
                                    with st.expander("All candidates"):
                                        for rank, (genus, conf) in enumerate(top_preds, 1):
                                            st.text(f"{rank}. {genus} ({conf}%)")

                                else:
                                    st.caption("No confident identification")
                                    with st.popover("⚠ Low confidence",
                                                    use_container_width=True):
                                        if is_novel:
                                            st.markdown(
                                                "This photo also looks unlike the training "
                                                "specimens. Try a sharper, closer **dorsal** "
                                                "photo on a **plain background**."
                                            )
                                        else:
                                            st.markdown(
                                                "No genus scored above the classification "
                                                "confidence threshold. Lower the threshold in "
                                                "the sidebar to see weaker candidates."
                                            )

                            except Exception as ex:
                                st.error(f"Error: {ex}")

            if show_results:
                render_guidance()

with tab2:
    st.header("About the project")
    st.markdown(
        "This tool was developed using YOLOv11 models from the Ultralytics "
        "Python library (Jocher et al., 2023). A two-stage detection and "
        "classification pipeline is used. The first stage detects the beetle "
        "thorax and elytra; the second classifies it to genus. "
        "Full details can be found in the associated publication: "
        "Gong Y, Harmer AMT, Ward DF. (2026). Deep learning pipeline "
        "for the identification of ground beetles (Carabidae) in New Zealand. "
        "New Zealand Journal of Zoology, 53, e70049. https://doi.org/10.1002/njz2.70049."
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
        "- **Unusual image or novel genus warning** — the specimen falls far outside the "
        "training distribution. This has two possible causes, which the "
        "measure cannot distinguish: the *photograph* is unlike the training "
        "images (a low-magnification field photo, a cluttered background), or "
        "the specimen belongs to a *genus the model was never trained on* "
        "(only 76 genera are covered). In practice the first is far more "
        "common, but a genuinely unknown genus would look the same. The "
        "identification is still shown and should be verified against "
        "reference material."
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

    st.markdown("#### Limited training data warning")
    st.markdown(
        "Some genera in the dataset are represented by a small number of original "
        "specimen images (fewer than 20). Augmentation increases the size of the "
        "training set but cannot substitute for genuine biological variation across "
        "specimens. For these genera, cross-validation performance estimates are "
        "substantially more variable, and a correct identification in one fold may "
        "not reflect reliable generalisation.\n\n"
        "When the top-ranked prediction belongs to a genus with fewer than 20 "
        "original training images, the app displays a warning. In these cases, "
        "the identification should be treated as a lead rather than a confirmed "
        "result, and verified against reference material or expert opinion before "
        "finalising."
    )

    st.subheader("Trained genera")
    st.markdown(
        "The classification model was trained on the genera listed below. "
        "Image counts are for the original dataset before augmentation. "
        f"Genera with fewer than {LOW_SAMPLE_THRESHOLD} images are flagged "
        "with a warning at inference time."
    )
    st.dataframe(df_classes, height=600)
