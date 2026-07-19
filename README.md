# CarabID

**A computer vision pipeline for identifying New Zealand carabid beetles to genus from photographs.**
<table>
  <tr>
    <td valign="middle"><a href="https://carabid.streamlit.app"><img src="app/static/carabid_icon.png" height="50" alt="CarabID"/></a></td>
    <td valign="middle"><a href="https://carabid.streamlit.app">Streamlit App</a></td>
  </tr>
</table>

CarabID uses a two-stage YOLO (You Only Look Once) detection and classification pipeline:

1. **Detection** — a YOLOv11n model locates the beetle (thorax + elytra region) in the photograph.
2. **Classification** — a YOLOv11n-cls model classifies the crop to one of 76 genera.

Post-training calibration corrects overconfident probability estimates (temperature scaling) and flags specimens that do not resemble any trained genus (Mahalanobis distance out-of-distribution detection).

Associated publication: Gong, Harmer & Ward (in review).

---

## Running the app locally

```bash
# from the carabID/ root
streamlit run app/app.py
```

Requirements: Python 3.11, see `app/requirements.txt`.  
Model weights must be present in `app/static/` (see below).

### Static assets (`app/static/`)

| File | Description |
|---|---|
| `detection.pt` | YOLOv11n detection model weights |
| `classification.pt` | YOLOv11n-cls classification model weights |
| `temperature.pt` | Temperature scaling parameter (fitted on val set) |
| `ood_model.pkl` | Mahalanobis OOD detector (fitted on training embeddings) |
| `class_counts.csv` | Genus names and training image counts |
| `carabid_icon.png` | App icon |
| `example_image.jpg` | Example beetle photograph |

The `.pt` model files are tracked in this repository (< 15 MB each).  
Training images (~4 700 images, ~9.9 GB) are archived on Zenodo: https://zenodo.org/records/20634783

---

## Calibration

Post-hoc calibration is implemented in `calibrated/` and applies to the final trained model without retraining.

### Temperature scaling

Raw softmax probabilities from neural networks are typically overconfident.
Temperature scaling (Guo et al., 2017) divides the pre-softmax logits by a
scalar *T* ≥ 1, which spreads the probability distribution without affecting
the ranked order of predictions.  *T* is fitted by minimising negative
log-likelihood on the validation set.

### Mahalanobis out-of-distribution detection

To flag specimens belonging to genera not seen during training, we follow
Lee et al. (2018).  Per-class centroids and a shared covariance matrix are
fitted on training-set embeddings extracted from the final linear layer.
At inference, the minimum Mahalanobis distance to any centroid is computed.
A threshold is set at the 95th percentile of validation-set distances (5% false
novelty rate).

### Running calibration

```bash
python calibrated/calibrate_yolo_classifier.py \
    --model  runs/classify/final_carabid_model_11ncls_ep30_autobatch_do02_lr001/weights/best.pt \
    --data   imgs/final_classification_set \
    --imgsz  640

# Copy artefacts to app/static/ to activate calibration in the app
cp runs/classify/.../weights/temperature.pt app/static/
cp runs/classify/.../weights/ood_model.pkl  app/static/
```

### Computing calibration metrics

```bash
python calibrated/compute_calibration_metrics.py \
    --model           runs/classify/.../weights/best.pt \
    --data            imgs/final_classification_set \
    --calibration-dir runs/classify/.../weights \
    --out             calibrated/results
```

Reliability diagram (requires R with ggplot2 + patchwork):

```bash
Rscript calibrated/plot_reliability_diagram.R
```

---

## Training

### Final model (single run)

```bash
python scripts/train_classification_model.py
```

### k-fold cross-validation (5 folds)

```bash
python scripts/kfold_train_classification_models.py
python scripts/kfold_assess_classification_models.py
```

Training images are required in `imgs/final_classification_set/` (train/ and val/ sub-folders in ImageFolder format).  Download from [Zenodo](https://zenodo.org/records/20634783).

---

## Repository structure

```
carabID/
├── app/                            ← Streamlit app (deployed)
│   ├── app.py                      ← main app (calibrated)
│   ├── requirements.txt
│   ├── packages.txt
│   ├── .streamlit/config.toml
│   └── static/                     ← model weights + assets
│
├── calibrated/                     ← post-hoc calibration utilities
│   ├── calibration_utils.py        ← TemperatureScaler + MahalanobisOOD
│   ├── calibrate_yolo_classifier.py
│   ├── compute_calibration_metrics.py
│   ├── assess_classification_model.py
│   └── plot_reliability_diagram.R
│
└── scripts/                        ← data preparation, training, analysis
    ├── run_corrected_pipeline.py    ← split-before-augment dataset build
    ├── train_classification_model.py
    ├── train_detection_model.py
    ├── kfold_train_classification_models.py
    ├── kfold_assess_classification_models.py
    ├── evaluate_novelty_detection.py    ← validates the OOD threshold
    └── ...                          ← figures and analysis (R + Python)
```

`imgs/` (training data, available on [Zenodo](https://zenodo.org/records/20634783)) and `runs/` (model checkpoints) are excluded from version control.

---

## Python environment

```bash
conda create -n carabID python=3.11
pip install -r app/requirements.txt
```

For GPU training (CUDA 12.x):

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *International Conference on Machine Learning*, 1321–1330.
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO. https://github.com/ultralytics/ultralytics
- Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in Neural Information Processing Systems*, 31.

---

## Citation

If you use CarabID in your research, please cite:

> Gong, Y., Harmer, A. M. T., & Ward, D. F. (in review). CarabID: a computer vision pipeline for identification of New Zealand carabid beetles.
