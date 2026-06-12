"""
Standalone calibration utilities for YOLO classification models.

Classes
-------
TemperatureScaler   Post-hoc calibration via a single learned temperature scalar.
MahalanobisOOD      Shared-covariance Mahalanobis distance OOD detector.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Temperature Scaling                                                         #
# --------------------------------------------------------------------------- #

class TemperatureScaler(nn.Module):
    """
    Divide logits by a learnable scalar T before softmax.

    T > 1  →  model was overconfident  (probabilities spread out)
    T < 1  →  model was underconfident (probabilities sharpened)
    T = 1  →  no change

    Fitted by minimising NLL on held-out logits using LBFGS.
    Saved/loaded with torch.save / torch.load on state_dict.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled softmax probabilities (no gradient)."""
        with torch.no_grad():
            return torch.softmax(self.forward(logits), dim=-1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 100) -> "TemperatureScaler":
        """Fit T by minimising NLL on (logits, labels); both detached."""
        logits = logits.detach()
        labels = labels.detach()
        nll    = nn.CrossEntropyLoss()
        opt    = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @classmethod
    def load(cls, path) -> "TemperatureScaler":
        ts = cls()
        ts.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return ts


# --------------------------------------------------------------------------- #
#  Mahalanobis OOD                                                             #
# --------------------------------------------------------------------------- #

class MahalanobisOOD:
    """
    Shared-covariance Mahalanobis distance OOD detector.

    Fitted on L2-normalised training embeddings.  score(z) returns the
    minimum Mahalanobis distance from z to any class centroid — lower means
    more in-distribution.  Threshold tuned to a target false-novelty rate on
    a held-out in-distribution (validation) set.
    """

    def __init__(self, reg: float = 1e-5):
        self.reg          = reg
        self.class_means: dict        = {}
        self.precision:   np.ndarray  = None   # Σ⁻¹
        self.threshold:   float       = None
        self._classes:    np.ndarray  = None

    # ------------------------------------------------------------------ fit #

    def fit(self, embeddings: np.ndarray,
            labels: np.ndarray) -> "MahalanobisOOD":
        """
        embeddings : (N, D) float32/64 — L2-normalised feature vectors
        labels     : (N,)  int         — class indices
        """
        embeddings       = np.asarray(embeddings, dtype=np.float64)
        labels           = np.asarray(labels)
        self._classes    = np.unique(labels)
        D                = embeddings.shape[1]
        pooled_cov       = np.zeros((D, D), dtype=np.float64)
        n_total          = 0

        for c in self._classes:
            mask                  = labels == c
            z_c                   = embeddings[mask]
            mu_c                  = z_c.mean(axis=0)
            self.class_means[int(c)] = mu_c
            diff                  = z_c - mu_c
            pooled_cov           += diff.T @ diff
            n_total              += len(z_c)

        dof            = max(n_total - len(self._classes), 1)
        pooled_cov    /= dof
        pooled_cov    += self.reg * np.eye(D)
        self.precision = np.linalg.inv(pooled_cov)
        return self

    # --------------------------------------------------------------- score #

    def score(self, embedding: np.ndarray) -> float:
        """Minimum Mahalanobis distance to any class centroid."""
        z    = np.asarray(embedding, dtype=np.float64).flatten()
        best = float("inf")
        for mu in self.class_means.values():
            d    = z - mu
            dist = float(d @ self.precision @ d)
            if dist < best:
                best = dist
        return best

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        return np.array([self.score(z) for z in embeddings])

    # --------------------------------------------------- threshold tuning #

    def tune_threshold(self, val_embeddings: np.ndarray,
                       fpr_target: float = 0.05) -> float:
        """
        Set threshold at the (1 - fpr_target) quantile of in-distribution
        val scores so that at most fpr_target of known-class val samples are
        incorrectly flagged as novel.
        """
        scores         = self.score_batch(val_embeddings)
        self.threshold = float(np.quantile(scores, 1.0 - fpr_target))
        return self.threshold

    def is_novel(self, embedding: np.ndarray) -> bool:
        if self.threshold is None:
            raise RuntimeError("Call tune_threshold() before is_novel().")
        return self.score(embedding) > self.threshold

    # -------------------------------------------------------- persistence #

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "MahalanobisOOD":
        with open(path, "rb") as f:
            return pickle.load(f)
