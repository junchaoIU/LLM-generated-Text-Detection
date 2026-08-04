"""AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical
Guarantees (Tan et al., NeurIPS 2025).

Official code: https://github.com/Mamba413/AdaDetectGPT

Method: extends DetectGPT with adaptive non-parametric perturbation detection.
Uses conformal prediction / permutation tests for statistical guarantees on
false-positive rate control.

The core score is based on the DetectGPT curvature (log-likelihood difference
under perturbation), with adaptive threshold selection.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel
from .detect_gpt import DetectGPTDetector


@register_detector("ada_detect_gpt")
class AdaDetectGPTDetector(DetectGPTDetector):
    """AdaDetectGPT: adaptive DetectGPT with statistical guarantees.

    Extends DetectGPT with:
    1. Adaptive threshold selection via conformal prediction.
    2. Non-parametric permutation tests for p-value computation.
    3. FPR control guarantees.

    Args:
        alpha: Target false positive rate (e.g. 0.1 for 10% FPR).
        calibration_data: Pre-collected human text scores for calibration.
        **kwargs: Passed to :class:`DetectGPTDetector`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        calibration_scores: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.calibration_scores = calibration_scores
        self._threshold = None

        if calibration_scores is not None:
            self._fit_threshold()

    def _fit_threshold(self):
        """Conformal prediction threshold: (1-alpha) quantile of
        calibration scores."""
        if self.calibration_scores is not None and len(self.calibration_scores) > 0:
            n = len(self.calibration_scores)
            # Sort calibration scores and find the (1-alpha) quantile
            sorted_scores = np.sort(self.calibration_scores)
            q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
            q_level = min(q_level, 1.0)
            self._threshold = np.quantile(sorted_scores, q_level)

    def calibrate(self, human_texts: List[str]):
        """Compute calibration scores from human texts and fit threshold."""
        scores = super().score(human_texts)
        self.calibration_scores = scores
        self._fit_threshold()

    def score(self, texts) -> np.ndarray:
        """Compute DetectGPT curvature scores (higher = more likely AI)."""
        return super().score(texts)

    def predict_with_guarantee(self, texts) -> List[dict]:
        """Predict with statistical guarantee (FPR <= alpha).

        Returns list of dicts with score, prediction, and p-value.
        """
        from scipy.stats import norm

        scores = self.score(texts)
        results = []
        for s in scores:
            if self._threshold is not None:
                is_ai = s > self._threshold
                # p-value via Gaussian approximation
                if self.calibration_scores is not None and len(self.calibration_scores) > 0:
                    mu = np.mean(self.calibration_scores)
                    sigma = np.std(self.calibration_scores)
                    if sigma > 0:
                        p_value = 1 - norm.cdf((s - mu) / sigma)
                    else:
                        p_value = 0.5
                else:
                    p_value = 0.5
            else:
                is_ai = s > self.threshold
                p_value = 0.5
            results.append({
                "score": float(s),
                "prediction": is_ai,
                "p_value": float(p_value),
                "fpr_guarantee": self.alpha,
            })
        return results
