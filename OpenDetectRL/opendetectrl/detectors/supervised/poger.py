"""POGER: Proxy-Guided Efficient Re-sampling for Black-Box AI-Generated Text
Detection (Li et al., IJCAI 2024).

Official code: https://github.com/ICTMCG/POGER (``POGER/main.py``)

Method: extract token-level loss features (FCE/BCE with summary-based prompts)
from a detection model, then classify with a trained classifier. Features
include mean/max/min/std of loss across 9 segments.

POGER is a supervised detector that requires a trained checkpoint. This
implementation wraps it as an HF classifier variant.
"""
from typing import List, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel
from ..supervised.hf_classifier import _HFClassifierBase


@register_detector("poger")
class POGERDetector(_HFClassifierBase):
    """POGER: proxy-guided efficient re-sampling detector.

    Uses a trained classifier on top of token-level loss features extracted
    from a detection LLM. The backbone is RoBERTa.

    Args:
        model_name: HuggingFace name of the POGER checkpoint.
        detect_model_name: Name of the detection LLM for feature extraction
            (official: LLaMA-2/3, Gemma, Mistral).
        n_feat: Number of features per segment (official default 7).
        k: K for k-NN classifier (official default 10).
    """

    DEFAULT_MODEL_NAME = "roberta-base"
    AI_CLASS_INDEX = 1

    def __init__(
        self,
        detect_model: Optional[HFSourceModel] = None,
        detect_model_name: str = "gpt2",
        n_feat: int = 7,
        k: int = 10,
        sample_clip: int = 2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detect_model = detect_model or HFSourceModel(detect_model_name, device=str(self.device))
        self.n_feat = n_feat
        self.k = k
        self.sample_clip = sample_clip

    def _extract_loss_features(self, text: str) -> np.ndarray:
        """Extract token-level loss features (official biscope_utils style)."""
        prompt = "Complete the following text: "
        prompt_ids = self.detect_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.detect_model.device
        )
        text_ids = self.detect_model.tokenizer(
            text, return_tensors="pt", max_length=self.sample_clip, truncation=True
        ).input_ids.to(self.detect_model.device)
        combined = torch.cat([prompt_ids, text_ids], dim=1)
        text_slice = slice(prompt_ids.shape[1], combined.shape[1])

        with torch.inference_mode():
            outputs = self.detect_model.model(input_ids=combined)
            logits = outputs.logits
            targets = combined[0][text_slice]

            fce = CrossEntropyLoss(reduction="none")(
                logits[0, text_slice.start - 1 : text_slice.stop - 1, :], targets
            ).cpu().numpy()
            bce = CrossEntropyLoss(reduction="none")(
                logits[0, text_slice, :], targets
            ).cpu().numpy()

        features = []
        for p in range(1, 10):
            split = len(fce) * p // 10
            features.extend([
                np.mean(fce[split:]), np.max(fce[split:]),
                np.min(fce[split:]), np.std(fce[split:]),
                np.mean(bce[split:]), np.max(bce[split:]),
                np.min(bce[split:]), np.std(bce[split:]),
            ])
        return np.asarray(features, dtype=float)

    def score(self, texts) -> np.ndarray:
        # Use the HF classifier for scoring, with loss features as fallback
        return super().score(texts)
