"""Polish Ratio (PR) / HPPT detector (Yang et al., APSIPA TSP 2024).

Official code: https://github.com/FreedomIntelligence/ChatGPT-Detection-PR-HPPT
Paper: "Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect
ChatGPT-Generated Text" (Yang, Jiang, Li).

The official implementation trains a RoBERTa classifier (HPPT) and a
RoBERTa regression model (PR) on the HPPT dataset of human-written and
ChatGPT-polished academic abstracts. The Polish Ratio quantifies the
degree of ChatGPT involvement via editing distance (Jaccard / Levenshtein)
between the original human text and the polished version. A PR > 0.2
indicates ChatGPT involvement; PR > 0.6 implies ChatGPT generated most
of the text.

This zero-shot adaptation captures the paper's central observation that
ChatGPT-polished text has **lower perplexity** than human text: a causal
language model assigns higher likelihood (lower perplexity) to polished
text because polishing removes the surface irregularities of human
writing. The detector computes the perplexity of each text under the
source model and returns the negated perplexity as the score, so that
**higher = more likely AI-generated**.
"""
from typing import Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("polish_ratio")
class PolishRatioDetector(BaseDetector):
    """Polish Ratio: perplexity-based detection of ChatGPT-polished text.

    The score is the negated perplexity of the text under the source
    model. ChatGPT-polished text has lower perplexity (higher likelihood)
    than human text, so a higher (less negative) score indicates more
    likely AI-generated text.

    Args:
        model: Pre-built :class:`HFSourceModel` used for scoring.
        model_name: HuggingFace name of the scoring model (default
            ``gpt2``).
        threshold: Decision threshold; should be calibrated on validation
            data (the negated-perplexity score is always negative, so the
            default of 0.0 classifies everything as human until tuned).
        device: Target device; defaults to CUDA if available, else CPU.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model

    def _perplexity(self, text: str) -> float:
        """Perplexity of ``text`` under the source model.

        ``perplexity = exp(-mean_log_likelihood)`` where the mean
        log-likelihood is ``-model(..., labels=ids).loss`` (official
        DetectGPT ``get_ll`` convention).
        """
        ll = self.model.get_mean_log_likelihood(text)
        # ll <= 0 (mean log-likelihood of real text is non-positive);
        # clamp to avoid overflow for extremely low-likelihood inputs.
        return float(np.exp(-np.clip(ll, -100.0, 0.0)))

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            # Negate perplexity: lower perplexity (more polished / AI-like)
            # => higher score (higher = more likely AI-generated).
            scores.append(-self._perplexity(text))
        return np.asarray(scores, dtype=float)
