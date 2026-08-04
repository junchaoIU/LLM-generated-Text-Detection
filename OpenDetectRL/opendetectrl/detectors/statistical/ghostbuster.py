"""Ghostbuster: Exorcising Fake Content with Spectral Features (Verma et al., 2023).

Official code: https://github.com/vivek3141/ghostbuster

Method: compute the log-probabilities of the text under multiple language
models, then use the differences (spectral features) to train a classifier.
The official code uses three models and computes:
    score = sign(vec_H1 - vec_H2) @ (eigenvector of (H1-H2).T @ (H1-H2))

For a zero-shot variant, we use the variance of log-probabilities across
models as the score (higher variance = more likely human text, since LLM
outputs tend to have similar log-probs across models).

The full classifier version requires a trained model; the zero-shot variant
is provided here.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("ghostbuster")
class GhostbusterDetector(BaseDetector):
    """Ghostbuster: spectral features from multi-model log-probabilities.

    Zero-shot variant: compute the standard deviation of per-token
    log-probabilities across multiple models. Higher = more likely human
    (negative convention: we negate so higher = more likely AI).

    Args:
        models: List of pre-built :class:`HFSourceModel` instances.
        model_names: List of HuggingFace model names (default: ``gpt2``,
            ``gpt2-medium``, ``gpt2-large``).
    """

    requires_source_model = True

    DEFAULT_MODEL_NAMES = ["gpt2", "gpt2-medium", "gpt2-large"]

    def __init__(
        self,
        models: Optional[List[HFSourceModel]] = None,
        model_names: Optional[List[str]] = None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if models is None:
            model_names = model_names or self.DEFAULT_MODEL_NAMES
            models = [HFSourceModel(name, device=device) for name in model_names]
        self.models = models
        self.model_names = model_names or [m.model_name_or_path for m in models]

    @torch.inference_mode()
    def _get_log_probs(self, text: str) -> List[List[float]]:
        """Get per-token log-probs from each model."""
        all_lps = []
        for model in self.models:
            tokenized = model.tokenize(text)
            logits = model.model(**tokenized).logits
            lprobs = torch.log_softmax(logits, dim=-1)
            labels = tokenized.input_ids[:, 1:]
            token_lps = lprobs[0, :-1].gather(-1, labels[0].unsqueeze(-1)).squeeze(-1)
            all_lps.append(token_lps.cpu().numpy().tolist())
        return all_lps

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            all_lps = self._get_log_probs(text)
            # Align by minimum length (different tokenizers produce different lengths)
            min_len = min(len(lps) for lps in all_lps)
            truncated = np.array([lps[:min_len] for lps in all_lps])
            # Spectral feature: std across models, averaged over tokens
            # Lower std => models agree => more likely AI
            # We negate so higher score = more likely AI
            std_across = truncated.std(axis=0).mean()
            scores.append(-std_across)
        return np.asarray(scores, dtype=float)
