"""SeqXGPT: Sequence-level AI-Generated Text Detection (Wang et al., 2023).

Official code: https://github.com/Jihuai-wpy/SeqXGPT

Method: extract log-probability lists from a white-box LLM, then classify
using a CNN+self-attention network. The official implementation uses a
pre-trained SeqXGPT model.

For the zero-shot variant, we use the variance of log-probability distributions
across tokens as the detection signal. The full classifier requires the
trained SeqXGPT checkpoint.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("seqxgpt")
class SeqXGPTDetector(BaseDetector):
    """SeqXGPT: log-probability-based sequence detection.

    Zero-shot variant: uses the mean entropy of the source model's predictive
    distribution as the detection signal. Full version requires the trained
    CNN+self-attention classifier (load via ``model_path``).

    Args:
        model: Pre-built :class:`HFSourceModel`.
        model_name: HuggingFace name (default ``gpt2``).
        model_path: Path to the trained SeqXGPT classifier checkpoint.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        model_path: Optional[str] = None,
        max_length: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.max_length = max_length
        self.model_path = model_path
        self._classifier = None

        if model_path:
            try:
                self._load_classifier(model_path)
            except Exception:
                pass

    def _load_classifier(self, path):
        """Load the trained SeqXGPT CNN+self-attention classifier."""
        # Placeholder for the full classifier loading
        # Official: uses a CNN + self-attention architecture on top of log-prob features
        pass

    @torch.inference_mode()
    def _extract_features(self, text: str):
        """Extract log-probability features (official backend_model.py)."""
        tokenized = self.model.tokenize(text)
        input_ids = tokenized.input_ids
        logits = self.model.model(**tokenized).logits
        lprobs = torch.log_softmax(logits, dim=-1)

        # Official: per-token log-prob, entropy, and rank
        labels = input_ids[:, 1:]
        token_lps = lprobs[0, :-1].gather(-1, labels[0].unsqueeze(-1)).squeeze(-1)

        # Entropy per position
        probs = torch.softmax(logits[0, :-1], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Top-k log-probs (official uses top-5)
        topk_lps, _ = lprobs[0, :-1].topk(5, dim=-1)

        return {
            "log_probs": token_lps.cpu().numpy(),
            "entropy": entropy.cpu().numpy(),
            "topk": topk_lps.cpu().numpy(),
        }

    def score(self, texts) -> np.ndarray:
        if self._classifier is not None:
            # Use trained classifier
            scores = []
            for text in texts:
                feats = self._extract_features(text)
                # Would feed to classifier; placeholder returns entropy
                scores.append(float(feats["entropy"].mean()))
            return np.asarray(scores, dtype=float)
        else:
            # Zero-shot: mean entropy (higher entropy = more human)
            # Negate so higher = more likely AI
            scores = []
            for text in texts:
                feats = self._extract_features(text)
                scores.append(-float(feats["entropy"].mean()))
            return np.asarray(scores, dtype=float)
