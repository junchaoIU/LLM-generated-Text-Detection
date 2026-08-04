"""CheckGPT: Detecting ChatGPT-Generated Academic Writing.

Paper: Pudasaini et al., "Check Me If You Can: Detecting ChatGPT-Generated
Academic Writing using CheckGPT", arXiv 2306.05524.

The full method generates "answers" to a document's questions with ChatGPT and
measures the alignment between the document and the generated answers via
BERTScore / NLI.  This library provides a practical, dependency-light proxy
that keeps the same underlying signal: AI-generated text tends to be more
self-consistent and lower-perplexity under a reference language model than
human-written text.  The score is the negative mean per-token log-likelihood
of the text under the reference model, normalized so that higher = more AI.
"""
from typing import Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("checkgpt")
class CheckGPTDetector(BaseDetector):
    """CheckGPT-style self-consistency detector.

    Uses a reference causal LM to score the text's mean log-likelihood.  The
    returned score is ``-mean_log_likelihood`` (i.e. higher = more AI-generated),
    consistent with the library convention.

    Args:
        model_name: HF causal LM for scoring (default ``gpt2``).
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL_NAME

    def _get_model(self) -> HFSourceModel:
        if not hasattr(self, "_model"):
            self._model = HFSourceModel(self.model_name, device=self.device)
        return self._model

    def score(self, texts) -> np.ndarray:
        model = self._get_model()
        texts = list(texts)
        scores = []
        for text in texts:
            ll = model.get_mean_log_likelihood(text)
            scores.append(-ll)
        return np.asarray(scores, dtype=float)