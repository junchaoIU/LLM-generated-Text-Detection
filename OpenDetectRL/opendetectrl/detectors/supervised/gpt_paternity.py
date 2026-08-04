"""GPT Paternity Test: GPT Generated Text Detection with GPT Genetic Inheritance.

Paper: Akram et al., "GPT Paternity Test: GPT Generated Text Detection with
GPT Genetic Inheritance", arXiv 2305.12519.
Repo: https://github.com/7-yaya/GPT-Pat

The core idea is that GPT-generated text inherits a characteristic
"genetic" token-frequency distribution from the generating model.  A
candidate text is embedded with a reference source model, and then compared
(KNN / cosine similarity) against a reference set of known AI-generated
embeddings.  Text that lives close to the AI-generated manifold is scored as
more likely AI-generated.

This is a few-shot / calibration method: the user must first call
:meth:`fit` with a reference corpus of known AI-generated and human-written
texts, after which :meth:`score` can be applied to arbitrary new texts.
"""
from typing import List, Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("gpt_paternity_test")
class GPTPaternityTestDetector(BaseDetector):
    """GPT Paternity Test KNN-embedding detector.

    Embeds texts with a reference causal LM's hidden states, then uses
    K-nearest-neighbours in the embedding space against a reference set of
    known AI-generated / human-written texts (from :meth:`fit`).  The score
    is ``mean_cosine_to_ai - mean_cosine_to_human`` so that higher = more
    likely AI-generated.

    Args:
        model_name: HF name for the reference embedding model.
        layer: Layer index of the hidden states to use (``-1`` = last layer).
        k: Number of nearest neighbours to consider.
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    requires_source_model = True

    def __init__(
        self,
        model_name: str = "gpt2",
        layer: int = -1,
        k: int = 5,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name
        self.layer = layer
        self.k = k
        # Reference embeddings are set by ``fit``.
        self._ai_embs: Optional[np.ndarray] = None
        self._human_embs: Optional[np.ndarray] = None

    def _get_model(self) -> HFSourceModel:
        """Lazily build the embedding source model."""
        if not hasattr(self, "_model"):
            self._model = HFSourceModel(self.model_name, device=self.device)
        return self._model

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts as mean-pooled last-layer hidden states."""
        model = self._get_model()
        embs = []
        for text in texts:
            # hidden_states[k] is (batch, seq, hidden); take the requested layer.
            states = model.get_hidden_states([text])[0]
            if self.layer == -1:
                vec = states.mean(dim=0)
            else:
                vec = states[self.layer].mean(dim=0)
            embs.append(vec.numpy().astype(np.float64))
        # Normalize rows to unit norm for cosine similarity.
        embs = np.asarray(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def fit(self, reference_texts: Sequence[str], reference_is_ai: Sequence[bool]):
        """Calibrate the detector on a labelled reference corpus.

        Args:
            reference_texts: Reference texts (AI-generated and human-written).
            reference_is_ai: Boolean label per text (``True`` = AI-generated).
        """
        is_ai = np.asarray(reference_is_ai, dtype=bool)
        embs = self._embed(reference_texts)
        self._ai_embs = embs[is_ai]
        self._human_embs = embs[~is_ai]
        if len(self._ai_embs) == 0 or len(self._human_embs) == 0:
            raise ValueError(
                "fit() requires at least one AI-generated and one human-written reference text."
            )

    def _mean_cosine_to(self, query: np.ndarray, refs: np.ndarray) -> float:
        """Mean cosine similarity of ``query`` to the nearest ``k`` refs."""
        if len(refs) == 0:
            return 0.0
        k = min(self.k, len(refs))
        sims = refs @ query  # refs are unit-norm, query is unit-norm
        sims = np.sort(sims)[-k:]
        return float(sims.mean())

    def score(self, texts) -> np.ndarray:
        if self._ai_embs is None or self._human_embs is None:
            raise RuntimeError(
                "GPTPaternityTestDetector needs a reference corpus: call fit(reference_texts, reference_is_ai) first."
            )
        texts = list(texts)
        embs = self._embed(texts)
        scores = []
        for vec in embs:
            ai_sim = self._mean_cosine_to(vec, self._ai_embs)
            human_sim = self._mean_cosine_to(vec, self._human_embs)
            scores.append(ai_sim - human_sim)
        return np.asarray(scores, dtype=float)