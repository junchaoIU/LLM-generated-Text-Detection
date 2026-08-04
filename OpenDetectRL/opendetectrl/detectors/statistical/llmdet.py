"""LLMDet: probability-based detection using multiple reference LLMs.

Official code: https://github.com/TrustedLLM/LLMDet
(Wu et al., "LLMDet: A Third Party Large Language Models Generated Text
Detection Tool", EMNLP 2023 Findings).

LLMDet is a black-box detection tool that analyzes probability-based features
from multiple open-source LLMs. The core idea is that machine-generated text
has higher probability (lower perplexity) under the generating model. The
original tool computes "proxy perplexity" via pre-computed n-gram statistics to
avoid running the models directly, then feeds the per-model perplexities as
features to a LightGBM classifier that distinguishes human-written from
machine-generated text.

This implementation computes the actual mean log-likelihood of the text under
each reference model -- the quantity that proxy perplexity approximates -- and
aggregates across models. The score is the maximum mean log-likelihood across
all reference models: a text that is highly probable under at least one LLM is
more likely machine-generated.

Score direction: higher = more likely AI-generated.
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector, DetectionResult
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("llmdet")
class LLMDetDetector(BaseDetector):
    """LLMDet: multi-model probability-based generated-text detection.

    The detector computes the mean log-likelihood of each input text under
    multiple reference LLMs and takes the maximum. Machine-generated text
    tends to have high log-likelihood (low perplexity) under the model that
    generated it.

    The original LLMDet approximates these log-likelihoods via pre-computed
    n-gram statistics ("proxy perplexity") and uses a LightGBM classifier on
    the resulting feature vector. This implementation uses the actual model
    log-likelihoods (via :class:`HFSourceModel`) and a max aggregation as a
    self-contained substitute for the external classifier.

    Args:
        reference_models: Pre-built :class:`HFSourceModel` instances to use as
            reference models. Takes priority over ``reference_model_names``.
        reference_model_names: HuggingFace names of the reference models.
            Defaults to a small set of open-source causal LLMs.
        model: A single pre-built :class:`HFSourceModel` (convenience alias
            wrapped into a one-element reference-model list).
        model_name: A single HuggingFace model name (convenience alias).
        threshold: Decision threshold on the aggregated log-likelihood.
        device: Device for model inference.
        cache_dir: HuggingFace cache directory.
    """

    requires_source_model = True

    #: Default reference models (small open-source causal LLMs).
    DEFAULT_REFERENCE_MODELS = ["gpt2", "facebook/opt-125m"]

    def __init__(
        self,
        reference_models: Optional[List[HFSourceModel]] = None,
        reference_model_names: Optional[List[str]] = None,
        model: Optional[HFSourceModel] = None,
        model_name: Optional[str] = None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)

        if reference_models is not None:
            self.reference_models = list(reference_models)
        elif model is not None:
            self.reference_models = [model]
        elif model_name is not None:
            self.reference_models = [
                HFSourceModel(model_name, device=device, cache_dir=cache_dir)
            ]
        else:
            names = reference_model_names or self.DEFAULT_REFERENCE_MODELS
            self.reference_models = [
                HFSourceModel(name, device=device, cache_dir=cache_dir)
                for name in names
            ]

    @torch.inference_mode()
    def _compute(self, text: str):
        """Return ``(score, per_model_log_likelihoods)`` for a single text."""
        per_model: Dict[str, float] = {}
        for m in self.reference_models:
            per_model[m.model_name_or_path] = m.get_mean_log_likelihood(text)
        score = max(per_model.values()) if per_model else 0.0
        return score, per_model

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            s, _ = self._compute(text)
            scores.append(s)
        return np.asarray(scores, dtype=float)

    def predict(self, texts) -> List[DetectionResult]:
        """Per-text predictions with per-model log-likelihoods in metadata."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        results: List[DetectionResult] = []
        for text in texts:
            s, per_model = self._compute(text)
            results.append(
                DetectionResult(
                    score=float(s),
                    label="ai" if s > self.threshold else "human",
                    metadata={"per_model_log_likelihood": per_model},
                )
            )
        return results
