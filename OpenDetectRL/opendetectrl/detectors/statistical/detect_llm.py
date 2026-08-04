"""DetectLLM: LRR and NPR zero-shot detectors (Luo et al., EMNLP 2023 Findings).

Official code: https://github.com/mbzuai-nlp/DetectLLM
(``baselines/all_baselines.py``). Equations (6) and (7) of the paper:

* **LRR** (Log-Likelihood over Log-Rank):  ``-ll(x) / logrank(x)``
* **NPR** (Normalized Log-Rank Perturbation):
  ``(1/K) * sum_k logrank(x_k^perturbed) / logrank(x)``

Perturbations are generated with the same official T5 mask-filling pipeline
as DetectGPT (official default mask model for NPR: ``t5-small``).
Higher = more likely machine-generated.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel
from .baselines import get_positive_logrank
from .perturbation import MaskFillPerturber, PerturbConfig, load_mask_model


class _DetectLLMBaseDetector(BaseDetector):
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

    @torch.inference_mode()
    def _get_ll(self, text: str) -> float:
        """Official ``loss.py::get_ll``: ``-model(**tokenized, labels=labels).loss``."""
        return self.model.get_mean_log_likelihood(text)

    @torch.inference_mode()
    def _get_logrank(self, text: str) -> float:
        """Official ``rank.py::get_rank(log=True)``: mean log token rank (positive)."""
        tokenized = self.model.tokenize(text)
        logits = self.model.model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]
        return get_positive_logrank(logits, labels)


@register_detector("lrr")
class LRRDetector(_DetectLLMBaseDetector):
    """DetectLLM LRR: ``-likelihood / log-rank`` (Eq. 6, official)."""

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        # Official all_baselines.py::nll_lrr
        scores = []
        for text in texts:
            likelihood = self._get_ll(text)
            logrank = self._get_logrank(text)
            scores.append(-likelihood / logrank)
        return np.asarray(scores, dtype=float)


@register_detector("npr")
class NPRDetector(_DetectLLMBaseDetector):
    """DetectLLM NPR: perturbed log-rank normalized by the original (Eq. 7).

    Args:
        mask_filling_model_name: Official default ``t5-small`` for NPR.
        n_perturbations: Number of perturbations K (official repo default 5).
        span_length / pct_words_masked / buffer_size: Official defaults
            2 / 0.3 / 1.
        perturber: Pre-built :class:`MaskFillPerturber` (advanced).
        random_fills_tokens: Official debug option to skip the T5 mask model.
    """

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        mask_filling_model_name: str = "t5-small",
        n_perturbations: int = 5,
        span_length: int = 2,
        pct_words_masked: float = 0.3,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        chunk_size: int = 20,
        perturber: Optional[MaskFillPerturber] = None,
        random_fills: bool = False,
        random_fills_tokens: bool = False,
        cache_dir: Optional[str] = None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.n_perturbations = n_perturbations

        # random_fills_tokens implies random_fills (skips T5 mask model)
        if random_fills_tokens:
            random_fills = True

        self.perturb_config = PerturbConfig(
            span_length=span_length,
            pct_words_masked=pct_words_masked,
            buffer_size=buffer_size,
            mask_top_p=mask_top_p,
            chunk_size=chunk_size,
            mask_filling_model_name=mask_filling_model_name,
            random_fills=random_fills,
            random_fills_tokens=random_fills_tokens,
        )

        if perturber is None:
            mask_model, mask_tokenizer = (None, None)
            if not random_fills:
                mask_model, mask_tokenizer = load_mask_model(
                    mask_filling_model_name, self.model.device, cache_dir=cache_dir
                )
            perturber = MaskFillPerturber(
                mask_model,
                mask_tokenizer,
                device=self.model.device,
                config=self.perturb_config,
                base_tokenizer=self.model.tokenizer if random_fills_tokens else None,
            )
        self.perturber = perturber

    def _get_perturbations(self, texts: List[str]) -> List[List[str]]:
        n = self.n_perturbations
        p_texts = self.perturber.perturb_texts(
            [x for x in texts for _ in range(n)], ceil_pct=True
        )
        assert len(p_texts) == len(texts) * n, (
            f"Expected {len(texts) * n} perturbed samples, got {len(p_texts)}"
        )
        return [p_texts[idx * n : (idx + 1) * n] for idx in range(len(texts))]

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        # Official all_baselines.py::nll_lrr with method="npr":
        # npr = mean(perturbed logrank) / original logrank
        perturbations = self._get_perturbations(list(texts))
        scores = []
        for text, p_texts in zip(texts, perturbations):
            logrank = self._get_logrank(text)
            perturbed_logranks = [self._get_logrank(p) for p in p_texts]
            scores.append(float(np.mean(perturbed_logranks)) / logrank)
        return np.asarray(scores, dtype=float)
