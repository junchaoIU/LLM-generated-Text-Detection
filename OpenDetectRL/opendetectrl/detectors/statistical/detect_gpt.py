"""DetectGPT (Mitchell et al., ICML 2023).

Official code: https://github.com/eric-mitchell/detect-gpt (``run.py``).

Detection score (official criteria):
    * ``d``: ``ll(x) - mean(ll(perturbed))``
    * ``z``: ``(ll(x) - mean(ll(perturbed))) / std(ll(perturbed))``

where perturbations are produced by the official T5 mask-filling pipeline
(:mod:`opendetectrl.detectors.statistical.perturbation`). Higher = more
likely machine-generated.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel
from .perturbation import MaskFillPerturber, PerturbConfig, load_mask_model


@register_detector("detect_gpt")
class DetectGPTDetector(BaseDetector):
    """DetectGPT: log-probability curvature under mask-fill perturbations.

    Args:
        model: Pre-built :class:`HFSourceModel` (base model).
        model_name: HuggingFace name of the base model (official default
            ``gpt2-medium``).
        mask_filling_model_name: T5 model used for mask filling (official
            default ``t5-large``).
        n_perturbations: Number of perturbations per text (official runs use
            1 / 10 / 100).
        n_perturbation_rounds: Official ``--n_perturbation_rounds`` (default 1).
        span_length: Official ``--span_length`` (default 2).
        pct_words_masked: Official ``--pct_words_masked`` (default 0.3).
        buffer_size: Official ``--buffer_size`` (default 1).
        mask_top_p: Official ``--mask_top_p`` (default 1.0).
        chunk_size: Official ``--chunk_size`` (default 20).
        criterion: ``"z"`` (normalized, paper default) or ``"d"`` (raw diff).
        random_fills / random_fills_tokens: Official debug options that avoid
            loading the T5 mask model.
        perturber: Pre-built :class:`MaskFillPerturber` (advanced).
        cache_dir: HuggingFace cache directory for the mask model.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2-medium",
        mask_filling_model_name: str = "t5-large",
        n_perturbations: int = 10,
        n_perturbation_rounds: int = 1,
        span_length: int = 2,
        pct_words_masked: float = 0.3,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        chunk_size: int = 20,
        criterion: str = "z",
        random_fills: bool = False,
        random_fills_tokens: bool = False,
        perturber: Optional[MaskFillPerturber] = None,
        cache_dir: Optional[str] = None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        assert criterion in ("d", "z"), f"Unknown criterion: {criterion}"
        if model is None:
            model = HFSourceModel(model_name, device=device, cache_dir=cache_dir)
        self.model = model
        self.criterion = criterion
        self.n_perturbations = n_perturbations
        self.n_perturbation_rounds = n_perturbation_rounds

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

    def _get_ll(self, text: str) -> float:
        """Official ``get_ll``: ``-model(**tokenized, labels=labels).loss``."""
        return self.model.get_mean_log_likelihood(text)

    def _get_perturbations(self, texts: List[str]) -> List[List[str]]:
        """Official ``get_perturbation_results`` perturbation logic."""
        n = self.n_perturbations
        # replicate each text n_perturbations times, then perturb all at once
        p_texts = self.perturber.perturb_texts(
            [x for x in texts for _ in range(n)], ceil_pct=True
        )
        for _ in range(self.n_perturbation_rounds - 1):
            try:
                p_texts = self.perturber.perturb_texts(p_texts, ceil_pct=True)
            except AssertionError:
                break
        assert len(p_texts) == len(texts) * n, (
            f"Expected {len(texts) * n} perturbed samples, got {len(p_texts)}"
        )
        return [p_texts[idx * n : (idx + 1) * n] for idx in range(len(texts))]

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        perturbations = self._get_perturbations(list(texts))
        scores = []
        for text, p_texts in zip(texts, perturbations):
            ll = self._get_ll(text)
            p_lls = [self._get_ll(p) for p in p_texts]
            mean_p_ll = float(np.mean(p_lls))
            # Official: std is set to 1 when there is a single perturbation,
            # or when the std is zero.
            if len(p_lls) > 1:
                std_p_ll = float(np.std(p_lls))
                if std_p_ll == 0:
                    std_p_ll = 1.0
            else:
                std_p_ll = 1.0
            if self.criterion == "d":
                scores.append(ll - mean_p_ll)
            else:  # "z"
                scores.append((ll - mean_p_ll) / std_p_ll)
        return np.asarray(scores, dtype=float)
