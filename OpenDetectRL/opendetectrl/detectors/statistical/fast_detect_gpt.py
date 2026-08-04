"""Fast-DetectGPT (Bao et al., ICLR 2024).

Official code: https://github.com/baoguangsheng/fast-detect-gpt
(``scripts/fast_detect_gpt.py``). The criterion functions below are ported
verbatim from the official implementation.

Score direction follows the official evaluation: **higher discrepancy = more
likely machine-generated**.
"""
from typing import Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# ---------------------------------------------------------------------------
# Official criterion functions (fast-detect-gpt scripts/fast_detect_gpt.py)
# ---------------------------------------------------------------------------


def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)


def get_sampling_discrepancy(logits_ref, logits_score, labels):
    """Monte-Carlo sampling discrepancy (official)."""
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    """Analytic sampling discrepancy (official)."""
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("fast_detect_gpt")
class FastDetectGPTDetector(BaseDetector):
    """Fast-DetectGPT: conditional probability curvature discrepancy.

    The default configuration (``discrepancy_analytic=True`` with the sampling
    model identical to the scoring model) corresponds to Fast-DetectGPT as
    evaluated in the paper.

    Args:
        scoring_model: Pre-built :class:`HFSourceModel` used for scoring.
        scoring_model_name: HuggingFace name of the scoring model (default
            ``gpt2``, i.e. assume the source model is GPT-2).
        sampling_model: Optional pre-built reference (sampling) model.
        sampling_model_name: Optional HuggingFace name of the sampling model;
            defaults to the scoring model.
        discrepancy_analytic: Use the analytic discrepancy (paper default);
            otherwise the Monte-Carlo version.
    """

    requires_source_model = True

    def __init__(
        self,
        scoring_model: Optional[HFSourceModel] = None,
        scoring_model_name: str = "gpt2",
        sampling_model: Optional[HFSourceModel] = None,
        sampling_model_name: Optional[str] = None,
        discrepancy_analytic: bool = True,
        threshold: float = 0.0,
        device: Optional[str] = None,
        model: Optional[HFSourceModel] = None,  # alias for scoring_model
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if scoring_model is None:
            scoring_model = model  # accept model= as an alias
        if scoring_model is None:
            scoring_model = HFSourceModel(scoring_model_name, device=device)
        self.scoring_model = scoring_model
        self.scoring_model_name = scoring_model_name

        if sampling_model is None and sampling_model_name is not None:
            sampling_model = HFSourceModel(sampling_model_name, device=device)
        self.sampling_model = sampling_model
        self.sampling_model_name = sampling_model_name

        # Official: --discrepancy_analytic selects the criterion
        if discrepancy_analytic:
            self.criterion_fn = get_sampling_discrepancy_analytic
            self.criterion_name = "sampling_discrepancy_analytic"
        else:
            self.criterion_fn = get_sampling_discrepancy
            self.criterion_name = "sampling_discrepancy"

    def _same_model(self) -> bool:
        if self.sampling_model is None:
            return True
        if self.sampling_model is self.scoring_model:
            return True
        return self.sampling_model_name == self.scoring_model_name

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            # Original text scoring, following official experiment() body.
            tokenized = self.scoring_model.tokenize(text)
            labels = tokenized.input_ids[:, 1:]
            logits_score = self.scoring_model.model(**tokenized).logits[:, :-1]
            if self._same_model():
                logits_ref = logits_score
            else:
                tokenized = self.sampling_model.tokenize(text)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model.model(**tokenized).logits[:, :-1]
            scores.append(self.criterion_fn(logits_ref, logits_score, labels))
        return np.asarray(scores, dtype=float)
