"""Basic statistical detectors: log-likelihood, rank, log-rank, entropy.

Faithfully reimplemented from the official fast-detect-gpt codebase
(``scripts/baselines.py``, https://github.com/baoguangsheng/fast-detect-gpt),
which itself reproduces the baselines of the DetectGPT paper.

Score direction follows the official evaluation convention: **higher score =
more likely machine-generated**, assuming the scoring model is the (candidate)
source model (white-box setting).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# ---------------------------------------------------------------------------
# Official criterion functions (fast-detect-gpt scripts/baselines.py)
# ---------------------------------------------------------------------------


def get_likelihood(logits, labels):
    """Mean token log-likelihood (official ``baselines.py::get_likelihood``)."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()


def get_rank(logits, labels):
    """Negative mean token rank (official ``baselines.py::get_rank``)."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]
    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), (
        "Expected one match per timestep"
    )
    ranks = ranks.float() + 1  # convert to 1-indexed rank
    return -ranks.mean().item()


def get_logrank(logits, labels):
    """Negative mean log token rank (official ``baselines.py::get_logrank``)."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]
    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), (
        "Expected one match per timestep"
    )
    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()


def get_entropy(logits, labels):
    """Mean predictive entropy (official ``baselines.py::get_entropy``)."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()


def get_positive_logrank(logits, labels):
    """Mean log token rank (positive sign).

    Official DetectLLM (``mbzuai-nlp/DetectLLM``) convention, used by the
    LRR / NPR detectors. Same magnitude as ``-get_logrank``.
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), (
        "Expected one match per timestep"
    )
    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return ranks.mean().item()


# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------


class SingleSourceModelDetector(BaseDetector):
    """Base class for white-box detectors scoring texts with one source model.

    Args:
        model: A pre-built :class:`HFSourceModel`.
        model_name: HuggingFace name of the scoring model, used when ``model``
            is not given.
    """

    requires_source_model = True
    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name or self.DEFAULT_MODEL_NAME, device=device)
        self.model = model

    def criterion(self, logits, labels) -> float:
        raise NotImplementedError

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            # Official: labels = input_ids[:, 1:], logits = model(...).logits[:, :-1]
            logits, input_ids = self.model.get_logits(text)
            scores.append(self.criterion(logits[:, :-1], input_ids[:, 1:]))
        return np.asarray(scores, dtype=float)


@register_detector("log_likelihood")
class LogLikelihoodDetector(SingleSourceModelDetector):
    """Mean log-likelihood under the source model (higher = more likely AI)."""

    def criterion(self, logits, labels):
        return get_likelihood(logits, labels)


@register_detector("rank")
class RankDetector(SingleSourceModelDetector):
    """Negative mean token rank (higher = more likely AI)."""

    def criterion(self, logits, labels):
        return get_rank(logits, labels)


@register_detector("log_rank")
class LogRankDetector(SingleSourceModelDetector):
    """Negative mean log token rank (higher = more likely AI)."""

    def criterion(self, logits, labels):
        return get_logrank(logits, labels)


@register_detector("entropy")
class EntropyDetector(SingleSourceModelDetector):
    """Mean predictive entropy of the source model (official direction)."""

    def criterion(self, logits, labels):
        return get_entropy(logits, labels)
