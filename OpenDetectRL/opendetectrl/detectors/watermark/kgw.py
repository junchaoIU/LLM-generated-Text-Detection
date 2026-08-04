"""Watermark-based detectors: KGW, Unigram, SynthID-Text.

KGW: Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
Official code: https://github.com/jwkirchenbauer/lm-watermarking
(``watermark_processor.py``)

Unigram: Zhao et al., "Provable Robust Watermarking for AI-Generated Text", 2023.

SynthID-Text: Dathathri et al., "SynthID-Text: Identifying AI-Generated Text
Content", Nature 2024. Now integrated in HuggingFace transformers >= 4.46.

All watermark detectors require the watermark key/config used at generation
time, plus access to the model's tokenizer (and logits for SynthID-Text).
"""
import collections
from math import sqrt
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# ---------------------------------------------------------------------------
# KGW Watermark Detector (official watermark_processor.py)
# ---------------------------------------------------------------------------

# Just a large prime number to create a rng seed with sufficient bit width
DEFAULT_HASH_KEY = 15485863


class WatermarkBase:
    """Core watermark primitives from the official ``watermark_processor.py``."""

    def __init__(
        self,
        vocab,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = DEFAULT_HASH_KEY,
        select_green_tokens: bool = True,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None):
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")

    def _get_greenlist_ids(self, input_ids: torch.LongTensor):
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        if self.select_green_tokens:
            greenlist_ids = vocab_permutation[:greenlist_size]
        else:
            greenlist_ids = vocab_permutation[self.vocab_size - greenlist_size:]
        return greenlist_ids


@register_detector("kgw_watermark")
class KGWWatermarkDetector(BaseDetector):
    """KGW watermark detector (Kirchenbauer et al., ICML 2023).

    Official z-score statistic::

        z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))

    where G = green-list token count, T = total scored tokens. Higher z =
    more likely watermarked (AI-generated).

    Args:
        model: Pre-built :class:`HFSourceModel` (needed for tokenizer).
        model_name: HuggingFace name (default ``gpt2``).
        gamma: Green-list fraction (official default 0.5).
        delta: Watermark strength (official default 2.0; not used for detection).
        hash_key: Secret key for seeding RNG (official 15485863).
        seeding_scheme: Seeding scheme (official ``simple_1``).
        z_threshold: Decision threshold (official default 4.0).
        ignore_repeated_bigrams: Official default True.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = DEFAULT_HASH_KEY,
        seeding_scheme: str = "simple_1",
        z_threshold: float = 4.0,
        ignore_repeated_bigrams: bool = True,
        threshold: float = 4.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.tokenizer = model.tokenizer
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.seeding_scheme = seeding_scheme
        self.z_threshold = z_threshold
        self.ignore_repeated_bigrams = ignore_repeated_bigrams

        # Initialize the watermark base
        vocab = list(range(len(self.tokenizer)))
        self._wm = WatermarkBase(
            vocab=vocab,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            hash_key=hash_key,
        )
        self._wm.rng = torch.Generator(device=self.model.device)
        if seeding_scheme == "simple_1":
            self._wm.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")

    def _compute_z_score(self, observed_count, T):
        """Official ``_compute_z_score``."""
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        return numer / denom

    def _compute_p_value(self, z):
        """Official ``_compute_p_value``."""
        from scipy.stats import norm
        return norm.sf(z)

    def _score_sequence(self, input_ids: torch.Tensor):
        """Official ``_score_sequence`` (simplified, no normalizers)."""
        if self.ignore_repeated_bigrams:
            from nltk.util import ngrams

            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            green_token_count = 0
            for bigram in freq.keys():
                prefix = torch.tensor([bigram[0]], device=self.model.device)
                greenlist_ids = self._wm._get_greenlist_ids(prefix)
                if bigram[1] in greenlist_ids:
                    green_token_count += 1
        else:
            num_tokens_scored = len(input_ids) - self._wm.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError("Must have at least 1 token to score.")
            green_token_count = 0
            for idx in range(self._wm.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._wm._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1

        z = self._compute_z_score(green_token_count, num_tokens_scored)
        p = self._compute_p_value(z)
        return {
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "z_score": z,
            "p_value": p,
        }

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            tokenized = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized["input_ids"][0].to(self.model.device)
            # Remove BOS token if present
            if input_ids[0] == self.tokenizer.bos_token_id:
                input_ids = input_ids[1:]
            result = self._score_sequence(input_ids)
            scores.append(result["z_score"])
        return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Unigram Watermark Detector (Zhao et al., 2023)
# ---------------------------------------------------------------------------


@register_detector("unigram_watermark")
class UnigramWatermarkDetector(BaseDetector):
    """Unigram watermark detector (Zhao et al., 2023).

    Unlike KGW, the green/red partition is **fixed** (context-independent):
    a single random partition of the vocabulary is generated from the key
    and reused at every position. Detection is the same z-score statistic::

        z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))

    Args:
        model: Pre-built :class:`HFSourceModel` (needed for tokenizer).
        model_name: HuggingFace name (default ``gpt2``).
        gamma: Green-list fraction (official default 0.5).
        hash_key: Secret key for the fixed partition.
        z_threshold: Decision threshold (official default 4.0).
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        gamma: float = 0.5,
        hash_key: int = DEFAULT_HASH_KEY,
        z_threshold: float = 4.0,
        threshold: float = 4.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.tokenizer = model.tokenizer
        self.gamma = gamma
        self.hash_key = hash_key
        self.z_threshold = z_threshold

        # Build the fixed green/red partition
        vocab_size = len(self.tokenizer)
        gen = torch.Generator()
        gen.manual_seed(hash_key)
        greenlist_size = int(vocab_size * gamma)
        vocab_permutation = torch.randperm(vocab_size, generator=gen)
        self._greenlist = set(vocab_permutation[:greenlist_size].tolist())

    def _compute_z_score(self, green_count, T):
        numer = green_count - self.gamma * T
        denom = sqrt(T * self.gamma * (1 - self.gamma))
        return numer / denom if denom > 0 else 0.0

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # Remove BOS if present
            if input_ids and input_ids[0] == self.tokenizer.bos_token_id:
                input_ids = input_ids[1:]
            T = len(input_ids)
            if T == 0:
                scores.append(0.0)
                continue
            green_count = sum(1 for tid in input_ids if tid in self._greenlist)
            scores.append(self._compute_z_score(green_count, T))
        return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# SynthID-Text Detector (Dathathri et al., Nature 2024)
# ---------------------------------------------------------------------------


@register_detector("synthid_text")
class SynthIDTextDetector(BaseDetector):
    """SynthID-Text watermark detector (Google DeepMind, 2024).

    Uses the official ``synthid`` library / HF transformers integration
    (transformers >= 4.46). The detector computes a mean log-likelihood-ratio
    and a Bayesian significance test (p-value).

    Args:
        model: Pre-built :class:`HFSourceModel` (the watermarked model).
        model_name: HuggingFace name.
        watermark_config: ``SynthIDTextWatermarkingConfig`` or kwargs to build one.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gemma-2b",
        watermark_config=None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model

        # Build or use provided watermark config
        if watermark_config is None:
            try:
                from transformers import SynthIDTextWatermarkingConfig
                watermark_config = SynthIDTextWatermarkingConfig(
                    keys=[kwargs.get("key", 0)],
                    ngram_len=kwargs.get("ngram_len", 5),
                )
            except ImportError:
                raise ImportError(
                    "SynthID-Text requires transformers >= 4.46. "
                    "Install with: pip install transformers>=4.46"
                )
        self.watermark_config = watermark_config

        try:
            from transformers import SynthIDTextWatermarkDetector
            self._detector = SynthIDTextWatermarkDetector(
                detector_keys=watermark_config.keys,
                model=model.model,
                tokenizer=model.tokenizer,
                ngram_len=watermark_config.ngram_len,
            )
        except ImportError:
            self._detector = None

    def score(self, texts) -> np.ndarray:
        if self._detector is None:
            raise RuntimeError("SynthID-Text detector not available (need transformers >= 4.46)")
        scores = []
        for text in texts:
            result = self._detector.detect(text)
            # SynthID returns (predicted, p_value)
            # Higher p-value = more likely watermarked
            # We return -p_value so higher score = more AI (consistent)
            p_value = result[1] if len(result) > 1 else 0.5
            scores.append(1.0 - p_value)
        return np.asarray(scores, dtype=float)
