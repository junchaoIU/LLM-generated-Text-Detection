"""Binoculars (Hans et al., ICML 2024).

Official code: https://github.com/ahans30/Binoculars
(``binoculars/detector.py`` + ``binoculars/metrics.py``).

Score: ``Binoculars(x) = PPL_performer(x) / X-PPL_observer,performer(x)``.
Official thresholds (Falcon-7B + Falcon-7B-Instruct, bfloat16):

* accuracy mode: 0.9015310749276843 (optimized for F1)
* low-fpr mode:  0.8536432310785527 (optimized for low FPR at 0.01%)

The official decision rule is ``score < threshold => AI-generated``. To keep
the library-wide convention (higher = more likely AI), :meth:`score` returns
the negated raw Binoculars score and the default threshold is negated as
well; ``metadata["binoculars_raw"]`` in :meth:`predict` keeps the original
(raw) score.
"""
from typing import List, Optional, Union

import numpy as np
import torch
import transformers

from ...core.base import BaseDetector, DetectionResult
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

# ---------------------------------------------------------------------------
# Official metric functions (binoculars/metrics.py)
# ---------------------------------------------------------------------------

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(
            ~shifted_attention_mask.bool(), float("nan")
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
        ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()
    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)
    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())
    return agg_ce


def assert_tokenizer_consistency(model_id_1, model_id_2):
    """Official ``binoculars/utils.py::assert_tokenizer_consistency``."""
    from transformers import AutoTokenizer

    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1).vocab
        == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("binoculars")
class BinocularsDetector(BaseDetector):
    """Binoculars: cross-perplexity ratio between observer and performer.

    Args:
        observer_name_or_path: Official default ``tiiuae/falcon-7b``.
        performer_name_or_path: Official default ``tiiuae/falcon-7b-instruct``.
        observer_model / performer_model: Pre-built :class:`HFSourceModel`
            instances (skip loading; tokenizer-consistency check is skipped).
        use_bfloat16: Official default True.
        max_token_observed: Official default 512 (truncation length).
        mode: ``"low-fpr"`` (official default) or ``"accuracy"``.
        device: Device for both models (official uses two devices if
            available; a single device is used here by default).
    """

    requires_source_model = True

    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        observer_model: Optional[HFSourceModel] = None,
        performer_model: Optional[HFSourceModel] = None,
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        official_threshold = self._official_threshold(mode)
        # Library convention: higher = AI. Raw Binoculars decides
        # "raw < threshold => AI", so the unified threshold is negated.
        if threshold is None:
            threshold = -official_threshold
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.mode = mode
        self.official_threshold = official_threshold
        self.max_token_observed = max_token_observed

        if observer_model is not None and performer_model is not None:
            self.observer_model = observer_model
            self.performer_model = performer_model
            self.tokenizer = observer_model.tokenizer
        else:
            assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
            torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            self.observer_model = HFSourceModel(
                observer_name_or_path, device=device, torch_dtype=torch_dtype
            )
            self.performer_model = HFSourceModel(
                performer_name_or_path, device=device, torch_dtype=torch_dtype
            )
            self.tokenizer = self.observer_model.tokenizer

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _official_threshold(mode: str) -> float:
        if mode == "low-fpr":
            return BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            return BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def change_mode(self, mode: str) -> None:
        """Official ``change_mode``: switch thresholds at runtime."""
        self.mode = mode
        self.official_threshold = self._official_threshold(mode)
        self.threshold = -self.official_threshold

    def _tokenize(self, batch: List[str]) -> transformers.BatchEncoding:
        """Official ``_tokenize``."""
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding):
        observer_logits = self.observer_model.model(**encodings).logits
        performer_logits = self.performer_model.model(**encodings).logits
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[List[str], str]) -> Union[float, List[float]]:
        """Raw Binoculars score (official ``compute_score``). Lower = more AI."""
        batch = [input_text] if isinstance(input_text, str) else list(input_text)
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def score(self, texts) -> np.ndarray:
        """Negated raw Binoculars score so that higher = more likely AI."""
        return -np.asarray(self.compute_score(list(texts)), dtype=float)

    def predict(self, texts) -> List[DetectionResult]:
        """Predictions following the official rule ``raw < threshold => AI``,
        with the raw score exposed in ``metadata``."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        raw_scores = np.asarray(self.compute_score(texts), dtype=float)
        results = []
        for raw in raw_scores:
            is_ai = bool(raw < self.official_threshold)
            results.append(
                DetectionResult(
                    score=float(-raw),
                    label="ai" if is_ai else "human",
                    confidence=1.0,
                    metadata={"binoculars_raw": float(raw), "mode": self.mode},
                )
            )
        return results
