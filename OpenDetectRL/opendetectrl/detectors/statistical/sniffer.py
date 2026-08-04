"""Sniffer (Li et al., 2023) - AI origin tracing via contrastive multi-model
discrepancy features.

Official code: https://github.com/OpenLMLab/Sniffer
(``sniffer_ppl_calculation.py``, ``sniffer/train.py``, ``sniffer/test.py``).

Method: compute per-word log-likelihoods of text under multiple open-source
LLMs (e.g. GPT-2, GPT-Neo/J, LLaMA), then extract model-wise discrepancy
features for each model pair:

* the mean loss per model (``losses``),
* the fraction of words where one model's loss is below the other's
  (``lt_zero_percents``),
* the standard deviation of the per-word loss differences
  (``std_deviations``),
* the Pearson and Spearman correlation of the per-word losses
  (``pearson_list`` / ``spearmann_list``).

These contrastive features capture the "fingerprint" of the generating model.
The official method trains an MLP classifier on the feature vector
(``hid_dim = model_num + C(model_num, 2) * feat_num`` where ``feat_num = 4``).
When a ``checkpoint_path`` is supplied the detector loads that classifier and
returns the AI-class probability (``1 - P(human)``). Otherwise a zero-shot
variant based on cross-model agreement -- the mean Pearson correlation across
model pairs (higher correlation = models agree more = more likely
AI-generated) -- is used.

Score direction: higher = more likely AI-generated.
"""
import re
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# ---------------------------------------------------------------------------
# Official sentence splitting (sniffer_ppl_calculation.split_sentence)
# ---------------------------------------------------------------------------


def _split_en_sentence(sentence, use_sp=False):
    """Split an English sentence into words and whitespace (official)."""
    pattern = re.compile(r"\S+|\s")
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def _split_cn_sentence(sentence, use_sp=False):
    """Split a Chinese sentence into characters (official)."""
    words = list(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    """Official ``split_sentence``: route to English / Chinese splitter."""
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    chinese_char_count = sum("\u4e00" <= char <= "\u9fff" for char in sentence)
    if chinese_char_count / total_char_count > cn_percent:
        return _split_cn_sentence(sentence, use_sp)
    else:
        return _split_en_sentence(sentence, use_sp)


# ---------------------------------------------------------------------------
# Correlation helpers (numpy, no extra dependencies)
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling, like ``scipy.stats.rankdata``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:  # ties: assign average rank
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient (Pearson on ranks)."""
    return _pearson(_rankdata(x), _rankdata(y))


# ---------------------------------------------------------------------------
# Detector
#


@register_detector("sniffer")
class SnifferDetector(BaseDetector):
    """Sniffer: contrastive multi-model discrepancy features for origin tracing.

    The detector computes per-word log-likelihoods of the text under several
    open-source LLMs and extracts model-wise discrepancy features (the
    differences between model log-likelihoods) for each model pair, following
    the official Sniffer pipeline.

    The official method trains an MLP on these features. When a
    ``checkpoint_path`` is provided the MLP is loaded and the returned score is
    ``1 - P(human)``; otherwise a zero-shot score based on the mean Pearson
    correlation across model pairs is used (higher correlation = more likely
    AI-generated).

    Args:
        models: List of pre-built :class:`HFSourceModel` instances.
        model_names: List of HuggingFace model names (official English
            defaults are ``gpt2-xl``, ``EleutherAI/gpt-neo-2.7B``,
            ``EleutherAI/gpt-j-6B`` and a LLaMA checkpoint; smaller accessible
            defaults are used here -- configure this for the best accuracy).
        checkpoint_path: Path to a trained Sniffer MLP state dict
            (``torch.load``). Requires ``class_num`` to be set.
        class_num: Number of classes the checkpoint was trained with
            (official English: 5, Chinese: 6). Required when
            ``checkpoint_path`` is provided.
        max_length: Token truncation length (official 1024).
        threshold: Decision threshold (higher = more likely AI).
        device: Target device; defaults to CUDA if available, else CPU.
    """

    requires_source_model = True

    #: Number of per-pair discrepancy features (lt_zero, std, pearson, spearman).
    FEAT_NUM = 4

    #: Accessible default models (two diverse open-source families). The
    #: official Sniffer uses four larger models; override ``model_names`` to
    #: match the paper configuration.
    DEFAULT_MODEL_NAMES = ["gpt2", "EleutherAI/gpt-neo-125M"]

    def __init__(
        self,
        models: Optional[List[HFSourceModel]] = None,
        model_names: Optional[List[str]] = None,
        checkpoint_path: Optional[str] = None,
        class_num: Optional[int] = None,
        max_length: int = 1024,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(threshold=threshold, device=device, **kwargs)

        if models is not None:
            self.models = models
            self.model_names = [m.model_name_or_path for m in models]
        else:
            self.model_names = list(model_names or self.DEFAULT_MODEL_NAMES)
            self.models = [HFSourceModel(name, device=device) for name in self.model_names]

        if len(self.models) < 2:
            raise ValueError(
                "Sniffer requires at least two models to compute discrepancy "
                "features; provide `models` or `model_names` with >= 2 entries."
            )

        self.max_length = max_length
        self.model_num = len(self.models)
        self.classifier = None
        self.class_num = class_num
        if checkpoint_path is not None:
            if class_num is None:
                raise ValueError(
                    "`class_num` must be provided when `checkpoint_path` is set."
                )
            self.classifier = self._build_classifier(self.model_num, class_num)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(self.device).eval()

    # ------------------------------------------------------------------
    # Official MLP architecture (sniffer/train.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_classifier(model_num: int, class_num: int) -> torch.nn.Sequential:
        """Official Sniffer MLP: ``Linear(hid -> 300) -> Dropout -> ReLU``
        -> ``Linear(300 -> 300) -> Dropout -> ReLU`` -> ``Linear(300 -> class)``."""
        hid_dim = model_num + int(model_num * (model_num - 1) / 2) * SnifferDetector.FEAT_NUM
        return torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 300),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(300, class_num),
        )

    # ------------------------------------------------------------------
    # Per-word log-likelihood (official PPL calculation)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _get_per_word_losses(
        self, model: HFSourceModel, text: str
    ) -> Tuple[List[float], float]:
        """Return ``(per_word_losses, mean_loss)`` for one model.

        Follows the official ``BBPETokenizerPPLCalc.forward_calc_ppl`` /
        ``SPLlamaTokenizerPPLCalc.forward_calc_ppl`` pipeline: tokenize, compute
        per-token cross-entropy, map tokens to words via byte-level decoding,
        then average to per-word losses.
        """
        words = split_sentence(text)
        # Byte -> word index mapping (official get_bbpe_bytes / get_chars).
        bbs_to_words: List[int] = []
        for idx, word in enumerate(words):
            bbs_to_words.extend([idx] * len(word.encode("utf-8")))

        # Tokenize without special tokens so decoded bytes align with the
        # original text (the official code strips <s> for SentencePiece models).
        enc = model.tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).to(model.device)
        input_ids = enc.input_ids  # [1, seq]
        token_ids = input_ids.squeeze(0).cpu().tolist()

        if len(token_ids) < 2:
            return [], 0.0

        logits = model.model(**enc).logits  # [1, seq, vocab]
        # Per-token cross-entropy (official calc_sent_ppl).
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        ll = loss_func(shift_logits.squeeze(0), shift_labels.squeeze(0))
        ll = ll.cpu().numpy()
        mean_loss = float(ll.mean()) if ll.size else 0.0

        # Map each token's bytes to its loss (official get_bbs_ll). The first
        # token has no preceding context, so its loss is set to 0.
        bbs_ll: List[float] = []
        first_bytes = list(model.tokenizer.decode([token_ids[0]]).encode("utf-8"))
        bbs_ll.extend([0.0] * len(first_bytes))
        for idx in range(1, len(token_ids)):
            tok_bytes = list(
                model.tokenizer.decode([token_ids[idx]]).encode("utf-8")
            )
            bbs_ll.extend([float(ll[idx - 1])] * len(tok_bytes))

        # Group bytes by word -> per-word loss (official calc_token_ppl).
        per_word: List[float] = []
        start = 0
        n = min(len(bbs_to_words), len(bbs_ll))
        while start < n:
            end = start + 1
            while end < len(bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            per_word.append(float(np.mean(bbs_ll[start:end])))
            start = end
        return per_word, mean_loss

    # ------------------------------------------------------------------
    # Feature extraction (official construct_train_features ordering)
    # ------------------------------------------------------------------

    def _extract_features(self, text: str) -> Tuple[List[float], List[float]]:
        """Return ``(feature_vector, pearson_list)``.

        The feature vector follows the official
        ``constructurct_train_features`` ordering::

            losses + lt_zero_percents + std_deviations + pearson_list + spearmann_list

        so its length is ``model_num + C(model_num, 2) * 4``.
        """
        results = [self._get_per_word_losses(m, text) for m in self.models]
        per_word_list = [r[0] for r in results]
        losses = [r[1] for r in results]
        m = self.model_num

        # Align per-word losses to the minimum length across models (different
        # tokenizers / truncation may yield slightly different word coverage).
        min_len = min((len(p) for p in per_word_list), default=0)
        if min_len < 2:
            # Not enough aligned words for meaningful correlations.
            n_pairs = int(m * (m - 1) / 2)
            zeros = [0.0] * n_pairs
            return losses + zeros + zeros + zeros + zeros, zeros

        aligned = [np.asarray(p[:min_len], dtype=float) for p in per_word_list]

        lt_zero: List[float] = []
        std_devs: List[float] = []
        pearsons: List[float] = []
        spearmans: List[float] = []
        for i, j in combinations(range(m), 2):
            diff = aligned[i] - aligned[j]
            lt_zero.append(float(np.mean(diff < 0)))
            std_devs.append(float(diff.std()))
            pearsons.append(_pearson(aligned[i], aligned[j]))
            spearmans.append(_spearman(aligned[i], aligned[j]))

        features = losses + lt_zero + std_devs + pearsons + spearmans
        return features, pearsons

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D array of scores (higher = more likely AI-generated)."""
        scores: List[float] = []
        for text in texts:
            features, pearsons = self._extract_features(text)
            if self.classifier is not None:
                # Trained MLP: P(AI) = 1 - P(human); human is the last class.
                x = torch.tensor([features], dtype=torch.float32, device=self.device)
                logits = self.classifier(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                score = float(1.0 - probs[-1])
            else:
                # Zero-shot: mean Pearson correlation across model pairs.
                # Higher correlation = models agree more = more likely AI.
                score = float(np.mean(pearsons)) if pearsons else 0.0
            scores.append(score)
        return np.asarray(scores, dtype=float)
