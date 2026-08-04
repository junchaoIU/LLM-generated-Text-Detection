"""Stylometric Detection (Kumarage et al., 2022).

Official code: https://github.com/TSKumarage/Stylo-Det-AI-Gen-Twitter-Timelines
(``Stylometry Models/T1_Stylometry_LM_Fusion.ipynb`` and
``Stylometry Models/T2_StyloCPA.ipynb``).

Method: extract writing-style (stylometric) features from text — character,
word, sentence, punctuation, and function-word statistics — and combine them
into a composite score where **higher = more likely AI-generated**.

This is a pure feature-based detector that does NOT require a language model.
Feature extraction follows the official ``Stylometry`` class feature groups
(phraseology, diversity, and punctuation analysis) and augments them with
standard character-level ratios and function-word frequency. To keep the
detector self-contained, all tokenization is performed with the standard
library (no nltk / lexicalrichness / readability downloads required).

The final score is a logistic-regression-like weighted sum of feature
z-scores measuring deviation from typical human-writing baselines, passed
through a sigmoid so the score lies in ``[0, 1]`` (higher = more AI-like).
"""
import re
import string
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector

# ---------------------------------------------------------------------------
# Tokenization helpers (standard-library only, no nltk downloads needed)
# ---------------------------------------------------------------------------

# A compact list of common English function words (closed-class words).
_FUNCTION_WORDS = frozenset(
    """
    a an the this that these those
    and or but nor yet so for as if because while whereas although though
    since until before after
    of in on at to for with by from about into onto upon over under through
    across behind between among
    is am are was were be been being
    has have had do does did
    i you he she it we they me him her us them my your his its our their
    mine yours hers ours theirs
    not no
    will would shall should can could may might must ought
    which who whom whose what where when why how
    here there now then
    some any all each every few more most other such
    about above below up down out off
    """.split()
)

_PUNCTUATION = set(string.punctuation)

# Sentence-end pattern: terminators followed by whitespace or end of string.
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
# Word pattern: runs of alphanumeric characters (apostrophes keep contractions).
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
# A line must contain at least one letter or digit to count as a paragraph.
_NON_PUNCT_RE = re.compile(r".*[A-Za-z0-9].*")


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into words (alphanumeric runs, contractions kept)."""
    return _WORD_RE.findall(text)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using terminator punctuation / newlines."""
    text = text.strip()
    if not text:
        return []
    # Normalize whitespace so paragraph breaks do not split sentences.
    text = re.sub(r"\s+", " ", text)
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _safe_div(a: float, b: float) -> float:
    """Safe float division (returns ``0.0`` when ``b == 0``)."""
    return a / b if b else 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


# ---------------------------------------------------------------------------
# Stylometry feature extractor (ports the official Stylometry feature groups)
# ---------------------------------------------------------------------------


class StylometryFeatureExtractor:
    """Extract stylometric features from a document.

    Mirrors the official ``Stylometry`` class feature groups: phraseology
    (word / sentence / paragraph counts and per-unit statistics), diversity
    (vocabulary richness), and punctuation analysis (total and
    per-punctuation ratios), augmented with character-level ratios and
    function-word frequency.
    """

    #: Special punctuation marks tracked individually (official order).
    SPECIAL_PUNCTS = ["!", "'", ",", "-", ":", ";", "?", "@", '"', "=", "#"]

    def word_count(self, document: str) -> int:
        """Number of word tokens (must contain a letter or digit)."""
        return len(_tokenize_words(document))

    def sentence_count(self, document: str) -> int:
        """Number of sentences (must contain a letter or digit)."""
        return len(_split_sentences(document))

    def paragraph_count(self, document: str) -> int:
        """Number of non-empty paragraphs (lines with a letter/digit)."""
        return len([p for p in document.splitlines() if _NON_PUNCT_RE.match(p)])

    def word_count_sent(self, document: str):
        """Return ``(mean, std)`` of word counts per sentence (official logic)."""
        sentences = _split_sentences(document)
        counts = [self.word_count(s) for s in sentences]
        if not counts:
            return 0.0, 0.0
        mean = sum(counts) / len(counts)
        var = sum((x - mean) ** 2 for x in counts) / len(counts)
        return mean, var ** 0.5

    def total_punc_count(self, document: str) -> int:
        """Total count of punctuation characters (official ``total_punc_count``)."""
        return sum(1 for ch in document if ch in _PUNCTUATION)

    def special_punc_count(self, document: str) -> List[float]:
        """Ratio of each special punctuation over total punctuation.

        Official ``special_punc_count``: returns ``count / total_puncts``.
        """
        total = self.total_punc_count(document)
        if total == 0:
            return [0.0 for _ in self.SPECIAL_PUNCTS]
        return [document.count(p) / total for p in self.SPECIAL_PUNCTS]

    def special_punc_count_sent(self, document: str) -> List[float]:
        """Mean count of each special punctuation per sentence.

        Official ``special_punc_count_sent``: sums counts over sentences and
        divides by the number of sentences.
        """
        sentences = _split_sentences(document)
        if not sentences:
            return [0.0 for _ in self.SPECIAL_PUNCTS]
        counts = [0.0 for _ in self.SPECIAL_PUNCTS]
        for sent in sentences:
            for i, punct in enumerate(self.SPECIAL_PUNCTS):
                counts[i] += sent.count(punct)
        return [c / len(sentences) for c in counts]

    def get_features(self, document: str) -> Dict[str, float]:
        """Return a dict of named stylometric features for ``document``."""
        if not document or not document.strip():
            document = "empty"

        words = _tokenize_words(document)
        n_words = len(words)
        n_chars = len(document)
        letters = sum(1 for ch in document if ch.isalpha())
        digits = sum(1 for ch in document if ch.isdigit())
        upper = sum(1 for ch in document if ch.isupper())
        spaces = sum(1 for ch in document if ch.isspace())

        word_counts_per_sent = [self.word_count(s) for s in _split_sentences(document)]

        # ---- Character-level features ----
        char_count = n_chars
        letter_ratio = _safe_div(letters, n_chars)
        digit_ratio = _safe_div(digits, n_chars)
        uppercase_ratio = _safe_div(upper, max(letters, 1))
        whitespace_ratio = _safe_div(spaces, n_chars)

        # ---- Word-level features ----
        avg_word_length = _safe_div(sum(len(w) for w in words), n_words)
        # Vocabulary richness: type-token ratio (TTR).
        ttr = _safe_div(len({w.lower() for w in words}), n_words)

        # ---- Sentence-level features ----
        if word_counts_per_sent:
            mean_sent_len = sum(word_counts_per_sent) / len(word_counts_per_sent)
            var = sum((x - mean_sent_len) ** 2 for x in word_counts_per_sent) / len(
                word_counts_per_sent
            )
            std_sent_len = var ** 0.5
        else:
            mean_sent_len = 0.0
            std_sent_len = 0.0

        # ---- Punctuation features ----
        total_punct = self.total_punc_count(document)
        punct_density = _safe_div(total_punct, n_chars)
        special_ratios = self.special_punc_count(document)
        comma_ratio = special_ratios[self.SPECIAL_PUNCTS.index(",")]
        period_ratio = _safe_div(document.count("."), max(total_punct, 1))

        # ---- Function-word frequency ----
        func_count = sum(1 for w in words if w.lower() in _FUNCTION_WORDS)
        function_word_ratio = _safe_div(func_count, n_words)

        return {
            # character-level
            "char_count": char_count,
            "letter_ratio": letter_ratio,
            "digit_ratio": digit_ratio,
            "uppercase_ratio": uppercase_ratio,
            "whitespace_ratio": whitespace_ratio,
            # word-level
            "word_count": n_words,
            "avg_word_length": avg_word_length,
            "ttr": ttr,
            # sentence-level
            "sentence_count": len(_split_sentences(document)),
            "avg_sentence_length": mean_sent_len,
            "std_sentence_length": std_sent_len,
            # punctuation
            "punctuation_density": punct_density,
            "comma_ratio": comma_ratio,
            "period_ratio": period_ratio,
            # function words
            "function_word_ratio": function_word_ratio,
            # phraseology
            "paragraph_count": self.paragraph_count(document),
        }


# ---------------------------------------------------------------------------
# Scoring: weighted deviation from typical human-writing baselines
# ---------------------------------------------------------------------------

#: Human-writing baseline (mean, std) and AI-direction sign per feature.
#:  sign = +1  -> higher-than-human value is more AI-like;
#:  sign = -1  -> lower-than-human value is more AI-like.
_BASELINE: Dict[str, Dict[str, float]] = {
    "letter_ratio": {"mean": 0.78, "std": 0.06, "sign": 1.0},
    "digit_ratio": {"mean": 0.025, "std": 0.02, "sign": -1.0},
    "uppercase_ratio": {"mean": 0.05, "std": 0.04, "sign": -1.0},
    "avg_word_length": {"mean": 4.6, "std": 0.6, "sign": 1.0},
    "ttr": {"mean": 0.55, "std": 0.12, "sign": -1.0},
    "avg_sentence_length": {"mean": 18.0, "std": 6.0, "sign": 1.0},
    "std_sentence_length": {"mean": 8.0, "std": 3.0, "sign": -1.0},
    "punctuation_density": {"mean": 0.045, "std": 0.02, "sign": -1.0},
    "comma_ratio": {"mean": 0.15, "std": 0.07, "sign": 1.0},
    "period_ratio": {"mean": 0.30, "std": 0.10, "sign": -1.0},
    "function_word_ratio": {"mean": 0.45, "std": 0.06, "sign": 1.0},
}

#: Default per-feature weights for the weighted z-score sum.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "letter_ratio": 1.0,
    "digit_ratio": 0.7,
    "uppercase_ratio": 0.8,
    "avg_word_length": 1.0,
    "ttr": 1.2,
    "avg_sentence_length": 1.0,
    "std_sentence_length": 1.1,
    "punctuation_density": 0.9,
    "comma_ratio": 0.8,
    "period_ratio": 0.6,
    "function_word_ratio": 1.0,
}


@register_detector("stylometric")
class StylometricDetector(BaseDetector):
    """Stylometric AI-text detector (Kumarage et al., 2022).

    Extracts writing-style features (character, word, sentence, punctuation,
    and function-word statistics) and combines them into a composite score
    where **higher = more AI-generated**. The score is a
    logistic-regression-like weighted sum of z-score deviations from typical
    human-writing baselines, passed through a sigmoid.

    This is a pure feature-based detector — no language model is required.

    Args:
        threshold: Decision threshold on the ``[0, 1]`` score (default 0.5).
        device: Ignored; kept for API compatibility with other detectors.
        baseline: Optional override of the human baseline (``mean``/``std``/
            ``sign``) per feature name.
        weights: Optional override of the per-feature weights.
        temperature: Scaling of the weighted z-score sum before the sigmoid
            (higher -> sharper decision boundary; default 2.0).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        device: Optional[str] = None,
        baseline: Optional[Dict[str, Dict[str, float]]] = None,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.extractor = StylometryFeatureExtractor()
        self.baseline = dict(_BASELINE)
        if baseline:
            for name, spec in baseline.items():
                self.baseline[name] = dict(spec)
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.temperature = temperature

    def _score_text(self, text: str) -> float:
        """Composite stylometric score for a single text in ``[0, 1]``."""
        features = self.extractor.get_features(text)

        z_sum = 0.0
        total_weight = 0.0
        for name, spec in self.baseline.items():
            value = features.get(name, 0.0)
            std = spec.get("std") or 1.0
            weight = self.weights.get(name, 0.0)
            z_sum += weight * spec["sign"] * ((value - spec["mean"]) / std)
            total_weight += weight

        # Weighted-average z-score so magnitudes stay comparable.
        z = z_sum / (total_weight or 1.0)
        return _sigmoid(self.temperature * z)

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D array of scores (higher = more likely AI-generated)."""
        scores = [self._score_text(t) for t in texts]
        return np.asarray(scores, dtype=float)
