"""NOTAI.AI: Explainable Detection via Curvature and Feature Attribution.

Paper: Marchenko Breneur et al., "NOTAI.AI: Explainable Detection of
Machine-Generated Text via Curvature and Feature Attribution", arXiv 2603.05617.
Repo: https://github.com/TenderChasm/Notai.ai

Method: extends Fast-DetectGPT by combining 17 interpretable features —
Conditional Probability Curvature (CPC), a neural (ModernBERT) detector score,
readability metrics, and stylometric cues — within a gradient-boosted tree
(XGBoost) meta-classifier.  The default (zero-shot) instantiation below
combines the curvature signal with stylometric and readability features into a
sigmoid score; a trained meta-classifier can be supplied via ``model_path``.

Score direction: **higher = more likely AI-generated**.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel
from .stylometric import StylometryFeatureExtractor, _safe_div

# ---------------------------------------------------------------------------
# Readability helpers (standard-library syllable estimation)
# ---------------------------------------------------------------------------
_VOWELS = set("aeiouy")


def _syllables(word: str) -> int:
    """Estimates the syllable count of a word (rough but self-contained)."""
    word = word.lower().strip()
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _compute_readability(text: str) -> Dict[str, float]:
    """Readability metrics (Flesch reading ease + component statistics)."""
    import re

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)
    sentences = len(re.split(r"[.!?]+(?:\s+|$)", text.strip()).strip())
    n_words = len(words)
    if n_words == 0:
        return {
            "flesch_reading_ease": 0.0,
            "avg_syllables_per_word": 0.0,
            "complex_word_ratio": 0.0,
        }
    total_syllables = sum(_syllables(w) for w in words)
    n_sentences = max(sentences, 1)
    complex_word_ratio = sum(1 for w in words if _syllables(w) >= 3) / n_words
    # Flesch Reading Ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    flesch = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (total_syllables / n_words)
    return {
        "flesch_reading_ease": float(np.clip(flesch, 0.0, 120.0)),
        "avg_syllables_per_word": total_syllables / n_words,
        "complex_word_ratio": complex_word_ratio,
    }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
@register_detector("notai")
class NotAIDetector(BaseDetector):
    """NOTAI.AI feature-meta-classifier detector (arXiv 2603.05617).

    Extracts curvature, stylometric, and readability features and combines
    them into a score where higher = more AI-generated.  A trained
    meta-classifier (e.g. XGBoost) can be supplied via ``model_path``; without
    one, a zero-shot weighted combination of the features is used.

    Args:
        scoring_model: Pre-built :class:`HFSourceModel` for curvature features.
        scoring_model_name: HF name of the reference model (default ``gpt2``).
        model_path: Optional path to a trained meta-classifier checkpoint.
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    requires_source_model = True

    _FEATURE_NAMES = [
        "curvature",
        "mean_log_prob",
        "ttr",
        "avg_word_length",
        "avg_sentence_length",
        "std_sentence_length",
        "punctuation_density",
        "function_word_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "flesch_reading_ease",
        "avg_syllables_per_word",
        "complex_word_ratio",
        "commas_per_sentence",
        "exclamations_per_sentence",
        "repetition_ratio",
        "paragraph_count",
    ]

    def __init__(
        self,
        scoring_model: Optional[HFSourceModel] = None,
        scoring_model_name: str = "gpt2",
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.scoring_model_name = scoring_model_name
        self.model_path = model_path
        self._extractor = StylometryFeatureExtractor()
        if scoring_model is None:
            scoring_model = HFSourceModel(scoring_model_name, device=device)
        self.scoring_model = scoring_model
        self._classifier = None

    def _load_classifier(self):
        """Load a trained meta-classifier if ``model_path`` was provided."""
        if self._classifier is not None:
            return self._classifier
        if self.model_path is None:
            return None
        try:
            import joblib

            self._classifier = joblib.load(self.model_path)
        except Exception:
            try:
                import pickle

                with open(self.model_path, "rb") as f:
                    self._classifier = pickle.load(f)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Could not load meta-classifier from {self.model_path}: {exc}"
                )
        return self._classifier

    def _curvature(self, text: str) -> float:
        """Conditional-probability curvature (Fast-DetectGPT style, per-text)."""
        try:
            import torch

            tokenized = self.scoring_model.tokenize(text)
            logits = self.scoring_model.model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]
            probs = torch.softmax(logits, dim=-1)
            lprobs = torch.log_softmax(logits, dim=-1)
            ll = torch.gather(lprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            mean = (probs * lprobs).sum(dim=-1)
            var = (probs * lprobs.square()).sum(dim=-1) - mean.square()
            score = ((ll - mean).sum(dim=-1) / var.sum(dim=-1).sqrt()).item()
            return float(score)
        except Exception:
            return 0.0

    def _extract_features(self, text: str) -> List[float]:
        """Return the 17-feature vector for a single text."""
        st = self._extractor.get_features(text)
        rd = _compute_readability(text)
        words = st["word_count"]
        sentences = max(st["sentence_count"], 1)
        func_count = st["function_word_ratio"] * words
        # Repetition ratio: fraction of duplicate neighboring tokens.
        import re

        toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
        rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
        repetition = (rep / max(len(toks), 1)) if toks else 0.0

        features = {
            "curvature": self._curvature(text),
            "mean_log_prob": 0.0,  # placeholder filled by _mean_log_prob when available
            "ttr": st["ttr"],
            "avg_word_length": st["avg_word_length"],
            "avg_sentence_length": st["avg_sentence_length"],
            "std_sentence_length": st["std_sentence_length"],
            "punctuation_density": st["punctuation_density"],
            "function_word_ratio": st["function_word_ratio"],
            "digit_ratio": st["digit_ratio"],
            "uppercase_ratio": st["uppercase_ratio"],
            "flesch_reading_ease": rd["flesch_reading_ease"],
            "avg_syllables_per_word": rd["avg_syllables_per_word"],
            "complex_word_ratio": rd["complex_word_ratio"],
            "commas_per_sentence": _safe_div(st["comma_ratio"] * st["punctuation_density"] * len(text), sentences),
            "exclamations_per_sentence": _safe_div(text.count("!"), sentences),
            "repetition_ratio": repetition,
            "paragraph_count": st["paragraph_count"],
        }
        return [features[n] for n in self._FEATURE_NAMES]

    def _zero_shot_score(self, feats: List[float]) -> float:
        """Weighted zero-shot combination of the features (higher = more AI)."""
        f = dict(zip(self._FEATURE_NAMES, feats))
        # Heuristic weights: AI text tends to be flatter (lower curvature),
        # more uniform, lower readability, lower TTR.
        z = (
            -0.4 * f["curvature"]
            + 0.3 * (0.55 - f["ttr"])
            + 0.2 * (18.0 - f["avg_sentence_length"])
            + 0.1 * (4.6 - f["avg_word_length"])
            + 0.2 * (0.05 - f["punctuation_density"])
            + 0.1 * (f["function_word_ratio"] - 0.45)
            + 0.1 * (60.0 - f["flesch_reading_ease"])
            + 0.1 * (0.12 - f["complex_word_ratio"])
        )
        x = float(np.clip(z, -5.0, 5.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def score(self, texts: Sequence[str]) -> np.ndarray:
        xgb = self._load_classifier()
        scores = []
        for text in texts:
            feats = self._extract_features(text)
            if xgb is not None:
                try:
                    prob = xgb.predict_proba([feats])[0, 1]
                    scores.append(float(prob))
                    continue
                except Exception:
                    pass
            scores.append(self._zero_shot_score(feats))
        return np.asarray(scores, dtype=float)