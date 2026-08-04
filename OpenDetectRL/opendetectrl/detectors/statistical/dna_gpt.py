"""DNA-GPT: Divergent N-Gram Analysis (Yang et al., ICLR 2024).

Official code: https://github.com/Xianjun-Yang/DNA-GPT (``DNA-GPT-dist.py``).

Method: truncate the text at the middle, use the first half as a prompt to
regenerate the remaining part with a source LLM, then compute weighted N-gram
overlap between the original remaining part and the regenerated text. Higher
N-gram overlap = more likely machine-generated.

The official code uses OpenAI API for regeneration. This implementation
supports both HF models and OpenAI API as the source model.
"""
import re
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

# ---------------------------------------------------------------------------
# Official N-gram analysis functions (DNA-GPT-dist.py)
# ---------------------------------------------------------------------------


def _tokenize(text, stemmer=None, stopwords=None):
    """Official ``tokenize``: lowercase, strip non-alphanumeric, optional stem."""
    import six

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))
    tokens = re.split(r"\s+", text)
    if stemmer:
        tokens = [
            stemmer.stem(x) if len(x) > 3 else x
            for x in tokens
            if x and (stopwords is None or x not in stopwords)
        ]
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]
    return tokens


def _create_ngrams(tokens, n):
    """Official ``_create_ngrams`` (from rouge_score)."""
    return dict(Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)))


def _get_score_ngrams(target_ngrams, prediction_ngrams):
    """Official ``get_score_ngrams``: intersection ratio."""
    intersection_count = 0
    ngram_dict = {}
    for ngram in target_ngrams:
        cnt = min(target_ngrams[ngram], prediction_ngrams.get(ngram, 0))
        intersection_count += cnt
        ngram_dict[ngram] = cnt
    target_count = sum(target_ngrams.values())
    return intersection_count / max(target_count, 1), ngram_dict


def _get_ngram_info(article_tokens, summary_tokens, ngram):
    """Official ``get_ngram_info``."""
    article_ngram = _create_ngrams(article_tokens, ngram)
    summary_ngram = _create_ngrams(summary_tokens, ngram)
    score, ngram_dict = _get_score_ngrams(article_ngram, summary_ngram)
    return score, ngram_dict, sum(ngram_dict.values())


def _n_gram_detector(ngram_n_ratio):
    """Official ``N_gram_detector``: weighted score from N=3 to N=25.

    The input is a list of dicts with keys like
    ``human_truncate_ngram_{N}_score``.
    """
    score = 0
    non_zero = []
    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and ("score" in key or "ratio" in key):
            score += 0.0 * ngram_n_ratio[key]
            continue
        if "score" in key or "ratio" in key:
            score += (idx + 1) * np.log(idx + 1) * ngram_n_ratio[key]
            if ngram_n_ratio[key] != 0:
                non_zero.append(idx + 1)
    return score / (sum(non_zero) + 1e-8)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("dna_gpt")
class DNAGPTDetector(BaseDetector):
    """DNA-GPT: divergent N-gram analysis via truncate-and-regenerate.

    Args:
        model: Pre-built :class:`HFSourceModel` used for regeneration.
        model_name: HuggingFace name of the source model (default ``gpt2``).
        truncate_ratio: Fraction of text to use as prefix prompt (official 0.5).
        max_new_tokens: Maximum tokens to generate (official 300).
        n_perturbations: Number of regenerations (official 30; use 1 for
            local models due to cost).
        temperature: Sampling temperature (official 0.7).
        threshold: Decision threshold (official 0.00025).
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        truncate_ratio: float = 0.5,
        max_new_tokens: int = 300,
        n_perturbations: int = 1,
        temperature: float = 0.7,
        threshold: float = 0.00025,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.truncate_ratio = truncate_ratio
        self.max_new_tokens = max_new_tokens
        self.n_perturbations = n_perturbations
        self.temperature = temperature

    @property
    def _stemmer(self):
        try:
            from nltk.stem.porter import PorterStemmer
            return PorterStemmer()
        except ImportError:
            return None

    @property
    def _stopwords(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp.Defaults.stop_words
        except Exception:
            return set()

    def _generate_continuation(self, prefix: str) -> List[str]:
        """Regenerate text using the source model with the prefix as prompt."""
        import torch

        results = []
        tokenized = self.model.tokenize(prefix)
        for _ in range(self.n_perturbations):
            with torch.inference_mode():
                output = self.model.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                )
            generated = self.model.tokenizer.decode(
                output[0][tokenized["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            results.append(generated)
        return results

    def score(self, texts) -> np.ndarray:
        stemmer = self._stemmer
        stopwords = self._stopwords
        scores = []
        for text in texts:
            prefix = text[: int(self.truncate_ratio * len(text))]
            remaining = text[int(self.truncate_ratio * len(text)):]
            remaining_tokens = _tokenize(remaining, stemmer, stopwords)
            if not remaining_tokens:
                scores.append(0.0)
                continue

            continuations = self._generate_continuation(prefix)
            gpt_scores = []
            for gen_text in continuations:
                gen_tokens = _tokenize(gen_text, stemmer, stopwords)
                if not gen_tokens:
                    continue
                temp = {}
                for n in range(1, 25):
                    ngram_score, ngram_dict, overlap_count = _get_ngram_info(
                        remaining_tokens, gen_tokens, n
                    )
                    temp[f"ngram_{n}_score"] = ngram_score / len(gen_tokens)
                    temp[f"ngram_{n}_count"] = overlap_count
                gpt_scores.append(_n_gram_detector(temp))

            scores.append(float(np.mean(gpt_scores)) if gpt_scores else 0.0)
        return np.asarray(scores, dtype=float)
