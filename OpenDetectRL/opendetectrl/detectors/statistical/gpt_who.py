"""GPT-who: An Information Density-based Machine-Generated Text Detector
(Venkatraman et al., NAACL 2024 Findings).

Official code: https://github.com/saranya-venkatraman/gpt-who
(``get_uid_features.py`` + ``gpt-who.py``).

Method: based on the Uniform Information Density (UID) principle, which posits
that humans tend to spread information evenly during language production.
GPT-who extracts UID-based features from the per-token surprisal (information
content) distribution of a text under a language model.

Core features (official ``get_uid_features.py``):

* **UID variance** (``uid_var``): variance of per-token surprisals. AI-generated
  text tends to have more uniform information density (lower variance), while
  human text has higher variance.
* **UID diff** (``uid_diff``): mean absolute difference between consecutive
  surprisals.
* **UID diff2** (``uid_diff2``): mean squared difference between consecutive
  surprisals.
* **Mean surprisal** (``mean``): average information content.

The official pipeline feeds these features (plus UID min/max span features) to
a logistic-regression classifier. The zero-shot score used here is the negated
UID variance, so that lower variance (more uniform information density, more
likely AI) maps to a higher score, following the library-wide convention that
higher = more likely AI-generated.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("gpt_who")
class GPTWhoDetector(BaseDetector):
    """GPT-who: UID variance-based detection.

    Args:
        model: Pre-built :class:`HFSourceModel` (scoring model).
        model_name: HuggingFace name (official default ``gpt2-xl``).
        threshold: Decision threshold on the negated UID variance
            (higher = more likely AI).
        device: Device for the model.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2-xl",
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model

    @torch.inference_mode()
    def _get_surprisals(self, text: str) -> np.ndarray:
        """Official ``get_line_uid_surp``: per-token surprisals.

        Prepends the EOS token (official convention) and computes
        ``-log p(x_t | x_{<t})`` for each token ``t >= 1``.
        """
        eos_token = self.model.tokenizer.eos_token
        logits, input_ids = self.model.get_logits(eos_token + text)
        # logits[t] predicts input_ids[t+1]; surprisal = -log p(next token)
        log_probs = F.log_softmax(logits[0], dim=-1)
        target_ids = input_ids[0, 1:]
        surprisals = -log_probs[:-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return surprisals.cpu().numpy()

    def _uid_features(self, surprisals: np.ndarray) -> dict:
        """Compute the 4 official UID features from per-token surprisals.

        ``uid_diff`` and ``uid_diff2`` are divided by ``len(surprisals)``
        (not ``len-1``) to match the official ``local_diff`` / ``local_diff2``.
        """
        n = len(surprisals)
        if n == 0:
            return {"uid_var": 0.0, "uid_diff": 0.0, "uid_diff2": 0.0, "mean": 0.0}
        mu = float(np.mean(surprisals))
        uid_var = float(np.var(surprisals))
        if n > 1:
            diffs = np.diff(surprisals)
            uid_diff = float(np.sum(np.abs(diffs)) / n)
            uid_diff2 = float(np.sum(diffs ** 2) / n)
        else:
            uid_diff = 0.0
            uid_diff2 = 0.0
        return {"uid_var": uid_var, "uid_diff": uid_diff, "uid_diff2": uid_diff2, "mean": mu}

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            surprisals = self._get_surprisals(text)
            features = self._uid_features(surprisals)
            # Negate variance: lower variance (more uniform, more AI) -> higher
            scores.append(-features["uid_var"])
        return np.asarray(scores, dtype=float)
