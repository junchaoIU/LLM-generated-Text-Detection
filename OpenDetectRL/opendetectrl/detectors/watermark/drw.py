"""Distillation-Resistant Watermarking for Model Protection in NLP (Zhao et al.).

Paper: Zhao et al., "Distillation-Resistant Watermarking for Model Protection
in NLP", arXiv 2210.03312.
Repo: https://github.com/xuandongzhao/drw

DRW embeds a watermark into a model's behaviour via a secret trigger set.  A
watermarked model tends to produce triggers (or responses containing them)
when prompted with the secret trigger phrases.  Detection therefore checks
whether the text carries the watermark signal from the secret trigger set.
Higher signal => more likely watermarked (AI-generated).
"""
from typing import List, Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("drw")
class DRWDetector(BaseDetector):
    """DRW watermark detector (Zhao et al., arXiv 2210.03312).

    Checks whether the text contains any phrase from the secret trigger set
    used at deployment time.  The score is the fraction of trigger phrases
    found in the text (bounded to ``[0, 1]``), higher = more likely watermarked.

    Args:
        trigger_set: Secret trigger phrases used when watermarking the model.
        threshold: Decision threshold on the score.
    """

    def __init__(
        self,
        trigger_set: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.trigger_set = list(trigger_set) if trigger_set is not None else []
        if not self.trigger_set:
            raise RuntimeError(
                "DRWDetector requires the secret trigger set used when "
                "watermarking the model. Pass trigger_set=[...]."
            )

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores = []
        for text in texts:
            found = sum(1 for t in self.trigger_set if t in text)
            scores.append(found / len(self.trigger_set))
        return np.asarray(scores, dtype=float)