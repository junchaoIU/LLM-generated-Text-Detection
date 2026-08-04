"""CATER: Conditional Watermarking for Text Generation APIs (He et al., NeurIPS 2022).

Paper: He et al., "CATER: Intellectual Property Protection on Text Generation
APIs via Conditional Watermarks", NeurIPS 2022.
Repo: https://github.com/xlhex/cater_neurips

CATER is a *conditional* watermarking scheme: a text is watermarked only if it
contains a secret trigger phrase chosen at generation time.  Detection is
therefore a deterministic, phrase-level check: the text is deemed watermarked
(AI-generated) when the required trigger phrase appears in it.
"""
from typing import List, Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("cater")
class CATERDetector(BaseDetector):
    """CATER conditional watermark detector (He et al., NeurIPS 2022).

    Checks whether the secret trigger phrase selected at generation time
    appears in the text.  Returns 1.0 if present (watermarked), else 0.0.

    Args:
        trigger_phrase: The secret trigger phrase used at generation time.
        threshold: Decision threshold on the score.
    """

    def __init__(
        self,
        trigger_phrase: Optional[str] = None,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.trigger_phrase = trigger_phrase
        if trigger_phrase is None:
            raise RuntimeError(
                "CATERDetector requires the secret trigger phrase used at "
                "generation time. Pass trigger_phrase=<phrase>."
            )

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores = []
        for text in texts:
            scores.append(1.0 if self.trigger_phrase in text else 0.0)
        return np.asarray(scores, dtype=float)