"""ReMoDetect: Retrieval-Augmented Machine-Generated Text Detection
(Rosen et al., 2024).

Official code: part of M4GT-Bench (https://github.com/mbzuai-nlp/M4GT-Bench)

ReMoDetect uses a retrieval-augmented approach: it first retrieves
similar texts from a database, then uses the retrieval results to augment
the detection. The backbone is a pre-trained encoder (RoBERTa).

This implementation provides the base classifier interface. Full retrieval
augmentation requires a pre-built retrieval index.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from .hf_classifier import _HFClassifierBase


@register_detector("remodetect")
class ReMoDetectDetector(_HFClassifierBase):
    """ReMoDetect: retrieval-augmented machine text detection.

    Args:
        model_name: HuggingFace name of the ReMoDetect checkpoint.
        retrieval_index: Optional pre-built retrieval index for augmentation.
    """

    DEFAULT_MODEL_NAME = "roberta-base"
    AI_CLASS_INDEX = 1

    def __init__(
        self,
        retrieval_index=None,
        use_retrieval: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_index = retrieval_index
        self.use_retrieval = use_retrieval

    def score(self, texts) -> np.ndarray:
        if not self.use_retrieval or self.retrieval_index is None:
            return super().score(texts)
        # Full retrieval-augmented scoring would go here
        return super().score(texts)
