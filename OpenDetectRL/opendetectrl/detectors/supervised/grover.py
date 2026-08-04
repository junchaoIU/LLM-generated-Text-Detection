"""GROVER: Defending Against Neural Fake News (Zellers et al., NeurIPS 2019).

Official code: https://github.com/rowanz/grover

GROVER is a model that can both generate and detect neural fake news.
The original implementation uses TensorFlow 1.13. This wrapper provides a
PyTorch-compatible interface by loading the GROVER discriminator as an
HF AutoModelForSequenceClassification checkpoint (if available from a
community port) or using a fine-tuned RoBERTa classifier as a substitute.

The official model checkpoint is ``grover-base`` / ``grover-mega``.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from .hf_classifier import _HFClassifierBase


@register_detector("grover")
class GROVERDetector(_HFClassifierBase):
    """GROVER discriminator (Zellers et al., NeurIPS 2019).

    The original GROVER uses TensorFlow 1.13. This implementation wraps the
    community PyTorch port (if available) or falls back to a compatible
    HF classifier. The score is the probability of the "machine" class.

    Args:
        model_name: HuggingFace model name of the GROVER discriminator port.
    """

    DEFAULT_MODEL_NAME = "rowanz/grover-base"
    AI_CLASS_INDEX = 0  # GROVER: class 0 = machine-generated
