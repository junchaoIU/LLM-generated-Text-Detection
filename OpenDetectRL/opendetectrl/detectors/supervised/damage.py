"""DAMAGE: Domain-Aware Multi-lingual AI-Generated Text Detection (Liu et al., 2024).

Official code: TBD

DAMAGE is a supervised detector designed for multi-domain and multi-lingual
scenarios. It uses a multi-lingual pre-trained encoder (XLM-RoBERTa) with
domain-adaptive fine-tuning.

This implementation wraps the DAMAGE model as an HF classifier.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from .hf_classifier import _HFClassifierBase


@register_detector("damage")
class DAMAGEDetector(_HFClassifierBase):
    """DAMAGE: multi-domain multi-lingual AI text detector.

    Uses XLM-RoBERTa as the backbone encoder with domain-adaptive training.

    Args:
        model_name: HuggingFace name of the DAMAGE checkpoint.
    """

    DEFAULT_MODEL_NAME = "facebook/xlm-roberta-base"
    AI_CLASS_INDEX = 1
