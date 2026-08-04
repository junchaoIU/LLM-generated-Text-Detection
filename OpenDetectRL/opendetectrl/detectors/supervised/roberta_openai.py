"""OpenAI RoBERTa supervised detectors.

Official usage (detect-gpt ``run.py::eval_supervised``):

    roberta = transformers.RobertaForSequenceClassification.from_pretrained(
        'roberta-base-openai-detector')
    ...
    outputs.logits.softmax(-1)[:, 0]   # probability of class 0 = machine-generated

The score is the softmax probability of class 0, which (per the official
evaluation code) corresponds to "AI-generated / fake". Higher = more likely
AI, matching the library convention.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


class _RoBERTaOpenAIBase(BaseDetector):
    """Shared implementation for the OpenAI RoBERTa detectors."""

    DEFAULT_MODEL_NAME = "roberta-base-openai-detector"
    AI_CLASS_INDEX = 0  # official eval_supervised: softmax(-1)[:, 0]

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        import transformers

        model_name = model_name or self.DEFAULT_MODEL_NAME
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # Official: padding + truncation to max_length=512
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokenized)
            probs = outputs.logits.softmax(-1)
            scores.extend(probs[:, self.AI_CLASS_INDEX].cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)


@register_detector("roberta_openai")
class RoBERTaOpenAIDetector(_RoBERTaOpenAIBase):
    """``roberta-base-openai-detector`` (OpenAI, 2019)."""


@register_detector("roberta_large_openai")
class RoBERTaLargeOpenAIDetector(_RoBERTaOpenAIBase):
    """``roberta-large-openai-detector`` (OpenAI, 2019)."""

    DEFAULT_MODEL_NAME = "roberta-large-openai-detector"
