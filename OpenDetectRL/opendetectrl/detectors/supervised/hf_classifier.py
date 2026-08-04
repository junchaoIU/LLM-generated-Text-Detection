"""Generic HuggingFace sequence-classification detector.

Supports any model loadable via ``AutoModelForSequenceClassification``:
DeBERTa, GROVER (if ported to HF), Sniffer, and arbitrary fine-tuned
classifiers. Score = softmax probability of the AI class.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


class _HFClassifierBase(BaseDetector):
    """Shared base for HF AutoModelForSequenceClassification detectors."""

    DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
    AI_CLASS_INDEX = 1  # convention: class 1 = AI-generated

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        ai_class_index: int = None,
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
        if ai_class_index is not None:
            self.AI_CLASS_INDEX = ai_class_index
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokenized)
            probs = torch.nn.functional.softmax(outputs.logits, -1)
            scores.extend(probs[:, self.AI_CLASS_INDEX].cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)


@register_detector("deberta_classifier")
class DeBERTaClassifierDetector(_HFClassifierBase):
    """Generic DeBERTa-v3 classifier for AI text detection."""

    DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


@register_detector("hf_classifier")
class HFClassifierDetector(_HFClassifierBase):
    """Generic HF classifier — load any AutoModelForSequenceClassification."""

    DEFAULT_MODEL_NAME = "roberta-base"
