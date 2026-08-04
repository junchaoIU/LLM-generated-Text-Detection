"""RADAR: Robust AI-Text Detection via Adversarial Learning (Hu et al., NeurIPS 2023).

Official code: https://github.com/IBM/RADAR

Usage (from official README):
    detector = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp()

Score: softmax probability of class 0 (AI-generated). Higher = more likely AI.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


class _RADARBase(BaseDetector):
    """Shared implementation for RADAR detectors."""

    DEFAULT_MODEL_NAME = "TrustSafeAI/RADAR-Vicuna-7B"
    AI_CLASS_INDEX = 0

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
            probs = torch.nn.functional.log_softmax(outputs.logits, -1)
            scores.extend(probs[:, self.AI_CLASS_INDEX].exp().cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)


@register_detector("radar")
class RADARDetector(_RADARBase):
    """RADAR-Vicuna-7B: adversarially trained RoBERTa-large detector."""


@register_detector("radar_mix")
class RADARMixDetector(_RADARBase):
    """RADAR-Mix: trained on mixed LLM data for broader coverage."""

    DEFAULT_MODEL_NAME = "TrustSafeAI/RADAR-Mix"
