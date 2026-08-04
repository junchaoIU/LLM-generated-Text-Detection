"""ConDA: Contrastive Domain Adaptation for AI-generated Text Detection.

Paper: Bhattacharjee et al., "ConDA: Contrastive Domain Adaptation for
AI-generated Text Detection", IJCNLP-AACL 2023.
Official code: https://github.com/AmritaBh/ConDA-gen-text-detection
Model: https://huggingface.co/AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection

ConDA uses a RoBERTa-large backbone with contrastive learning and domain
adaptation to make the detector robust across domains and prompts. The
released HuggingFace checkpoint
``AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection`` is a RoBERTa-large
sequence classifier.

Label mapping (following the official repo): class 1 = AI-generated,
class 0 = human-written. Hence the AI class index is 1.

Score: softmax probability of class 1 (AI-generated). Higher = more likely AI.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("conda")
class ConDADetector(BaseDetector):
    """ConDA (Contrastive Domain Adaptation) detector.

    Loads the ``AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection`` RoBERTa-large
    checkpoint, which is trained with contrastive learning and domain
    adaptation to detect AI-generated text across domains. At inference this
    wrapper exposes a standard sequence-classification forward pass; the
    returned score is the softmax probability of the AI class (index 1).

    Args:
        model_name: HuggingFace checkpoint. Defaults to the official
            ``AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection``.
        max_length: Tokenizer truncation length (RoBERTa context 512).
        batch_size: Number of texts scored per forward pass.
        ai_class_index: Override the index of the "AI" output class.
        threshold: Decision threshold on the score (higher = more AI).
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    DEFAULT_MODEL_NAME = "AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection"
    AI_CLASS_INDEX = 1  # official repo: AI-generated -> class 1

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