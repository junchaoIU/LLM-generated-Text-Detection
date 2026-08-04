"""MPU: Multiscale Positive-Unlabeled Detection of AI-Generated Texts
(Tian et al., ICLR 2024 Spotlight).

Official code: https://github.com/yuchuantian/aigc_text_detector
Model: https://huggingface.co/yuchuantian/AIGC_detector_env1

MPU trains a transformer classifier (RoBERTa-base) on positive (AI-generated)
and unlabeled (mixed) data at multiple text granularities (sentence,
paragraph, document). The multiscale positive-unlabeled learning is a
training-time strategy; at inference the detector behaves as a standard
sequence classifier.

Label mapping (from official ``dataset.py``): ``real_texts = chatgpt/AI ->
label 1``, ``fake_texts = human -> label 0`` (the comment
``# in this context, real->label1->chatgpt`` confirms this). Hence class 1
is the AI-generated class, which matches the ``accuracy_sum`` rule
``classification = (logits[..., 0] < logits[..., 1])``.

Usage (from official HuggingFace model card):

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("yuchuantian/AIGC_detector_env1")
    model = AutoModelForSequenceClassification.from_pretrained("yuchuantian/AIGC_detector_env1")
    probs = model(**inputs).logits.softmax(-1)   # column 1 = AI

Score: softmax probability of class 1 (AI-generated). Higher = more likely AI.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("mpu")
class MPUDetector(BaseDetector):
    """MPU (Multiscale Positive-Unlabeled) AI-text detector.

    Loads a HuggingFace checkpoint fine-tuned with the MPU multiscale
    positive-unlabeled learning method (Tian et al., ICLR 2024 Spotlight).
    The default checkpoint is the English v1 model trained on HC3-English
    from a pretrained RoBERTa-base. Other released checkpoints can be
    selected via ``model_name``, e.g. ``yuchuantian/AIGC_detector_env3``
    (En v3, strongest), ``yuchuantian/AIGC_detector_zhv3`` (Chinese v3),
    or the ``-short`` variants.

    Inference is a standard sequence-classification forward pass; the
    multiscale PU strategy is applied only at training time. The returned
    score is the softmax probability of the AI class (index 1).
    """

    DEFAULT_MODEL_NAME = "yuchuantian/AIGC_detector_env1"
    AI_CLASS_INDEX = 1  # official dataset.py: chatgpt/AI -> label 1

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
