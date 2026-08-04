"""A Private Watermark for Large Language Models (Wu et al., ICLR 2024).

Paper: Wu et al., "A Private Watermark for Large Language Models", ICLR 2024.
Repo: https://github.com/THU-BPM/private_watermark

The scheme embeds a watermark using a private, secret DNN.  At generation time
a secret is injected into the model's output distribution; at detection time a
watermark detector (trained with the secret key/embeddings) verifies whether
the text carries the watermark signal.

This wrapper implements the *detection* half of the scheme: it requires a
pre-trained watermark detector (a ``transformers.AutoModelForSequenceClassification``
or a custom checkpoint path) and computes a watermark score in ``[0, 1]``,
where higher = more likely watermarked (AI-generated).
"""
from typing import Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("private_watermark")
class PrivateWatermarkDetector(BaseDetector):
    """Private Watermark detector (Wu et al., ICLR 2024).

    Verification step of the private watermarking scheme.  A pre-trained
    watermark detector checkpoint (trained with the secret key) must be
    supplied via ``checkpoint_path``; the score is the softmax probability of
    the watermarked class.

    Args:
        checkpoint_path: Path/name of the watermark detector checkpoint.
        max_length: Tokenizer truncation length.
        ai_class_index: Index of the watermarked ("AI") output class.
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_length: int = 512,
        ai_class_index: int = 1,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.max_length = max_length
        self.ai_class_index = ai_class_index
        if checkpoint_path is None:
            raise RuntimeError(
                "PrivateWatermarkDetector requires a pre-trained watermark "
                "detector checkpoint. Pass checkpoint_path=<path> (trained with "
                "the secret key from the official private_watermark repo)."
            )

    def _get_model(self):
        if not hasattr(self, "_model"):
            import transformers

            self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint_path, trust_remote_code=True
            )
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.checkpoint_path, trust_remote_code=True
            )
            self._model.eval()
        return self._model, self._tokenizer

    def score(self, texts) -> np.ndarray:
        model, tokenizer = self._get_model()
        import torch

        texts = list(texts)
        scores = []
        for text in texts:
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            with torch.inference_mode():
                logits = model(**tokenized).logits
            probs = torch.nn.functional.softmax(logits, -1)
            scores.append(probs[0, self.ai_class_index].item())
        return np.asarray(scores, dtype=float)