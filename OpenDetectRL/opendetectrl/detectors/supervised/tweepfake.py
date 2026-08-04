"""TweepFake: Detecting Deepfake Tweets.

Paper: Fagni et al., "TweepFake: About Detecting Deepfake Tweets", PLoS ONE
2020.
Official code: https://github.com/tizfa/tweepfake_deepfake_text_detection

The official TweepFake method extracts BERT (BERTweet) embeddings from each
tweet and feeds them to an LSTM classifier to distinguish human tweets from
deepfake (machine-generated) tweets.

This wrapper provides a HuggingFace sequence-classifier version that loads a
released fine-tuned checkpoint when available. The TweepFake authors did not
publish a canonical HF classifier checkpoint, so the default is set to
``tizfa/tweepfake_bert``; if that checkpoint does not exist on the HF hub, you
should supply your own fine-tuned checkpoint via ``model_name`` (e.g. a
BERTweet/BERT sequence classifier fine-tuned on the TweepFake dataset). The
default can be changed at will since the model is loaded lazily at ``__init__``.

Label mapping (following the official repo): class 1 = AI-generated / deepfake,
class 0 = human. Hence the AI class index is 1.

Score: softmax probability of class 1 (AI-generated). Higher = more likely AI.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("tweepfake")
class TweepFakeDetector(BaseDetector):
    """TweepFake deepfake-tweet detector.

    Loads a HuggingFace sequence classifier fine-tuned on the TweepFake
    dataset to distinguish human tweets from deepfake (machine-generated)
    tweets. The official method uses BERT (BERTweet) embeddings feeding an
    LSTM; this wrapper exposes the equivalent sequence-classification signal
    directly. At inference it performs a standard sequence-classification
    forward pass; the returned score is the softmax probability of the AI
    class (index 1).

    Note: the TweepFake authors did not release a canonical HuggingFace
    classifier checkpoint. The default ``model_name`` is ``tizfa/tweepfake_bert``
    (the author's org); if this is not published on the HF hub, supply your own
    fine-tuned checkpoint (e.g. a BERTweet/BERT classifier fine-tuned on
    TweepFake) via ``model_name``, or fall back to a plain ``roberta-base``.

    Args:
        model_name: HuggingFace checkpoint. Defaults to ``tizfa/tweepfake_bert``
            (fall back to a fine-tuned checkpoint via ``model_name`` if
            unavailable).
        max_length: Tokenizer truncation length.
        batch_size: Number of texts scored per forward pass.
        ai_class_index: Override the index of the "AI" output class.
        threshold: Decision threshold on the score (higher = more AI).
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    DEFAULT_MODEL_NAME = "tizfa/tweepfake_bert"
    AI_CLASS_INDEX = 1  # official repo: deepfake/AI -> class 1

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