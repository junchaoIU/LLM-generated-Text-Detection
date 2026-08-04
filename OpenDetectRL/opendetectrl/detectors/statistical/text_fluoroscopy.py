"""Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features
(Yu et al., EMNLP 2024).

Official code: https://github.com/Fish-and-Sheep/Text-Fluoroscopy
(``gte-qwen/save_KL_with_first_and_last_layer.py`` +
``embedding_classify/classify_with_max_KL_layer.py``).

Method: Text Fluoroscopy identifies the model layer with the largest
distribution difference (KL divergence) from the last and first layers
when projected to the vocabulary space. This "max-KL layer" captures
intrinsic text features that reveal the actual fingerprints of AI
generation.

The official implementation trains a binary MLP classifier on the
last-token-pooled embedding from the max-KL layer. This zero-shot
adaptation uses the mean per-token entropy of the vocabulary distribution
at the max-KL layer as the detection signal: AI-generated text tends to
have more predictable (lower-entropy) distributions at the intrinsic
feature layer. Score direction: higher = more likely AI-generated.
"""
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...core.base import BaseDetector, DetectionResult
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("text_fluoroscopy")
class TextFluoroscopyDetector(BaseDetector):
    """Text Fluoroscopy: intrinsic feature detection via layer-wise KL.

    The detector follows the official pipeline:

    1. Forward the text through a causal LM with ``output_hidden_states``.
    2. Project each layer's hidden state to vocabulary space via ``lm_head``.
    3. For each middle layer, compute KL(first || middle) + KL(last || middle).
    4. Select the layer with the maximum KL divergence ("max-KL layer").
    5. Compute the mean per-token entropy of the vocabulary distribution at
       that layer.

    AI-generated text has more predictable (lower-entropy) patterns at the
    intrinsic feature layer, so the score is the negated mean entropy
    (higher = more likely AI).

    Args:
        model: Pre-built :class:`HFSourceModel`.
        model_name: HuggingFace name (official: ``gte-Qwen1.5-7B-instruct``;
            default ``gpt2`` for local testing).
        max_length: Maximum token length (official default 512).
        layer_selection: ``"max_kl"`` (default) to dynamically select the
            layer with maximum KL divergence, or an int to use a fixed
            hidden-state layer index (official optimization: 30).
        threshold: Decision threshold; score > threshold => AI (default 0.0).
        device: Device for the model.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        max_length: int = 512,
        layer_selection: Union[str, int] = "max_kl",
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_length = max_length
        self.layer_selection = layer_selection

    @torch.inference_mode()
    def _compute_layer_features(self, text: str):
        """Compute per-layer KL divergences and the selected layer's entropy.

        Returns ``(layer_idx, kls, mean_entropy)`` where ``layer_idx`` is the
        selected hidden-state layer index, ``kls`` is a list of KL values for
        each middle layer, and ``mean_entropy`` is the mean per-token entropy
        at the selected layer.
        """
        batch_dict = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.model(**batch_dict, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, seq, dim]
        lm_head = self.model.model.lm_head

        # Project first and last layers to vocabulary space (official style)
        first_logits = lm_head(hidden_states[0])   # [1, seq, vocab]
        last_logits = lm_head(hidden_states[-1])   # [1, seq, vocab]
        first_probs = F.softmax(first_logits, dim=-1)
        last_probs = F.softmax(last_logits, dim=-1)

        kls = []
        middle_log_probs_list = []
        for i in range(1, len(hidden_states) - 1):
            middle_logits = lm_head(hidden_states[i])
            middle_log_probs = F.log_softmax(middle_logits, dim=-1)
            kl_first = F.kl_div(
                middle_log_probs, first_probs, reduction="batchmean"
            ).item()
            kl_last = F.kl_div(
                middle_log_probs, last_probs, reduction="batchmean"
            ).item()
            kls.append(kl_first + kl_last)
            middle_log_probs_list.append(middle_log_probs)

        # Select the layer (official: max KL; optimized: fixed layer index)
        if self.layer_selection == "max_kl":
            list_idx = int(np.argmax(kls))
        else:
            # layer_selection is a hidden-state index (0=embedding, 1=first layer, ...)
            list_idx = int(self.layer_selection) - 1
            if list_idx < 0 or list_idx >= len(middle_log_probs_list):
                raise ValueError(
                    f"layer_selection={self.layer_selection} is out of range "
                    f"for {len(middle_log_probs_list)} middle layers "
                    f"(valid hidden-state indices: 1..{len(hidden_states) - 2})"
                )

        layer_idx = list_idx + 1  # convert back to hidden_states index

        # Mean per-token entropy at the selected layer
        selected_log_probs = middle_log_probs_list[list_idx]  # [1, seq, vocab]
        selected_probs = selected_log_probs.exp()
        entropy = -(selected_probs * selected_log_probs).sum(dim=-1)  # [1, seq]

        # Mask out padding tokens
        attention_mask = batch_dict["attention_mask"].float()
        entropy = entropy.squeeze(0) * attention_mask.squeeze(0)
        mean_entropy = entropy.sum() / attention_mask.sum()

        return layer_idx, kls, mean_entropy.item()

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        """Negated mean entropy at the max-KL layer (higher = more likely AI)."""
        scores = []
        for text in texts:
            _, _, mean_entropy = self._compute_layer_features(text)
            scores.append(-mean_entropy)
        return np.asarray(scores, dtype=float)

    def predict(self, texts) -> List[DetectionResult]:
        """Predictions with the max-KL layer index and KL values in metadata."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        results = []
        for text in texts:
            layer_idx, kls, mean_entropy = self._compute_layer_features(text)
            score = -mean_entropy
            label = "ai" if score > self.threshold else "human"
            results.append(
                DetectionResult(
                    score=float(score),
                    label=label,
                    confidence=1.0,
                    metadata={
                        "max_kl_layer": layer_idx,
                        "kl_values": kls,
                        "mean_entropy": mean_entropy,
                    },
                )
            )
        return results
