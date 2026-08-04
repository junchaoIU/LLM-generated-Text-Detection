"""BiScope: AI-Generated Text Detection by Checking Memorization
(Ye et al., NeurIPS 2024).

Official code: https://github.com/MarkGHX/BiScope (``biscope.py`` + ``biscope_utils.py``)

Method: compute Forward Cross-Entropy (FCE) and Backward Cross-Entropy (BCE)
losses of the text under a detection model, using a summary-based prompt.
Extract 7 features (mean, max, min, std of FCE + mean, max, min, std of BCE)
across 9 segments, then classify with RandomForest.

Zero-shot variant: use the mean FCE loss as the detection score (lower loss =
more memorized = more likely AI; we negate so higher = more AI).
"""
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel

COMPLETION_PROMPT_ONLY = "Complete the following text: "
COMPLETION_PROMPT = "Given the summary:\n{prompt}\n Complete the following text: "


@register_detector("biscope")
class BiScopeDetector(BaseDetector):
    """BiScope: loss-based memorization checking.

    Args:
        model: Pre-built :class:`HFSourceModel` (detection model).
        model_name: HuggingFace name (official: LLaMA-2/3, Gemma, Mistral).
        sample_clip: Max token length (official default 2000).
        use_summary: Whether to use summary-based prompt (requires a summary
            model; set to False for zero-shot without summary).
        summary_model: Optional summary model for prompt generation.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        sample_clip: int = 2000,
        use_summary: bool = False,
        summary_model=None,
        summary_tokenizer=None,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.sample_clip = sample_clip
        self.use_summary = use_summary
        self.summary_model = summary_model
        self.summary_tokenizer = summary_tokenizer

    def _compute_fce_loss(self, logits, targets, text_slice):
        """Official ``compute_fce_loss``: CE with index shift by 1."""
        loss = CrossEntropyLoss(reduction="none")(
            logits[0, text_slice.start - 1 : text_slice.stop - 1, :],
            targets,
        )
        return loss.detach().cpu().numpy()

    def _compute_bce_loss(self, logits, targets, text_slice):
        """Official ``compute_bce_loss``: CE without index shift."""
        loss = CrossEntropyLoss(reduction="none")(
            logits[0, text_slice, :],
            targets,
        )
        return loss.detach().cpu().numpy()

    @torch.inference_mode()
    def _detect_single_sample(self, sample: str) -> dict:
        """Official ``detect_single_sample``: compute FCE/BCE loss features."""
        prompt_text = COMPLETION_PROMPT_ONLY

        # Tokenize
        prompt_ids = self.model.tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            self.model.device
        )
        text_ids = self.model.tokenizer(
            sample, return_tensors="pt", max_length=self.sample_clip, truncation=True
        ).input_ids.to(self.model.device)
        combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
        text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])

        outputs = self.model.model(input_ids=combined_ids)
        logits = outputs.logits
        targets = combined_ids[0][text_slice]

        fce_loss = self._compute_fce_loss(logits, targets, text_slice)
        bce_loss = self._compute_bce_loss(logits, targets, text_slice)

        # Official: 7 features per segment, 9 segments (p=1..9)
        features = []
        for p in range(1, 10):
            split = len(fce_loss) * p // 10
            features.extend([
                np.mean(fce_loss[split:]), np.max(fce_loss[split:]),
                np.min(fce_loss[split:]), np.std(fce_loss[split:]),
                np.mean(bce_loss[split:]), np.max(bce_loss[split:]),
                np.min(bce_loss[split:]), np.std(bce_loss[split:]),
            ])
        return {"features": features, "fce_mean": float(np.mean(fce_loss))}

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            result = self._detect_single_sample(text)
            # Zero-shot: use negative mean FCE (lower loss = more memorized = AI)
            scores.append(-result["fce_mean"])
        return np.asarray(scores, dtype=float)
