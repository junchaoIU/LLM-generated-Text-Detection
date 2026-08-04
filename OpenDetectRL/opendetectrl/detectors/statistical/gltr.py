"""GLTR: Giant Language model Test Room (Strobelt, Gehrmann & Rush, 2019).

Official code: https://github.com/HendrikStrobelt/detecting-fake-text
(``backend/api.py``).

GLTR (Generating Language TRuthfulness) examines the rank of each token
under a language model. For each token, it checks whether the token falls
within the top-k most likely predictions. Human text tends to have more
"surprising" tokens (higher fraction outside top-k), while AI text has
fewer surprises. Score direction: higher = more likely AI-generated, so
:meth:`score` returns the fraction of tokens that ARE in the top-k.
"""
from typing import Optional, Tuple

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("gltr")
class GLTRDetector(BaseDetector):
    """GLTR: token-rank based detection.

    For each token, the language model's predicted distribution is sorted
    by probability and the rank of the actual token is recorded. Tokens
    with rank < ``topk`` are considered "expected"; the rest are
    "surprising". Human text tends to have more surprising tokens, while
    AI text has fewer. The detection score is the fraction of expected
    (top-k) tokens -- higher = more likely AI-generated.

    Args:
        model: Pre-built :class:`HFSourceModel`.
        model_name: HuggingFace name of the base model (official default
            ``gpt2``).
        topk: Number of top predictions considered "expected" (official
            default 40).
        threshold: Decision threshold; score > threshold => AI (default 0.5).
        device: Device for the model.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        topk: int = 40,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.tokenizer = model.tokenizer
        self.topk = topk

    def _start_token_id(self) -> Optional[int]:
        """BOS token id for prepending context (official ``start_token``)."""
        bos = getattr(self.tokenizer, "bos_token", None)
        if bos is None:
            bos = getattr(self.tokenizer, "eos_token", None)
        if bos is None:
            return None
        return self.tokenizer(bos, add_special_tokens=False).input_ids[-1]

    @torch.inference_mode()
    def _forward(
        self, text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize (with BOS prepended), forward, return ``(token_ids, probs, targets)``.

        * ``token_ids``: 1-D tensor of the full sequence (BOS + text).
        * ``probs``: ``[n_positions, vocab_size]`` softmax probabilities.
        * ``targets``: 1-D tensor of actual next-token ids.
        """
        token_ids = self.tokenizer(
            text, return_tensors="pt", return_token_type_ids=False
        ).input_ids[0]
        start_id = self._start_token_id()
        if start_id is not None:
            token_ids = torch.cat(
                [torch.tensor([start_id], dtype=token_ids.dtype), token_ids]
            )
        token_ids = token_ids.to(self.model.device)

        output = self.model.model(token_ids.unsqueeze(0))
        all_logits = output.logits[0, :-1]  # [n_pos, vocab]
        all_probs = torch.softmax(all_logits, dim=-1)
        y = token_ids[1:]  # actual next tokens
        return token_ids, all_probs, y

    @torch.inference_mode()
    def check_probabilities(self, text: str) -> dict:
        """Official ``check_probabilities`` payload for a single text.

        Returns a dict with:
            * ``real_topk``: list of ``(rank, prob)`` per token.
            * ``bpe_strings``: list of token strings (including BOS).
            * ``pred_topk``: per-position list of ``(token, prob)``.
        """
        token_ids, all_probs, y = self._forward(text)
        n = y.shape[0]

        # Sort predictions descending -> token ids in rank order
        sorted_preds = torch.argsort(all_probs, dim=-1, descending=True)
        # Rank of each actual token (official np.where approach, vectorized)
        real_topk_pos = [
            int((sorted_preds[i] == y[i].item()).nonzero(as_tuple=True)[0].item())
            for i in range(n)
        ]
        real_topk_probs = all_probs[torch.arange(n), y].cpu().numpy().tolist()
        real_topk = list(zip(real_topk_pos, real_topk_probs))

        bpe_strings = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        topk_val = min(self.topk, all_probs.shape[-1])
        topk_prob_values, topk_prob_inds = torch.topk(
            all_probs, k=topk_val, dim=-1
        )
        pred_topk = [
            list(
                zip(
                    self.tokenizer.convert_ids_to_tokens(
                        topk_prob_inds[i].tolist()
                    ),
                    topk_prob_values[i].cpu().numpy().tolist(),
                )
            )
            for i in range(n)
        ]

        return {
            "bpe_strings": bpe_strings,
            "real_topk": real_topk,
            "pred_topk": pred_topk,
        }

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        """Fraction of tokens in the top-k (higher = more likely AI)."""
        scores = []
        for text in texts:
            _, all_probs, y = self._forward(text)
            n = y.shape[0]
            if n == 0:
                scores.append(0.0)
                continue

            # Efficient rank: number of tokens with strictly higher prob
            actual_probs = all_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            token_ranks = (all_probs > actual_probs.unsqueeze(-1)).sum(dim=-1)
            fraction_in_topk = (token_ranks < self.topk).float().mean().item()
            scores.append(fraction_in_topk)

        return np.asarray(scores, dtype=float)
