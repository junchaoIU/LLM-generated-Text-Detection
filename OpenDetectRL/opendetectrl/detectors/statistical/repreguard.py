"""RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation
Patterns (Chen et al., EMNLP 2025).

Official code: https://github.com/NLP2CT/RepreGuard (``repreGuard_detector.py``)

Method: use representation reading (RepE) to extract hidden representation
patterns from the source LLM. The method:
1. Train a representation reader (PCA-based direction finder) on AI vs human
   text pairs.
2. For each test text, compute the representation reading score across all
   layers and tokens.
3. The mean score serves as the detection signal.

The official implementation uses the `repe` library for representation
engineering. This implementation provides the core logic.
"""
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("repreguard")
class RepreGuardDetector(BaseDetector):
    """RepreGuard: representation-based detection via hidden state patterns.

    Args:
        model: Pre-built :class:`HFSourceModel` (the source LLM).
        model_name: HuggingFace name (official: LLaMA-2/3).
        rep_token: Token position for representation reading
            (official default -1 = last token).
        n_difference: Number of difference directions (official default 1).
        direction_method: Method for finding directions (official ``pca``).
        n_train: Number of training samples for direction finding
            (official default 128).
        train_data: List of dicts with ``direct_prompt`` (AI) and
            ``human_text`` (human) keys for training the rep reader.
    """

    requires_source_model = True

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        rep_token: int = -1,
        n_difference: int = 1,
        direction_method: str = "pca",
        n_train: int = 128,
        train_data: Optional[list] = None,
        ai_weight: float = 1.0,
        human_weight: float = 1.0,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if model is None:
            model = HFSourceModel(model_name, device=device)
        self.model = model
        self.rep_token = rep_token
        self.n_difference = n_difference
        self.direction_method = direction_method
        self.n_train = n_train
        self.ai_weight = ai_weight
        self.human_weight = human_weight
        self.hidden_layers = list(
            range(-1, -self.model.model.config.num_hidden_layers, -1)
        ) if hasattr(self.model.model.config, "num_hidden_layers") else [-1]

        # Train rep reader if data provided
        self.rep_reader = None
        self.direction_signs = None
        self.directions = None
        if train_data:
            self._train_rep_reader(train_data)

    def _train_rep_reader(self, train_data: list):
        """Train the representation reader using PCA on hidden states.

        Official: uses `repe` pipeline with PCA direction finding on
        AI vs human text pairs.
        """
        # Extract hidden states for AI and human texts
        ai_texts = [item["direct_prompt"] for item in train_data[: self.n_train]]
        human_texts = [item["human_text"] for item in train_data[: self.n_train]]

        ai_states = self._extract_hidden_states_batch(ai_texts)
        human_states = self._extract_hidden_states_batch(human_texts)

        # PCA: find direction that maximizes variance between AI and human
        self.directions = {}
        self.direction_signs = {}
        for layer in self.hidden_layers:
            ai_layer = np.stack([s[layer].mean(0) for s in ai_states])
            human_layer = np.stack([s[layer].mean(0) for s in human_states])
            diff = ai_layer - human_layer
            # PCA via SVD
            U, S, Vt = np.linalg.svd(diff, full_matrices=False)
            self.directions[layer] = Vt[0]
            self.direction_signs[layer] = [1.0]

    @torch.inference_mode()
    def _extract_hidden_states_batch(self, texts: List[str]):
        """Extract hidden states for a batch of texts."""
        all_states = []
        for text in texts:
            tokenized = self.model.tokenize(text)
            outputs = self.model.model(**tokenized, output_hidden_states=True)
            # hidden_states: tuple of (batch, seq, hidden_dim) per layer
            states_per_layer = []
            for layer_idx in range(len(outputs.hidden_states)):
                states_per_layer.append(
                    outputs.hidden_states[layer_idx][0].cpu().numpy()
                )
            all_states.append(states_per_layer)
        return all_states

    @torch.inference_mode()
    def _get_rep_score(self, text: str) -> float:
        """Compute the representation reading score for a text."""
        tokenized = self.model.tokenize(text)
        outputs = self.model.model(**tokenized, output_hidden_states=True)

        token_scores = []
        num_tokens = tokenized.input_ids.shape[1]
        for token_idx in range(1, num_tokens):
            layer_scores = []
            for layer in self.hidden_layers:
                hidden = outputs.hidden_states[layer][0, token_idx].cpu().numpy()
                if self.directions and layer in self.directions:
                    score = np.dot(hidden, self.directions[layer])
                    sign = self.direction_signs[layer][0]
                    layer_scores.append(score * sign)
                else:
                    layer_scores.append(0.0)
            token_scores.append(np.mean(layer_scores))
        return float(np.mean(token_scores)) if token_scores else 0.0

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            scores.append(self._get_rep_score(text))
        return np.asarray(scores, dtype=float)
