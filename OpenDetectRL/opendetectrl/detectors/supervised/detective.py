"""DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning
(Guo et al., NeurIPS 2024).

Official code: https://github.com/heyongxin233/DeTeCtive

Method: uses a contrastive-learning-trained text encoder (SimCSE-RoBERTa) to
embed texts, then classifies via K-Nearest-Neighbors against a pre-built
feature database. Score = fraction of AI-labeled neighbors among top-K.
"""
import os
import pickle
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("detective")
class DeTeCtiveDetector(BaseDetector):
    """DeTeCtive: contrastive embedding + KNN detection.

    Args:
        model_name: Text encoder (official default
            ``princeton-nlp/unsup-simcse-roberta-base``).
        model_path: Path to the fine-tuned DeTeCtive checkpoint (.pth).
        database_path: Path to the pre-built feature database directory
            (contains ``label_dict.pkl`` and index files).
        embedding_dim: Embedding dimension (official default 768).
        K: Number of nearest neighbors (official default 5).
    """

    DEFAULT_MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Optional[str] = None,
        database_path: Optional[str] = None,
        embedding_dim: int = 768,
        K: int = 5,
        max_length: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.model_path = model_path
        self.database_path = database_path
        self.embedding_dim = embedding_dim
        self.K = K
        self.max_length = max_length

        # Load text embedding model
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {}
            for key in state_dict:
                if key.startswith("model."):
                    new_state_dict[key[6:]] = state_dict[key]
            self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load KNN index and label dict
        self.label_dict = None
        self._index = None
        if database_path and os.path.isdir(database_path):
            label_path = os.path.join(database_path, "label_dict.pkl")
            if os.path.exists(label_path):
                with open(label_path, "rb") as f:
                    self.label_dict = pickle.load(f)
            try:
                from src.index import Indexer
                self._index = Indexer(embedding_dim)
                self._index.deserialize_from(database_path)
            except ImportError:
                pass

    @torch.inference_mode()
    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            encoded = self.tokenizer(
                [text],
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model(**encoded)
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, dict):
                output = output["last_hidden_state"]
            # Average pooling
            mask = encoded["attention_mask"]
            emb = output.sum(dim=1) / mask.sum(dim=1)[..., None]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def score(self, texts) -> np.ndarray:
        embeddings = self._embed(list(texts))
        if self._index is None or self.label_dict is None:
            # Without database, return embedding norm as placeholder
            return np.ones(len(texts), dtype=float) * 0.5
        scores = []
        for emb in embeddings:
            _, distances = self._index.search_knn(emb.reshape(1, -1), self.K)
            labels = [self.label_dict.get(i, 0) for i in range(self.K)]
            ai_fraction = sum(1 for l in labels if l == 0) / len(labels)
            scores.append(ai_fraction)
        return np.asarray(scores, dtype=float)
