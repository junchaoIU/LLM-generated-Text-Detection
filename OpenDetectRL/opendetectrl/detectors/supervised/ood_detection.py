"""OOD Detection: Human Texts Are Outliers (Zeng et al., NeurIPS 2025).

Official code: https://github.com/cong-zeng/ood-llm-detect
Paper: "Human Texts Are Outliers: Detecting LLM-generated Texts via
Out-of-distribution Detection" (https://arxiv.org/abs/2510.08602)

Method: reframes AI-text detection as an out-of-distribution (OOD)
detection problem. Texts are embedded with a SimCSE-RoBERTa encoder
(the same ``princeton-nlp/unsup-simcse-roberta-base`` backbone used by the
official repo and by DeTeCtive, on which the official code is built). A
cluster centroid (and optionally a covariance for Mahalanobis / density
estimation) is fit on a reference set of HUMAN texts, which define the
in-distribution. AI-generated texts fall outside this cluster and are
scored by their distance from it -- the further from the human centroid,
the more OOD (and hence the more likely AI-generated).

Scoring follows the official DeepSVDD inference scheme:
``dist = ||emb - c||`` with a soft-boundary radius ``R`` set to the
``(1 - nu)``-quantile of the in-distribution distances. The raw distance is
calibrated against the in-distribution spread so that the default
``threshold = 0.5`` separates the two populations.

Score: calibrated OOD score in (0, 1). Higher = more OOD = more likely AI.
"""
import os
from typing import List, Optional, Sequence

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector


@register_detector("ood_detection")
class OODDetectionDetector(BaseDetector):
    """OOD-detection-based AI-text detector.

    Treats human-written text as the in-distribution cluster and scores each
    input by how outlier-like its embedding is relative to that cluster
    (distance from the centroid, or Mahalanobis distance as a density
    estimate). A reference set of human texts is used to fit the centroid
    and the calibration statistics.

    Args:
        reference_texts: A sequence of human-written texts that define the
            in-distribution. If provided, the centroid and calibration
            stats are fit immediately at construction (see :meth:`fit`).
        model_name: HuggingFace encoder (official default
            ``princeton-nlp/unsup-simcse-roberta-base``).
        model_path: Optional path to a fine-tuned encoder checkpoint
            (``.pth``). The official OOD checkpoints wrap the RoBERTa
            encoder under nested ``model.`` prefixes; those are stripped
            best-effort and the matching keys are loaded into the HF
            ``AutoModel``. When not provided, the plain pretrained encoder
            is used.
        method: Distance metric to the centroid. ``"euclidean"`` (default,
            matches the official DeepSVDD inference) or ``"mahalanobis"``
            (density estimation via the in-distribution covariance).
        nu: Outlier fraction used to set the soft-boundary radius ``R`` as
            the ``(1 - nu)``-quantile of in-distribution distances
            (official default ``0.1``).
        temperature: Scaling of the calibrated score; larger values produce
            softer probabilities.
        cov_reg: Diagonal shrinkage added to the covariance before inversion
            (only used when ``method == "mahalanobis"``).
        max_length: Max token length per text.
        batch_size: Batch size for embedding.
        threshold: Decision threshold on the score (higher = more AI).
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto).
    """

    DEFAULT_MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"

    def __init__(
        self,
        reference_texts: Optional[Sequence[str]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Optional[str] = None,
        method: str = "euclidean",
        nu: float = 0.1,
        temperature: float = 1.0,
        cov_reg: float = 1e-3,
        max_length: int = 512,
        batch_size: int = 32,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        import transformers

        if method not in ("euclidean", "mahalanobis"):
            raise ValueError(
                f"method must be 'euclidean' or 'mahalanobis', got {method!r}"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.model_path = model_path
        self.method = method
        self.nu = float(nu)
        self.temperature = float(temperature)
        self.cov_reg = float(cov_reg)
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._try_load_checkpoint(self.model, self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # In-distribution (human) parameters; populated by :meth:`fit`.
        self._centroid: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._radius: Optional[float] = None
        self._ref_std: Optional[float] = None

        if reference_texts is not None:
            self.fit(reference_texts)

    # -- encoder & embeddings -------------------------------------------------

    @staticmethod
    def _try_load_checkpoint(model, model_path: Optional[str]) -> bool:
        """Best-effort load of fine-tuned encoder weights from a checkpoint.

        Official OOD checkpoints wrap the RoBERTa encoder under nested
        ``model.`` prefixes (``TextEmbeddingModel`` -> ``AutoModel``). We
        strip those prefixes and load only the keys matching the plain HF
        ``AutoModel``. Returns whether any keys were loaded.
        """
        if not model_path or not os.path.exists(model_path):
            return False
        state_dict = torch.load(model_path, map_location="cpu")
        target = set(model.state_dict().keys())
        sd = dict(state_dict)
        for _ in range(4):
            matched = {k: v for k, v in sd.items() if k in target}
            if matched:
                model.load_state_dict(matched, strict=False)
                return True
            sd = {
                k[len("model."):]: v
                for k, v in sd.items()
                if k.startswith("model.")
            }
            if not sd:
                break
        return False

    @torch.inference_mode()
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized mean-pooled embeddings for ``texts``."""
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            hidden = output.last_hidden_state  # (B, L, H)
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embeddings.append(pooled.cpu().numpy())
        if not embeddings:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)

    # -- in-distribution fitting ---------------------------------------------

    def fit(self, reference_texts: Sequence[str]) -> "OODDetectionDetector":
        """Fit the in-distribution (human) centroid and calibration stats."""
        texts = list(reference_texts)
        if not texts:
            return self
        emb = self._embed(texts)
        self._centroid = emb.mean(axis=0)
        if self.method == "mahalanobis":
            cov = np.cov(emb, rowvar=False)
            if cov.ndim == 0:  # single reference text
                cov = np.array([[float(cov)]])
            cov = cov + self.cov_reg * np.eye(emb.shape[1], dtype=cov.dtype)
            self._cov_inv = np.linalg.inv(cov)
        # Calibration: distance distribution of the in-distribution texts.
        d_ref = self._distance(emb)
        self._ref_std = float(np.std(d_ref)) + 1e-8
        self._radius = float(np.quantile(d_ref, 1.0 - self.nu))
        return self

    def _distance(self, emb: np.ndarray) -> np.ndarray:
        """Distance of each embedding to the fitted centroid."""
        if self._centroid is None:
            raise RuntimeError("Detector is not fitted; call fit() first.")
        diff = emb - self._centroid
        if self.method == "mahalanobis" and self._cov_inv is not None:
            md = np.einsum("ij,jk,ik->i", diff, self._cov_inv, diff)
            return np.sqrt(np.maximum(md, 0.0))
        return np.sqrt(np.maximum((diff ** 2).sum(axis=1), 0.0))

    # -- scoring --------------------------------------------------------------

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=float)
        emb = self._embed(texts)
        if self._centroid is None:
            # Without a fitted in-distribution cluster we cannot compute an
            # OOD score; return a neutral placeholder (cf. DeTeCtive).
            return np.full(emb.shape[0], 0.5, dtype=float)
        d = self._distance(emb)
        # Standardize relative to the soft-boundary radius of the human
        # cluster: texts beyond R score above 0.5 (more AI), texts inside R
        # score below 0.5 (more human). Higher distance => higher score.
        z = (d - self._radius) / self._ref_std
        scores = 1.0 / (1.0 + np.exp(-z / self.temperature))
        return np.asarray(scores, dtype=float)
