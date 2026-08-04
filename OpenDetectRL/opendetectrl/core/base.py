"""Core abstractions: ``DetectionResult`` and ``BaseDetector``."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class DetectionResult:
    """Standardized output of a detector for a single text."""

    #: Higher = more likely AI-generated.
    score: float
    #: "ai" or "human".
    label: str
    #: Confidence in the label, in [0, 1].
    confidence: float = 0.0
    #: Detector-specific extra info (e.g. avg_log_likelihood, mean_curvature).
    metadata: Dict = field(default_factory=dict)


class BaseDetector(ABC):
    """Base class for all detectors.

    Subclasses must implement :meth:`score`. By convention the returned score
    is normalized so that **higher = more likely AI-generated**, regardless of
    the detector's internal signal direction.
    """

    #: Whether this detector needs access to the source LLM (white-box).
    requires_source_model: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.threshold = threshold
        self.device = device

    @abstractmethod
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D array of scores, one per input text."""

    def predict(self, texts: Sequence[str]) -> List[DetectionResult]:
        """Threshold the scores into per-text :class:`DetectionResult`."""
        scores = np.asarray(self.score(texts), dtype=float)
        results: List[DetectionResult] = []
        for s in scores:
            s = float(s)
            label = "ai" if s > self.threshold else "human"
            results.append(DetectionResult(score=s, label=label))
        return results

    def __call__(self, texts: Sequence[str]) -> List[DetectionResult]:
        return self.predict(texts)
