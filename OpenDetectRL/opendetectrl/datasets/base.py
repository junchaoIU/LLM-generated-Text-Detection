"""Dataset abstractions for AI-generated text detection.

Provides a unified ``BaseDataset`` interface and ``DatasetSample`` dataclass
so that detectors and evaluators can consume any benchmark dataset
through the same API.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSample:
    """A single sample from a detection dataset.

    Attributes:
        text: The text content to be classified.
        label: 0 = human-written, 1 = AI-generated.
        metadata: Optional dict with extra info (source model, domain, id, etc.).
    """

    text: str
    label: int  # 0 = human, 1 = AI
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for detection benchmark datasets.

    Subclasses must implement :meth:`load` which returns a list of
    :class:`DatasetSample`.

    Class attributes:
        name: Short identifier for the registry.
        domain: Text domain (e.g. ``"academic"``, ``"news"``).
        source_models: List of LLM names that generated the AI samples.
        languages: List of language codes.
    """

    name: str = "base"
    domain: str = "unknown"
    source_models: List[str] = []
    languages: List[str] = ["en"]

    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        self.cache_dir = cache_dir
        self.kwargs = kwargs

    @abstractmethod
    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[DatasetSample]:
        """Load dataset samples.

        Args:
            split: ``"train"``, ``"test"``, or ``"all"``.
            max_samples: If given, randomly subsample to this many samples.

        Returns:
            List of :class:`DatasetSample`.
        """
        ...

    def load_texts_labels(
        self, split: str = "test", max_samples: Optional[int] = None
    ) -> tuple:
        """Convenience: return ``(texts, labels)`` instead of samples."""
        samples = self.load(split=split, max_samples=max_samples)
        texts = [s.text for s in samples]
        labels = [s.label for s in samples]
        return texts, labels
