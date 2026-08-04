from opendetectrl.core import (
    BaseDetector,
    DetectionResult,
    SourceModel,
    list_attackers,
    list_detectors,
    load_attack,
    load_detector,
    register_attacker,
    register_detector,
)
from opendetectrl.attacks.base import BaseAttacker
from opendetectrl.datasets import (
    BaseDataset,
    CHEATDataset,
    DatasetSample,
    list_datasets,
    load_dataset,
)
from opendetectrl.evaluation.evaluator import (
    Evaluator,
    calibrate_threshold,
    compute_metrics,
    split_texts_labels,
)

# Import detectors package to register all built-in detectors.
try:
    from opendetectrl import detectors as _detectors  # noqa: F401
except ImportError:
    pass

__version__ = "0.0.1"

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "SourceModel",
    "BaseAttacker",
    "BaseDataset",
    "DatasetSample",
    "CHEATDataset",
    "Evaluator",
    "calibrate_threshold",
    "compute_metrics",
    "split_texts_labels",
    "register_detector",
    "register_attacker",
    "load_detector",
    "load_attack",
    "list_detectors",
    "list_attackers",
    "load_dataset",
    "list_datasets",
    "__version__",
]
