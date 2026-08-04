"""Detector implementations.

Importing this package registers all built-in detectors in the global
``DETECTORS`` registry (requires the ``[detectors]`` extras: torch +
transformers).
"""
try:
    from . import statistical as _statistical  # noqa: F401
    from . import supervised as _supervised  # noqa: F401
    from . import watermark as _watermark  # noqa: F401
    from . import blackbox as _blackbox  # noqa: F401

    DETECTOR_EXTRAS_AVAILABLE = True
except ImportError:  # torch / transformers not installed
    DETECTOR_EXTRAS_AVAILABLE = False
