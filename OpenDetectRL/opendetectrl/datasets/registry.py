"""Dataset registry: register and load datasets by name."""
from typing import Dict, List, Optional

_DATASETS: Dict[str, type] = {}


def register_dataset(cls):
    """Decorator to register a dataset class.

    Usage::

        @register_dataset
        class CHEATDataset(BaseDataset):
            name = "cheat"
            ...
    """
    name = getattr(cls, "name", cls.__name__)
    _DATASETS[name] = cls
    return cls


def load_dataset(name: str, **kwargs):
    """Load a dataset by its registered name.

    Args:
        name: Registered dataset name (e.g. ``"cheat"``).
        **kwargs: Passed to the dataset constructor.

    Returns:
        An instance of :class:`BaseDataset`.
    """
    if name not in _DATASETS:
        raise KeyError(
            f"Dataset '{name}' not registered. Available: {list_datasets()}"
        )
    return _DATASETS[name](**kwargs)


def list_datasets() -> List[str]:
    """List all registered dataset names."""
    return sorted(_DATASETS.keys())


def get_dataset_info(name: str) -> dict:
    """Get metadata about a registered dataset."""
    if name not in _DATASETS:
        raise KeyError(f"Dataset '{name}' not registered.")
    cls = _DATASETS[name]
    return {
        "name": cls.name,
        "domain": cls.domain,
        "source_models": cls.source_models,
        "languages": cls.languages,
    }
