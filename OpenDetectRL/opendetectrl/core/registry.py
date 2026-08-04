"""Registry pattern for detectors and attackers.

Every detector / attacker registers itself under a string key, and is later
instantiated via the ``load_detector`` / ``load_attack`` factory functions.
"""
from typing import Any, Dict, List


class Registry:
    """A generic name -> class registry."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str):
        """Decorator that registers a class under ``key``."""

        def decorator(cls):
            if key in self._items:
                raise ValueError(f"[{self.name}] '{key}' is already registered.")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(
                f"[{self.name}] '{key}' not found. Available: {self.list()}"
            )
        return self._items[key]

    def list(self) -> List[str]:
        return sorted(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


# Global registries.
DETECTORS = Registry("detectors")
ATTACKERS = Registry("attackers")


def register_detector(key: str):
    """Decorator to register a detector class."""
    return DETECTORS.register(key)


def register_attacker(key: str):
    """Decorator to register an attacker class."""
    return ATTACKERS.register(key)


def load_detector(key: str, **kwargs):
    """Instantiate a registered detector by name."""
    return DETECTORS.get(key)(**kwargs)


def load_attack(key: str, **kwargs):
    """Instantiate a registered attacker by name."""
    return ATTACKERS.get(key)(**kwargs)


def list_detectors() -> List[str]:
    return DETECTORS.list()


def list_attackers() -> List[str]:
    return ATTACKERS.list()
