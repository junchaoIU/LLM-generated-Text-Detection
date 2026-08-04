"""Base class for attacks that modify text to evade detectors."""
from abc import ABC, abstractmethod
from typing import List, Sequence


class BaseAttacker(ABC):
    """Base class for all attackers.

    An attacker transforms input texts (typically AI-generated) so that
    detectors are more likely to misclassify them as human-written.
    """

    @abstractmethod
    def attack(self, texts: Sequence[str]) -> List[str]:
        """Return the attacked versions of the input texts."""

    def __call__(self, texts: Sequence[str]) -> List[str]:
        return self.attack(texts)
