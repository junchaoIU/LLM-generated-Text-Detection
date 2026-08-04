"""Supervised detectors."""
from .checkgpt import CheckGPTDetector
from .coco import CoCoDetector
from .conda import ConDADetector
from .damage import DAMAGEDetector
from .deepfake_wild import DeepfakeWildDetector
from .detective import DeTeCtiveDetector
from .drgd import DRGDDetector
from .gpt_paternity import GPTPaternityTestDetector
from .grover import GROVERDetector
from .hf_classifier import DeBERTaClassifierDetector, HFClassifierDetector
from .mpu import MPUDetector
from .notai import NotAIDetector
from .ood_detection import OODDetectionDetector
from .outfox import OUTFOXDetector
from .poger import POGERDetector
from .radar import RADARDetector, RADARMixDetector
from .raidar import RaidarDetector
from .remodetect import ReMoDetectDetector
from .roberta_openai import RoBERTaLargeOpenAIDetector, RoBERTaOpenAIDetector
from .ship_of_theseus import ShipOfTheseusDetector
from .tweepfake import TweepFakeDetector

__all__ = [
    "RoBERTaOpenAIDetector",
    "RoBERTaLargeOpenAIDetector",
    "RADARDetector",
    "RADARMixDetector",
    "DeTeCtiveDetector",
    "GROVERDetector",
    "DAMAGEDetector",
    "ReMoDetectDetector",
    "DeBERTaClassifierDetector",
    "HFClassifierDetector",
    "POGERDetector",
    "DeepfakeWildDetector",
    "MPUDetector",
    "OODDetectionDetector",
    "OUTFOXDetector",
    "RaidarDetector",
    "CoCoDetector",
    "ConDADetector",
    "TweepFakeDetector",
    "GPTPaternityTestDetector",
    "ShipOfTheseusDetector",
    "CheckGPTDetector",
    "DRGDDetector",
    "NotAIDetector",
]
