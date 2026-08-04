"""Black-box LLM-as-judge detectors.

These detectors do not inspect the internals of the *source* model that
(possibly) wrote a text.  Instead they query a separate black-box LLM
(OpenAI chat API by default, or a HuggingFace causal LM for local dev)
and use its response to decide whether the text is AI-generated.
"""
from .beat_llms import BeatLLMsDetector
from .chatgpt_as_detector import ChatGPTAsDetectorDetector

__all__ = [
    "ChatGPTAsDetectorDetector",
    "BeatLLMsDetector",
]