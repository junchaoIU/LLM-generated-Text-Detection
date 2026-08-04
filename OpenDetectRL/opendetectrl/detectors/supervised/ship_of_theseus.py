"""Ship of Theseus: Paraphrasing Capabilities for Detecting LLM-Generated Text.

Paper: Gao et al., "A Ship of Theseus: Paraphrasing Capabilities for Detecting
LLM-Generated Text", ACL 2024. arXiv 2311.08374.

Method: LLMs asked to paraphrase a piece of text tend to modify *human-written*
text more heavily than *AI-generated* text, because the model already perceives
AI text as well written and close to its own distribution.  The detector
paraphrases the input text with an LLM and measures the word-level edit
distance between the original and the rewritten version.  Smaller edit
distance => more likely AI-generated.
"""
from typing import List, Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector


def _word_edit_distance(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein distance at the word level."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


@register_detector("ship_of_theseus")
class ShipOfTheseusDetector(BaseDetector):
    """Ship of Theseus paraphrase-based detector.

    Paraphrases each input with a causal LM and scores ``-edit_ratio`` where
    ``edit_ratio`` is the word-level edit distance normalized by the max of
    the original / rewritten lengths.  Higher score = more likely AI-generated.

    Args:
        model_name: HF causal LM used for paraphrasing (default ``gpt2``).
        max_new_tokens: Max tokens to generate for the paraphrase.
        max_length: Truncation length for the prompt.
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        model_name: str = None,
        max_new_tokens: int = 200,
        max_length: int = 1024,
        threshold: float = 0.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.rewrite_prompt = (
            "Rewrite the following text in your own words. "
            "Keep the same meaning but change the wording as much as possible.\n\n"
            "Text: {text}\n\nRewritten:"
        )

    def _get_model(self):
        """Lazily load the causal LM for paraphrasing."""
        if not hasattr(self, "_models"):
            import torch
            import transformers

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(self.device)
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name).to(
                self.device
            )
            model.eval()
            self._models = (model, tokenizer, torch)
        return self._models

    def _rewrite(self, text: str) -> str:
        model, tokenizer, torch = self._get_model()
        prompt = self.rewrite_prompt.format(text=text[:500])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        rewritten = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return rewritten

    def _edit_ratio(self, original: str, rewritten: str) -> float:
        w1 = original.split()
        w2 = rewritten.split()
        max_len = max(len(w1), len(w2))
        if max_len == 0:
            return 0.0
        return _word_edit_distance(w1, w2) / max_len

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores = []
        for text in texts:
            rewritten = self._rewrite(text)
            # Lower edit ratio = less change = more likely AI.
            scores.append(-self._edit_ratio(text, rewritten))
        return np.asarray(scores, dtype=float)