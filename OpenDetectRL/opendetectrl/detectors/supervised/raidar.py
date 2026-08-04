"""Raidar: geneRative AI Detection viA Rewriting.

Paper: Mao et al., "Raidar: geneRative AI Detection viA Rewriting",
ICLR 2024.
Repo: https://github.com/cvlab-columbia/RaidarLLMDetect

LLMs tend to modify human-written text more than AI-generated text when
asked to rewrite, because they perceive AI text as already high-quality.
Raidar detects AI text by prompting an LLM to rewrite the input and
measuring the word-level edit distance.  Lower edit distance → more likely
AI-generated.  Score is negated so higher = more AI.
"""
import os
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


@register_detector("raidar")
class RaidarDetector(BaseDetector):
    """Raidar: rewrite + edit distance detector.

    Uses a causal LM to rewrite the input text, then measures the word-level
    edit distance between original and rewritten text.  AI-generated text
    tends to have smaller edit distance (fewer modifications).  Score is
    ``-edit_distance / max_len`` so higher = more AI.

    Supports HuggingFace causal LMs (default GPT-2) or OpenAI API.
    """

    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        model_name: str = None,
        rewrite_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        max_length: int = 1024,
        threshold: float = 0.0,
        device: Optional[str] = None,
        use_openai: bool = False,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.rewrite_prompt = rewrite_prompt or (
            "Rewrite the following text in your own words. "
            "Keep the same meaning but change the wording.\n\n"
            "Text: {text}\n\nRewritten:"
        )

        if not use_openai:
            import torch
            import transformers

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
            self._torch = torch

    def _rewrite_hf(self, text: str) -> str:
        """Rewrite text using a HuggingFace causal LM."""
        torch = self._torch
        prompt = self.rewrite_prompt.format(text=text[:500])
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        rewritten = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return rewritten

    def _rewrite_openai(self, text: str) -> str:
        """Rewrite text using OpenAI chat API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        prompt = self.rewrite_prompt.format(text=text[:500])
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _compute_edit_ratio(self, original: str, rewritten: str) -> float:
        """Compute normalized word-level edit distance (0=same, 1=totally different)."""
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
            if self.use_openai:
                rewritten = self._rewrite_openai(text)
            else:
                rewritten = self._rewrite_hf(text)
            edit_ratio = self._compute_edit_ratio(text, rewritten)
            # Lower edit ratio = less change = more likely AI
            # Negate so higher score = more AI
            scores.append(-edit_ratio)
        return np.asarray(scores, dtype=float)
