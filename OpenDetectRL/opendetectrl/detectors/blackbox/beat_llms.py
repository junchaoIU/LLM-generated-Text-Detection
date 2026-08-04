"""Beat LLMs at Their Own Game: zero-shot detection via query variance.

Paper: Bhattacharjee et al., "Beat LLMs at Their Own Game: Zero-Shot
LLM-Generated Text Detection via Querying ChatGPT", EMNLP 2023.
Repo: https://github.com/thunlp/LLM-generated-text-detection

Method: a black-box LLM is asked to continue the given text multiple
times.  AI-generated text is more predictable to an LLM, so its
continuations are more consistent (lower variance / smaller spread),
whereas human-written text leads to more diverse continuations.  The
detection score is ``-divergence``, so that lower divergence (higher
predictability) maps to a higher (more AI) score.

This implementation supports any HuggingFace causal LM (default: GPT-2)
for local sampling, or an OpenAI chat model.
"""
import os
from typing import List, Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector

_CONTINUE_PROMPT_TEMPLATE = (
    "Continue the following text in a natural way.\n\n{text}\n"
)


@register_detector("beat_llms")
class BeatLLMsDetector(BaseDetector):
    """Beat-LLMs: detection via sampled-continuation divergence.

    Samples ``num_queries`` continuations of the target text from a
    black-box LLM and measures their average pairwise divergence.  The
    returned score is ``-divergence`` (higher = more AI, i.e. lower
    variance = more predictable = more AI-like).
    """

    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        model_name: str = "gpt2",
        num_queries: int = 5,
        temperature: float = 1.0,
        max_new_tokens: int = 50,
        use_openai: bool = False,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name
        self.num_queries = max(1, int(num_queries))
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

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

    def _build_prompt(self, target_text: str) -> str:
        """Build the "continue this text" prompt."""
        return _CONTINUE_PROMPT_TEMPLATE.format(text=target_text)

    def _sample_hf(self, prompt: str) -> List[str]:
        """Sample multiple continuations from a HuggingFace causal LM."""
        torch = self._torch
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_queries,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        continuations = []
        for i in range(self.num_queries):
            seq = outputs[i][input_ids.shape[1]:]
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            continuations.append(text)
        return continuations

    def _sample_openai(self, prompt: str) -> List[str]:
        """Sample multiple continuations from the OpenAI chat API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        continuations = []
        for _ in range(self.num_queries):
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=1,
            )
            continuation = response.choices[0].message.content.strip()
            continuations.append(continuation)
        return continuations

    @staticmethod
    def _tokenize_continuation(text: str) -> List[str]:
        """Split a continuation into whitespace-delimited tokens."""
        return [t for t in text.split() if t]

    def _divergence(self, continuations: List[str]) -> float:
        """Average pairwise normalized n-gram overlap divergence.

        Returns 1 - average pairwise Jaccard overlap of the token sets,
        i.e. a value in [0, 1] where higher = more diverse continuations.
        """
        if len(continuations) < 2:
            return 0.0

        token_sets = [set(self._tokenize_continuation(c)) for c in continuations]
        total = 0.0
        pairs = 0
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                if not token_sets[i] and not token_sets[j]:
                    overlap = 1.0
                elif not token_sets[i] or not token_sets[j]:
                    overlap = 0.0
                else:
                    union = len(token_sets[i] | token_sets[j])
                    if union == 0:
                        overlap = 1.0
                    else:
                        overlap = len(token_sets[i] & token_sets[j]) / union
                total += 1.0 - overlap
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def _score_text(self, prompt: str) -> float:
        """Score a single text via sampled-continuation divergence.

        Higher = more AI: predictable (low-divergence) continuations map to
        a high score, so the score is ``-divergence``.
        """
        if self.use_openai:
            continuations = self._sample_openai(prompt)
        else:
            continuations = self._sample_hf(prompt)
        divergence = self._divergence(continuations)
        return -divergence

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        scores = []
        for text in texts:
            prompt = self._build_prompt(text)
            scores.append(self._score_text(prompt))
        return np.asarray(scores, dtype=float)