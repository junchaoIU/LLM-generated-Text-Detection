"""GECScore: Zero-Shot LLM-Generated Text Detection via GEC (Wu et al., COLING 2025).

Official code: https://github.com/NLP2CT/GECScore (``detector/GECScore.py``)

Method: prompt an LLM to correct grammatical errors in the text, then compute
ROUGE-2 F-score between the original and corrected text. Human text has more
grammatical errors => higher ROUGE-2 score after correction. LLM-generated text
has fewer errors => lower ROUGE-2 score.

Score direction: the official code uses ROUGE-2 F-score where higher = more
human. We negate it so higher score = more likely AI-generated.

The official code uses OpenAI API for GEC; this implementation supports both
HF models and OpenAI API.
"""
from typing import List, Optional

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector
from ...core.source_model import HFSourceModel


@register_detector("gecscore")
class GECScoreDetector(BaseDetector):
    """GECScore: Grammatical Error Correction score for AI text detection.

    Args:
        model: Pre-built :class:`HFSourceModel` for GEC generation.
        model_name: HuggingFace name (default ``gpt2``).
        gec_prompt: Prompt template for grammatical error correction.
        threshold: Decision threshold (official default 0.9244).
        use_openai: Whether to use OpenAI API instead of HF model.
        openai_model: OpenAI model name (official default ``gpt-4o-mini``).
        openai_api_key: API key for OpenAI.
    """

    DEFAULT_GEC_PROMPT = "Correct the grammar errors in the following text: {text}\nCorrected text:"

    def __init__(
        self,
        model: Optional[HFSourceModel] = None,
        model_name: str = "gpt2",
        gec_prompt: Optional[str] = None,
        threshold: float = 0.9244,
        use_openai: bool = False,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self.gec_prompt = gec_prompt or self.DEFAULT_GEC_PROMPT

        if not use_openai:
            if model is None:
                model = HFSourceModel(model_name, device=device)
            self.model = model
        else:
            self.model = None

    def _gec_openai(self, text: str) -> str:
        """Official ``chat_with_gpt4o``: GEC via OpenAI API."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_api_key)
        prompt = self.gec_prompt.format(text=text)
        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,
        )
        return completion.choices[0].message.content

    def _gec_hf(self, text: str) -> str:
        """GEC via HuggingFace model generation."""
        import torch
        prompt = self.gec_prompt.format(text=text)
        tokenized = self.model.tokenize(prompt)
        with torch.inference_mode():
            output = self.model.model.generate(
                **tokenized,
                max_new_tokens=min(len(text.split()) * 2, 512),
                do_sample=False,
                pad_token_id=self.model.tokenizer.eos_token_id,
            )
        generated = self.model.tokenizer.decode(
            output[0][tokenized["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def _compute_rouge2(self, text: str, gec_text: str) -> float:
        """Official: ROUGE-2 F-score between original and corrected text."""
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(text, gec_text, avg=True)
            return scores["rouge-2"]["f"]
        except ImportError:
            # Fallback: simple bigram overlap
            def bigrams(s):
                tokens = s.lower().split()
                return set(zip(tokens, tokens[1:]))
            bg1 = bigrams(text)
            bg2 = bigrams(gec_text)
            if not bg1:
                return 0.0
            overlap = len(bg1 & bg2)
            precision = overlap / max(len(bg2), 1)
            recall = overlap / len(bg1)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

    def score(self, texts) -> np.ndarray:
        scores = []
        for text in texts:
            if self.use_openai:
                gec_text = self._gec_openai(text)
            else:
                gec_text = self._gec_hf(text)
            rouge2_f = self._compute_rouge2(text, gec_text)
            # Official: higher ROUGE-2 = more human
            # We negate so higher score = more likely AI
            scores.append(-rouge2_f)
        return np.asarray(scores, dtype=float)
