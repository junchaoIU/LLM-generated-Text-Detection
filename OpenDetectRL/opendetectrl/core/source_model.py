"""Source-model abstraction for white-box detectors.

White-box detectors (DetectGPT, Fast-DetectGPT, NPR, ...) need token
log-probabilities or hidden states from the model that (may have) generated
the text. This abstraction lets us plug in HuggingFace models or API-based
models (e.g. OpenAI) behind a single interface.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

import torch


class SourceModel(ABC):
    """Unified interface to the (candidate) source LLM."""

    @abstractmethod
    def get_logprobs(self, texts: Sequence[str]) -> Any:
        """Return per-token log-probabilities for each text."""

    @abstractmethod
    def get_hidden_states(self, texts: Sequence[str]) -> Any:
        """Return hidden states for each text."""


class HFSourceModel(SourceModel):
    """HuggingFace causal LM used as the (candidate) source model.

    Loading conventions follow the official detector codebases
    (fast-detect-gpt ``model.py``, detect-gpt ``run.py``): right padding,
    ``pad_token_id = eos_token_id``, model in ``eval()`` mode.

    Args:
        model_name_or_path: HuggingFace hub name or local path.
        device: Target device; defaults to CUDA if available, else CPU.
        cache_dir: HuggingFace cache directory.
        torch_dtype: Optional dtype for the model weights (e.g. ``torch.float16``).
        trust_remote_code: Passed through to ``from_pretrained``.
        token: HuggingFace token for gated models.
        model: Pre-loaded model (skips ``from_pretrained``; useful for tests).
        tokenizer: Pre-loaded tokenizer (skips ``from_pretrained``).
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None,
        model=None,
        tokenizer=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name_or_path = model_name_or_path

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_kwargs = {"trust_remote_code": trust_remote_code}
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if token is not None:
                model_kwargs["token"] = token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, **model_kwargs
            )
            self.model.to(self.device)

            # Official fast-detect-gpt convention (scripts/model.py).
            tok_kwargs = {"padding_side": "right", "trust_remote_code": trust_remote_code}
            if token is not None:
                tok_kwargs["token"] = token
            if "facebook/opt-" in model_name_or_path:
                tok_kwargs["fast"] = False  # "Using non-fast tokenizer for OPT"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, **tok_kwargs
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    # ------------------------------------------------------------------
    # Low-level scoring primitives (aligned with the official scripts)
    # ------------------------------------------------------------------
    def tokenize(self, text: str):
        """Tokenize a single text without padding (official scripts style)."""
        return self.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(
            self.device
        )

    def tokenize_batch(self, texts: Sequence[str], max_length: Optional[int] = None):
        """Tokenize a batch with right padding (official Binoculars style)."""
        batch_size = len(texts)
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=max_length is not None,
            max_length=max_length,
            return_token_type_ids=False,
        ).to(self.device)

    @torch.inference_mode()
    def get_logits(self, text: str):
        """Return ``(logits, input_ids)`` for a single text (no padding)."""
        tokenized = self.tokenize(text)
        logits = self.model(**tokenized).logits
        return logits, tokenized.input_ids

    @torch.inference_mode()
    def get_logits_batch(self, texts: Sequence[str], max_length: Optional[int] = None):
        """Return ``(logits, encodings)`` for a padded batch."""
        encodings = self.tokenize_batch(texts, max_length=max_length)
        logits = self.model(**encodings).logits
        return logits, encodings

    @torch.inference_mode()
    def get_mean_log_likelihood(self, text: str) -> float:
        """Mean per-token log-likelihood.

        Identical to the official DetectGPT ``get_ll``:
        ``-model(**tokenized, labels=input_ids).loss``.
        """
        tokenized = self.tokenize(text)
        labels = tokenized.input_ids
        return -self.model(**tokenized, labels=labels).loss.item()

    def get_logprobs(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Per-token log-probabilities ``log p(x_t | x_{<t})`` for t >= 1."""
        out = []
        for text in texts:
            logits, input_ids = self.get_logits(text)
            lprobs = torch.log_softmax(logits, dim=-1)
            token_lprobs = lprobs[0, :-1].gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            out.append(token_lprobs.cpu())
        return out

    @torch.inference_mode()
    def get_hidden_states(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Last-layer hidden states for each text (no padding)."""
        out = []
        for text in texts:
            tokenized = self.tokenize(text)
            outputs = self.model(**tokenized, output_hidden_states=True)
            out.append(outputs.hidden_states[-1][0].cpu())
        return out
