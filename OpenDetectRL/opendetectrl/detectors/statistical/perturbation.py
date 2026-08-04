"""Mask-filling perturbation pipeline from the official DetectGPT code.

Ported from https://github.com/eric-mitchell/detect-gpt (``run.py``).
The official DetectLLM repo (``mbzuai-nlp/DetectLLM``) reuses exactly this
pipeline (``baselines/detectGPT.py``) to generate NPR perturbations.

Functions keep the names and logic of the original implementation:
``tokenize_and_mask``, ``count_masks``, ``replace_masks``, ``extract_fills``,
``apply_extracted_fills``, ``perturb_texts_`` and ``perturb_texts``.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


@dataclass
class PerturbConfig:
    """Hyper-parameters of the official DetectGPT perturbation pipeline."""

    span_length: int = 2
    pct_words_masked: float = 0.3  # actually pct * span_length / (span_length + 2 * buffer_size)
    buffer_size: int = 1
    mask_top_p: float = 1.0
    chunk_size: int = 20  # official run.py default
    mask_filling_model_name: str = "t5-large"
    random_fills: bool = False
    random_fills_tokens: bool = False


def tokenize_and_mask(text, span_length, pct, buffer_size, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]
    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


class MaskFillPerturber:
    """T5 mask-filling perturber following the official DetectGPT pipeline.

    Args:
        mask_model: A T5-style seq2seq model (or ``None`` when using random fills).
        mask_tokenizer: Its tokenizer.
        device: Device to run the mask model on.
        config: :class:`PerturbConfig` with the perturbation hyper-parameters.
        base_tokenizer: Required for ``random_fills_tokens``.
        fill_dictionary: Required for ``random_fills`` (official: all words of
            the evaluation corpus).
    """

    def __init__(
        self,
        mask_model,
        mask_tokenizer,
        device,
        config: Optional[PerturbConfig] = None,
        base_tokenizer=None,
        fill_dictionary: Optional[Sequence[str]] = None,
    ):
        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.device = device
        self.config = config or PerturbConfig()
        self.base_tokenizer = base_tokenizer
        self.fill_dictionary = fill_dictionary

        if self.config.random_fills and not self.config.random_fills_tokens:
            assert self.fill_dictionary is not None, (
                "random_fills requires a fill_dictionary (official: all words of the corpus)"
            )
        if self.config.random_fills_tokens:
            assert self.base_tokenizer is not None, (
                "random_fills_tokens requires the base tokenizer"
            )

    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=self.config.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def _random_fills_tokens(self, texts):
        # Official run.py: replace replace_pct of the tokens with random tokens
        tokens = self.base_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        valid_tokens = tokens.input_ids != self.base_tokenizer.pad_token_id
        replace_pct = self.config.pct_words_masked * (
            self.config.span_length
            / (self.config.span_length + 2 * self.config.buffer_size)
        )
        random_mask = torch.rand(tokens.input_ids.shape, device=self.device) < replace_pct
        random_mask &= valid_tokens
        random_tokens = torch.randint(
            0, self.base_tokenizer.vocab_size, (random_mask.sum(),), device=self.device
        )
        # while any of the random tokens are special tokens, replace them with random non-special tokens
        while any(
            self.base_tokenizer.decode(x) in self.base_tokenizer.all_special_tokens
            for x in random_tokens
        ):
            random_tokens = torch.randint(
                0, self.base_tokenizer.vocab_size, (random_mask.sum(),), device=self.device
            )
        tokens.input_ids[random_mask] = random_tokens
        return self.base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)

    def _random_fills_words(self, texts):
        # Official run.py: replace each <extra_id_*> with span_length random
        # words from FILL_DICTIONARY
        import random as _random

        perturbed_texts = []
        for text in texts:
            masked_text = tokenize_and_mask(
                text,
                self.config.span_length,
                self.config.pct_words_masked,
                self.config.buffer_size,
                ceil_pct=True,
            )
            filled_text = masked_text
            for fill_idx in range(count_masks([masked_text])[0]):
                fill = _random.sample(self.fill_dictionary, self.config.span_length)
                filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
            assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
            perturbed_texts.append(filled_text)
        return perturbed_texts

    def perturb_texts_(self, texts, ceil_pct=False):
        cfg = self.config
        if not cfg.random_fills:
            masked_texts = [
                tokenize_and_mask(x, cfg.span_length, cfg.pct_words_masked, cfg.buffer_size, ceil_pct)
                for x in texts
            ]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            while '' in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
                print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
                masked_texts = [
                    tokenize_and_mask(x, cfg.span_length, cfg.pct_words_masked, cfg.buffer_size, ceil_pct)
                    for idx, x in enumerate(texts)
                    if idx in idxs
                ]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = extract_fills(raw_fills)
                new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            if cfg.random_fills_tokens:
                perturbed_texts = self._random_fills_tokens(texts)
            else:
                perturbed_texts = self._random_fills_words(texts)

        return perturbed_texts

    def perturb_texts(self, texts: List[str], ceil_pct=False) -> List[str]:
        chunk_size = self.config.chunk_size
        if '11b' in self.config.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        for i in range(0, len(texts), chunk_size):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], ceil_pct=ceil_pct))
        return outputs


def load_mask_model(
    mask_filling_model_name: str,
    device,
    cache_dir: Optional[str] = None,
):
    """Load the T5 mask-filling model + tokenizer (official run.py style)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    mask_model = AutoModelForSeq2SeqLM.from_pretrained(
        mask_filling_model_name, cache_dir=cache_dir
    )
    mask_model.to(device)
    mask_model.eval()
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(
        mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir
    )
    mask_tokenizer.pad_token = mask_tokenizer.eos_token
    return mask_model, mask_tokenizer
