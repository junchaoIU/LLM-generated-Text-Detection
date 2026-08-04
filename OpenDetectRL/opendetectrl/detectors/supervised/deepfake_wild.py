"""Deepfake Text Detection in the Wild (Li et al., 2023).

Official code: https://github.com/yafuly/DeepfakeTextDetect
Model: https://huggingface.co/nealcly/detection-longformer

A Longformer-based sequence classifier fine-tuned on the DeepfakeTextDetect
dataset (~450k human- and machine-generated texts spanning 10 domains and 27
LLMs in real-world / "in the wild" settings). The model's classification head
emits a probability over two classes; following the dataset convention class 0
is *machine-generated* and class 1 is *human-written*, so the softmax
probability of class 0 is returned as the "more AI" score.

The official pipeline normalizes input text (Moses-style punctuation
normalization, tokenization fix-ups, and URL/email/phone stripping via the
``clean-text`` library) before scoring. This preprocessing is applied by
default and can be turned off with ``preprocess=False``. The ``clean-text``
package is optional; when it is missing the corresponding sub-step falls back
to a small stdlib routine while the rest of the normalization still runs.

Reference decision boundary: the authors refine an out-of-distribution
threshold on the raw class-0 logit (``logit_machine > 3.08583984375`` flags
machine-generated). This wrapper instead exposes the softmax probability of
class 0 with a default 0.5 threshold, consistent with the library convention.
"""
import re
from itertools import chain
from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseDetector
from ...core.registry import register_detector

# ``clean-text`` is optional; degrade gracefully when absent.
try:
    from cleantext import clean as _cleantext_clean
except ImportError:  # pragma: no cover
    _cleantext_clean = None


class _MosesPunctNormalizer:
    """Python port of the Moses punctuation normalizer.

    Faithfully ported from the official DeepfakeTextDetect ``utils.py``
    (https://github.com/moses-smt/mosesdecoder).
    """

    EXTRA_WHITESPACE = [  # lines 21 - 30
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]
    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]  # lines 33 - 34
    NORMALIZE_UNICODE = [  # lines 37 - 50
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]
    FRENCH_QUOTES = [  # lines 52 - 57
        ("\u00A0«\u00A0", r'"'),
        ("«\u00A0", r'"'),
        ("«", r'"'),
        ("\u00A0»\u00A0", r'"'),
        ("\u00A0»", r'"'),
        ("»", r'"'),
    ]
    HANDLE_PSEUDO_SPACES = [  # lines 59 - 67
        ("\u00A0%", r"%"),
        ("nº\u00A0", "nº "),
        ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"),
        ("\u00A0cm", r" cm"),
        ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"),
        ("\u00A0;", r";"),
        (",\u00A0", r", "),
        (r" +", r" "),
    ]
    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]
    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),  # don't fix period at end of sentence
    ]
    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>"),
    ]
    OTHER = [
        ("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>"),
    ]
    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        (""", '"'),
        (""", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(
        self,
        lang="en",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=False,
        post_remove_control_chars=False,
    ):
        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]
        if penn:  # Adds the penn substitutions after extra_whitespace regexes.
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)
        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)
        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)
        self.substitutions = list(chain(*self.substitutions))
        self.pre_replace_unicode_punct = pre_replace_unicode_punct
        self.post_remove_control_chars = post_remove_control_chars

    def normalize(self, text):
        """Returns a string with normalized punctuation."""
        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)
        for regexp, substitution in self.substitutions:
            text = re.sub(regexp, substitution, str(text))
        if self.post_remove_control_chars:
            text = self.remove_control_chars(text)
        return text.strip()

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text

    def remove_control_chars(self, text):
        # stdlib approximation of ``regex.sub(r"\p{C}", "", text)``
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)


def _tokenization_norm(text):
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return text


def _clean_text(text):
    # remove PLM special tokens
    plm_special_tokens = r'(<pad>)|(<s>)|(</s>)|(<unk>)|(<\|endoftext\|>)'
    text = re.sub(plm_special_tokens, "", text)
    # normalize punctuation
    text = _MosesPunctNormalizer().normalize(text)
    # normalize tokenization
    text = _tokenization_norm(text)
    # remove specific text patterns, e.g. url, email and phone number
    if _cleantext_clean is not None:
        text = _cleantext_clean(
            text,
            fix_unicode=True,               # fix various unicode errors
            to_ascii=True,                  # transliterate to closest ASCII representation
            lower=False,                    # lowercase text
            no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                  # replace all URLs with a special token
            no_emails=True,                # replace all email addresses with a special token
            no_phone_numbers=True,         # replace all phone numbers with a special token
            no_numbers=False,               # replace all numbers with a special token
            no_digits=False,                # replace all digits with a special token
            no_currency_symbols=False,      # replace all currency symbols with a special token
            no_punct=False,                 # remove punctuations
            replace_with_punct="",          # instead of removing punctuations you may replace them
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="<NUMBER>",
            replace_with_digit="<DIGIT>",
            replace_with_currency_symbol="<CUR>",
            lang="en",                      # set to 'de' for German special handling
        )
    else:
        # minimal stdlib fallback when ``clean-text`` is not installed
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)
        text = text.replace("\u00a0", " ")
    # keep common puncts only
    punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
    text = re.sub(punct_pattern, '', text)
    # remove specific patterns
    spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
    text = re.sub(spe_pattern, '', text)
    # remove redundant spaces
    text = " ".join(text.split())
    return text


def _rm_line_break(text):
    text = text.replace("\n", "\\n")
    text = re.sub(r'(?:\\n)*\\n', r'\\n', text)
    text = re.sub(r'^.{0,3}\\n', '', text)
    text = text.replace("\\n", " ")
    return text


def _preprocess_text(text):
    """Official DeepfakeTextDetect preprocessing pipeline."""
    text = _rm_line_break(text)
    text = _clean_text(text)
    return text


@register_detector("deepfake_wild")
class DeepfakeWildDetector(BaseDetector):
    """Deepfake Text Detection in the Wild (Li et al., 2023).

    A Longformer sequence classifier fine-tuned on the DeepfakeTextDetect
    corpus of human- and machine-generated texts collected across many domains
    and LLMs in real-world settings.

    Score = softmax probability of class 0 (machine-generated); higher = more
    likely AI, matching the library convention. The dataset labels class 0 as
    machine-generated and class 1 as human-written.

    Args:
        model_name: HuggingFace checkpoint. Defaults to the official
            ``nealcly/detection-longformer``.
        max_length: Tokenizer truncation length. The Longformer supports up to
            4096 tokens.
        batch_size: Number of texts scored per forward pass.
        ai_class_index: Override the index of the "AI" output class.
        preprocess: Apply the official text normalization before scoring
            (recommended). Set to ``False`` to skip it.
        threshold: Decision threshold on the score (higher = more AI).
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    DEFAULT_MODEL_NAME = "nealcly/detection-longformer"
    AI_CLASS_INDEX = 0  # dataset convention: 0 = machine-generated
    #: Official OOD-refined class-0 logit threshold (flags machine if exceeded).
    OOD_LOGIT_THRESHOLD = 3.08583984375

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 4096,
        batch_size: int = 16,
        ai_class_index: int = None,
        preprocess: bool = True,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        import transformers

        model_name = model_name or self.DEFAULT_MODEL_NAME
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.preprocess = preprocess
        if ai_class_index is not None:
            self.AI_CLASS_INDEX = ai_class_index
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    @torch.inference_mode()
    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.preprocess:
            texts = [_preprocess_text(t) for t in texts]
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # Official: padding + truncation to the model's context length
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokenized)
            probs = torch.nn.functional.softmax(outputs.logits, -1)
            scores.extend(probs[:, self.AI_CLASS_INDEX].cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)
