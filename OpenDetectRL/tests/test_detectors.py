"""Tests for detector implementations.

These tests NEVER download models from the HuggingFace hub:

* criterion functions are unit-tested with synthetic logits;
* end-to-end smoke tests use a tiny randomly-initialized GPT-2 built in-memory
  together with a tiny word-level tokenizer;
* DetectGPT / NPR use the official ``random_fills_tokens`` option so that no
  T5 mask-filling model is required.

Full-size tests (gpt2, falcon-7b, t5-*, roberta-*-openai-detector) are meant
to run on the cluster; gate them with ``ODRL_CLUSTER_TESTS=1``.
"""
import os

import numpy as np
import pytest
import torch

from opendetectrl import list_detectors, load_detector
from opendetectrl.core import HFSourceModel
from opendetectrl.detectors.statistical import baselines, fast_detect_gpt
from opendetectrl.detectors.statistical.binoculars import entropy as binoculars_entropy
from opendetectrl.detectors.statistical.binoculars import perplexity as binoculars_perplexity
from opendetectrl.detectors.statistical.perturbation import (
    apply_extracted_fills,
    count_masks,
    extract_fills,
    tokenize_and_mask,
)

CLUSTER_TESTS = os.environ.get("ODRL_CLUSTER_TESTS") == "1"
cluster_only = pytest.mark.skipif(not CLUSTER_TESTS, reason="cluster tests disabled (set ODRL_CLUSTER_TESTS=1)")

TEXTS = [
    "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9",
    "w10 w11 w12 w13 w14 w15 w0 w1 w2 w3",
]


# ---------------------------------------------------------------------------
# Synthetic-logits unit tests (no model required)
# ---------------------------------------------------------------------------


class TestBaselineCriteria:
    def test_likelihood(self):
        # pos0: softmax([log3, 0]) = [0.75, 0.25], label 0 -> log(0.75)
        # pos1: softmax([0, 0]) = [0.5, 0.5], label 1 -> log(0.5)
        logits = torch.tensor([[[np.log(3.0), 0.0], [0.0, 0.0]]])
        labels = torch.tensor([[0, 1]])
        expected = (np.log(0.75) + np.log(0.5)) / 2
        assert baselines.get_likelihood(logits, labels) == pytest.approx(expected)

    def test_rank(self):
        # pos0: label 1 is the top token -> rank 1 -> -1
        # pos1: logits [5,1,2], label 2 -> rank 2 -> -2
        logits = torch.tensor([[[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]]])
        labels = torch.tensor([[1, 2]])
        assert baselines.get_rank(logits, labels) == pytest.approx(-(1 + 2) / 2)

    def test_logrank(self):
        logits = torch.tensor([[[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]]])
        labels = torch.tensor([[1, 2]])
        assert baselines.get_logrank(logits, labels) == pytest.approx(-(np.log(1) + np.log(2)) / 2)
        assert baselines.get_positive_logrank(logits, labels) == pytest.approx((np.log(1) + np.log(2)) / 2)

    def test_entropy(self):
        logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        labels = torch.tensor([[0]])
        assert baselines.get_entropy(logits, labels) == pytest.approx(np.log(4))


class TestFastDetectGPTCriteria:
    def test_analytic_deterministic_and_direction(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 8, 50)
        labels_high = logits.argmax(dim=-1)  # "machine-like": always top token
        labels_low = logits.argmin(dim=-1)  # "human-like": always bottom token
        d_high = fast_detect_gpt.get_sampling_discrepancy_analytic(logits, logits, labels_high)
        d_low = fast_detect_gpt.get_sampling_discrepancy_analytic(logits, logits, labels_low)
        assert d_high == pytest.approx(
            fast_detect_gpt.get_sampling_discrepancy_analytic(logits, logits, labels_high)
        )
        assert d_high > d_low

    def test_sampling_version_finite(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 20)
        labels = logits.argmax(dim=-1)
        d = fast_detect_gpt.get_sampling_discrepancy(logits, logits, labels)
        assert np.isfinite(d)


class TestBinocularsMetrics:
    def _encoding(self, input_ids, attention_mask):
        class Enc:  # duck-typed BatchEncoding
            pass

        enc = Enc()
        enc.input_ids = input_ids
        enc.attention_mask = attention_mask
        return enc

    def test_perplexity_matches_manual_ce(self):
        # logits seq must equal input_ids seq so that shifted dims match
        logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 1.0, 0.0]]])
        enc = self._encoding(torch.tensor([[0, 2, 1]]), torch.tensor([[1, 1, 1]]))
        ce1 = torch.nn.functional.cross_entropy(logits[0, 0], torch.tensor(2))
        ce2 = torch.nn.functional.cross_entropy(logits[0, 1], torch.tensor(1))
        ppl = binoculars_perplexity(enc, logits)
        assert ppl[0] == pytest.approx(((ce1 + ce2) / 2).item())

    def test_perplexity_masks_padding(self):
        # 3 positions, last is padding (mask=0)
        logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [10.0, -10.0, 0.0]]])
        enc = self._encoding(torch.tensor([[0, 2, 1]]), torch.tensor([[1, 1, 0]]))
        ppl_masked = binoculars_perplexity(enc, logits)[0]
        # padding at the last position is dropped: only the first CE counts
        ce1 = torch.nn.functional.cross_entropy(logits[0, 0], torch.tensor(2))
        assert ppl_masked == pytest.approx(ce1.item())

    def test_cross_entropy_positive_and_finite(self):
        torch.manual_seed(0)
        p_logits = torch.randn(1, 5, 30)
        q_logits = torch.randn(1, 5, 30)
        enc = self._encoding(torch.randint(0, 30, (1, 5)), torch.ones(1, 5, dtype=torch.long))
        x_ppl = binoculars_entropy(p_logits, q_logits, enc, pad_token_id=29)
        assert x_ppl[0] > 0 and np.isfinite(x_ppl[0])


class TestPerturbationPipeline:
    def test_tokenize_and_mask(self):
        np.random.seed(0)
        text = " ".join(f"w{i}" for i in range(30))
        masked = tokenize_and_mask(text, span_length=2, pct=0.3, buffer_size=1, ceil_pct=True)
        n = count_masks([masked])[0]
        assert n > 0
        assert f"<extra_id_{n - 1}>" in masked
        assert len(masked.split()) < 30

    def test_extract_and_apply_fills(self):
        masked = "w0 w1 <extra_id_0> w4 <extra_id_1> w7"
        # extract_fills expects the T5 output format: fills are bracketed by
        # the <extra_id_*> markers; one text -> list of fills
        t5_output = "<extra_id_0> aa bb <extra_id_1> cc <extra_id_2>"
        fills = extract_fills(["<pad>" + t5_output + "</s>"])
        assert fills == [["aa bb", "cc"]]
        filled = apply_extracted_fills([masked], fills)
        assert filled == ["w0 w1 aa bb w4 cc w7"]


# ---------------------------------------------------------------------------
# End-to-end smoke tests with a tiny in-memory GPT-2 (no downloads)
# ---------------------------------------------------------------------------


def _build_tiny_source_model() -> HFSourceModel:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = ["<pad>", "<unk>", "<bos>", "<eos>"] + [f"w{i}" for i in range(16)]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2
    )
    model = GPT2LMHeadModel(config)
    return HFSourceModel(model_name_or_path="tiny-gpt2", model=model, tokenizer=tokenizer, device="cpu")


@pytest.fixture(scope="module")
def tiny_model() -> HFSourceModel:
    return _build_tiny_source_model()


class TestDetectorsSmoke:
    @pytest.mark.parametrize(
        "name", ["log_likelihood", "rank", "log_rank", "entropy", "fast_detect_gpt", "lrr"]
    )
    def test_single_model_detectors(self, tiny_model, name):
        detector = load_detector(name, model=tiny_model)
        scores = detector.score(TEXTS)
        assert scores.shape == (len(TEXTS),)
        assert np.isfinite(scores).all()
        results = detector.predict(TEXTS)
        assert len(results) == len(TEXTS)
        assert all(r.label in ("ai", "human") for r in results)

    def test_detect_gpt_random_fills_tokens(self, tiny_model):
        torch.manual_seed(0)
        detector = load_detector(
            "detect_gpt",
            model=tiny_model,
            n_perturbations=2,
            random_fills_tokens=True,
        )
        scores = detector.score(TEXTS)
        assert scores.shape == (len(TEXTS),)
        assert np.isfinite(scores).all()

    def test_npr_random_fills_tokens(self, tiny_model):
        torch.manual_seed(0)
        detector = load_detector(
            "npr",
            model=tiny_model,
            n_perturbations=2,
            random_fills_tokens=True,
        )
        scores = detector.score(TEXTS)
        assert scores.shape == (len(TEXTS),)
        assert np.isfinite(scores).all()
        assert (scores > 0).all()  # log-ranks are positive

    def test_binoculars(self, tiny_model):
        detector = load_detector(
            "binoculars",
            observer_model=tiny_model,
            performer_model=tiny_model,
            device="cpu",
        )
        raw = detector.compute_score(TEXTS)
        assert len(raw) == len(TEXTS)
        assert all(np.isfinite(v) for v in raw)
        scores = detector.score(TEXTS)
        assert np.allclose(scores, -np.asarray(raw))
        results = detector.predict(TEXTS)
        assert len(results) == len(TEXTS)
        assert all("binoculars_raw" in r.metadata for r in results)

    def test_registry_lists_detectors(self):
        names = list_detectors()
        for expected in [
            "log_likelihood",
            "rank",
            "log_rank",
            "entropy",
            "fast_detect_gpt",
            "detect_gpt",
            "lrr",
            "npr",
            "binoculars",
            "roberta_openai",
            "roberta_large_openai",
        ]:
            assert expected in names


class TestNewDetectorsRegistration:
    """New detectors added in this batch are all registered (no model needed)."""

    def test_registration(self):
        names = list_detectors()
        for expected in [
            "coco",
            "conda",
            "tweepfake",
            "gpt_paternity_test",
            "ship_of_theseus",
            "checkgpt",
            "private_watermark",
            "cater",
            "drw",
            "chatgpt_as_detector",
            "beat_llms",
        ]:
            assert expected in names, f"{expected} not registered"


class TestCATERDetector:
    def test_scores_binary_by_trigger_presence(self):
        detector = load_detector("cater", trigger_phrase="the quick brown fox")
        scores = detector.score(
            ["a text containing the quick brown fox jump", "a plain human text"]
        )
        assert scores.tolist() == [1.0, 0.0]

    def test_requires_trigger_phrase(self):
        with pytest.raises(RuntimeError):
            load_detector("cater")


class TestDRWDetector:
    def test_scores_fraction_of_triggers_found(self):
        detector = load_detector("drw", trigger_set=["alpha", "beta", "gamma"])
        scores = detector.score(
            ["alpha beta here", "gamma only", "nothing matches"]
        )
        assert scores.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])

    def test_requires_trigger_set(self):
        with pytest.raises(RuntimeError):
            load_detector("drw")


class TestPrivateWatermarkDetector:
    def test_requires_checkpoint(self):
        with pytest.raises(RuntimeError):
            load_detector("private_watermark")


class TestGPTPaternityTestDetector:
    def test_requires_fit_before_score(self):
        detector = load_detector("gpt_paternity_test")
        with pytest.raises(RuntimeError):
            detector.score(["w0 w1 w2 w3"])


class TestDRGDDetector:
    def test_requires_checkpoint(self):
        with pytest.raises(RuntimeError):
            load_detector("drgd")


class TestNotAIDetector:
    def test_registered(self):
        detector = load_detector("notai")
        assert detector is not None


# ---------------------------------------------------------------------------
# Cluster tests (real models; run with ODRL_CLUSTER_TESTS=1 on the cluster)
# ---------------------------------------------------------------------------


@cluster_only
class TestCluster:
    def test_roberta_openai_direction(self):
        detector = load_detector("roberta_openai")
        ai_text = (
            "In recent years, artificial intelligence has made significant progress in "
            "a variety of fields. Machine learning algorithms have been applied to tasks "
            "such as image recognition, natural language processing, and game playing."
        )
        scores = detector.score([ai_text])
        assert 0.0 <= scores[0] <= 1.0

    def test_gpt2_baselines(self):
        detector = load_detector("log_likelihood", model_name="gpt2")
        scores = detector.score(TEXTS)
        assert np.isfinite(scores).all()
