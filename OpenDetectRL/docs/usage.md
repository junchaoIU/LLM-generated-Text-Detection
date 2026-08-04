# OpenDetectRL Usage Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Datasets](#datasets)
  - [CHEAT Dataset](#cheat-dataset)
- [Evaluation](#evaluation)
  - [Three Evaluation Modes](#evaluator--three-evaluation-modes)
- [Detector Reference](#detector-reference)
  - [Statistical / Zero-shot Detectors](#statistical--zero-shot-detectors)
  - [Supervised / Neural Detectors](#supervised--neural-detectors)
  - [Watermark Detectors](#watermark-detectors)
  - [Black-box / LLM-as-Judge Detectors](#black-box--llm-as-judge-detectors)
- [End-to-End Example: DetectGPT on CHEAT](#end-to-end-example-detectgpt-on-cheat)

---

## Quick Start

```python
from opendetectrl import load_detector, list_detectors

# List all registered detectors
print(list_detectors())

# Load a detector by name
detector = load_detector("log_likelihood", model_name="gpt2")

# Score texts (higher = more likely AI-generated)
scores = detector.score(["Hello world!", "This is AI-generated text."])

# Predict with threshold
results = detector.predict(["Hello world!", "This is AI-generated text."])
for r in results:
    print(f"score={r.score:.4f}  label={r.label}")
```

## Core API

### `load_detector(name, **kwargs) -> BaseDetector`

Instantiate a registered detector by its registry name.

### `list_detectors() -> list[str]`

Return all registered detector names.

### `BaseDetector`

| Method | Description |
|---|---|
| `score(texts) -> np.ndarray` | Return a 1-D array of scores (higher = more likely AI). |
| `predict(texts) -> list[DetectionResult]` | Threshold scores into per-text results. |
| `__call__(texts)` | Alias for `predict()`. |

### `DetectionResult`

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Detection score (higher = more AI). |
| `label` | `str` | `"ai"` or `"human"`. |
| `confidence` | `float` | Confidence in [0, 1] (default 0.0). |
| `metadata` | `dict` | Detector-specific extra info. |

All constructors accept a shared `threshold` and `device` parameter:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Decision boundary; `score > threshold` → AI. |
| `device` | `str \| None` | `None` | Torch device (`"cuda"`, `"cpu"`, or auto). |

---

## Datasets

### `load_dataset(name, **kwargs) -> BaseDataset`

Instantiate a registered dataset by name.

### `list_datasets() -> list[str]`

Return all registered dataset names.

### CHEAT Dataset

**CHEAT** (Yu et al., IEEE TBD 2025) is a large-scale dataset for detecting ChatGPT-written academic abstracts. It pairs human-written IEEE abstracts with ChatGPT-generated, polished, and hybrid-fusion versions.

**Official repo**: <https://github.com/botianzhe/CHEAT>

**Files**:

| File | Label | Description |
|---|---|---|
| `ieee-init.jsonl` | 0 (human) | Human-written IEEE abstracts |
| `ieee-chatgpt-generation.jsonl` | 1 (AI) | ChatGPT-generated abstracts |
| `ieee-chatgpt-polish.jsonl` | 1 (AI) | ChatGPT-polished abstracts |
| `ieee-chatgpt-fusion.jsonl` | 1 (AI) | Human/ChatGPT hybrid abstracts |

**Constructor parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cache_dir` | `str \| None` | `~/.cache/opendetectrl/cheat` | Directory for caching downloaded files. |
| `subsets` | `list[str] \| None` | `["generation", "polish", "fusion"]` | Which AI subsets to include. |
| `download` | `bool` | `True` | Whether to auto-download missing files. |

**`load()` parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `split` | `str` | `"all"` | CHEAT has no official split; `"train"`/`"test"` return the full dataset. |
| `max_samples` | `int \| None` | `None` | If given, randomly subsample to at most this many (balanced human/AI). |

**Usage**:

```python
from opendetectrl import load_dataset

# Load CHEAT (auto-downloads to ~/.cache/opendetectrl/cheat)
ds = load_dataset("cheat")

# Load only the "generation" subset, subsample to 200 samples
ds = load_dataset("cheat", subsets=["generation"], download=False)
samples = ds.load(max_samples=200)

texts, labels = ds.load_texts_labels(max_samples=200)
# texts:  list[str]   — the abstract text
# labels: list[int]   — 0 = human, 1 = AI
```

**Using local files (no download)**:

```python
ds = load_dataset(
    "cheat",
    cache_dir="/path/to/cheat/data",
    subsets=["generation"],
    download=False,  # raise FileNotFoundError if files are missing
)
samples = ds.load(max_samples=100)
```

---

## Evaluation

### `compute_metrics(labels, scores, threshold=0.5, target_fpr=0.01) -> dict`

Compute standard binary-classification metrics.

| Metric | Description |
|---|---|
| `accuracy` | Classification accuracy at the given threshold. |
| `precision` | Precision (AI class). |
| `recall` | Recall / TPR at the given threshold. |
| `f1` | F1 score at the given threshold. |
| `roc_auc` | Area under the ROC curve (threshold-independent). |
| `avg_precision` | Average precision / AUPRC (threshold-independent). |
| `tpr_at_fpr_{target_fpr}` | TPR at a fixed FPR level (default `tpr_at_fpr_0.01`). Critical for low-FPR operating points. |
| `f1_at_fpr_{target_fpr}` | F1 at the threshold where FPR ≈ `target_fpr` (default `f1_at_fpr_0.01`). |

Set `target_fpr` to other values (e.g. `0.001`, `0.05`) for different operating points.

### `calibrate_threshold(scores, labels, method="youden") -> float`

Find the optimal decision threshold on a calibration set.

| Method | Description |
|---|---|
| `"youden"` | Maximize Youden's J = TPR − FPR (default). Best for balanced datasets. |
| `"f1"` | Maximize F1 score. Best when the positive (AI) class is the minority. |
| `"eer"` | Equal Error Rate — threshold where FPR ≈ FNR. Best when you want symmetric error rates. |

### `split_texts_labels(texts, labels, test_size=0.5) -> (train, test)`

Split any dataset into train/test halves with stratification. Use it when
the dataset has no official train/test split (CHEAT, HC3, M4, etc.).

### `Evaluator(threshold=0.5, target_fpr=0.01)` — Three Evaluation Modes

```python
from opendetectrl.evaluation import Evaluator

# Default: target_fpr=0.01 (report TPR@FPR=1%)
ev = Evaluator()

# Custom FPR level (report TPR@FPR=0.1%)
ev = Evaluator(target_fpr=0.001)
```

`Evaluator.evaluate(detector, ...)` supports **three modes** depending on
which parameters you pass:

| Mode | Pass | Returns | Use case |
|---|---|---|---|
| **1. Train-only** | `train_texts`, `train_labels`, `method` | `(threshold, metrics)` | Calibrate threshold, inspect train performance |
| **2. Test-only** | `test_texts`, `test_labels`, `threshold` | `metrics` | Evaluate with a known/fixed threshold |
| **3. Train + Test** | both train & test data, `method` | `(threshold, metrics)` | Calibrate on train, evaluate on test (recommended for statistical detectors) |

> **When to use which mode?**
>
> - **Statistical / zero-shot detectors** (DetectGPT, Binoculars, LRR, …):
>   scores are continuous and the optimal boundary is data-dependent →
>   use **Mode 3** (calibrate on train, evaluate on test).
> - **Supervised detectors** (RADAR, RoBERTa, …): already trained to output
>   probabilities around 0.5 → **Mode 2** with `threshold=0.5` is usually fine.
> - **Watermark detectors** (KGW, SynthID): fixed z-score thresholds from
>   the paper → **Mode 2** with the paper's threshold.
> - **Quick sanity check** on a single split → **Mode 1** to see both the
>   calibrated threshold and train metrics.

#### Mode 1: Train-only (calibrate)

```python
from opendetectrl import load_detector, load_dataset
from opendetectrl.evaluation import Evaluator

ds = load_dataset("cheat", subsets=["generation"])
texts, labels = ds.load_texts_labels(max_samples=500)

detector = load_detector("detect_gpt", model_name="gpt2-medium", device="cuda")
evaluator = Evaluator()

# Calibrate threshold on the full dataset, inspect train performance
threshold, train_metrics = evaluator.evaluate(
    detector,
    train_texts=texts, train_labels=labels,
    method="youden",  # or "f1", "eer"
)
print(f"Calibrated threshold: {threshold:.4f}")
print(f"Train metrics: {train_metrics}")
```

#### Mode 2: Test-only (fixed threshold)

```python
# Evaluate with a pre-determined threshold (e.g. from a paper or prior run)
test_metrics = evaluator.evaluate(
    detector,
    test_texts=test_texts, test_labels=test_labels,
    threshold=threshold,  # use the threshold from Mode 1 or a paper default
)
print(f"Test metrics: {test_metrics}")
```

#### Mode 3: Train + Test (calibrate then evaluate) — recommended

```python
from opendetectrl.evaluation import Evaluator, split_texts_labels

# 1. Load CHEAT and split into train/test (CHEAT has no official split)
ds = load_dataset("cheat", subsets=["generation"])
texts, labels = ds.load_texts_labels(max_samples=500)
(train_texts, train_labels), (test_texts, test_labels) = split_texts_labels(
    texts, labels, test_size=0.5
)

# 2. Load a statistical detector
detector = load_detector("detect_gpt", model_name="gpt2-medium", device="cuda")

# 3. Calibrate on train, evaluate on test — one call
evaluator = Evaluator()
threshold, test_metrics = evaluator.evaluate(
    detector,
    train_texts=train_texts, train_labels=train_labels,
    test_texts=test_texts, test_labels=test_labels,
    method="youden",  # or "f1", "eer"
)
print(f"Calibrated threshold: {threshold:.4f}")
print(f"Test metrics: {test_metrics}")
# {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.88, 'f1': 0.85,
#  'roc_auc': 0.91, 'avg_precision': 0.90}
```

---

## Detector Reference

### Statistical / Zero-shot Detectors

#### `log_likelihood` — Log-Likelihood

Sum of log-probabilities of the text under a source LM. Lower likelihood → more AI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str \| None` | `"gpt2"` | HF model name (ignored if `model` is given). |

#### `rank` — Rank

Average token rank under a source LM. Lower rank → more AI.

Same parameters as `log_likelihood`.

#### `log_rank` — Log-Rank

Average log-rank of tokens under a source LM. Lower log-rank → more AI.

Same parameters as `log_likelihood`.

#### `entropy` — Entropy

Average entropy of the next-token distribution. Higher entropy → more AI.

Same parameters as `log_likelihood`.

#### `fast_detect_gpt` — Fast-DetectGPT

Sampling-based discrepancy between the scoring model and a perturbation model, computed analytically (no T5 needed).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scoring_model` | `HFSourceModel \| None` | `None` | Pre-loaded scoring model. |
| `scoring_model_name` | `str` | `"gpt2"` | HF model name for scoring. |
| `sampling_model` | `HFSourceModel \| None` | `None` | Pre-loaded sampling model. |
| `sampling_model_name` | `str \| None` | `None` | Defaults to `scoring_model_name`. |
| `discrepancy_analytic` | `bool` | `True` | Use the analytic discrepancy (no sampling). |
| `model` | alias | — | Alias for `scoring_model`. |

#### `detect_gpt` — DetectGPT

Perturbation-based detection using T5 mask-filling. Measures curvature of log-probability around the original text.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2-medium"` | HF model name. |
| `mask_filling_model_name` | `str` | `"t5-large"` | T5 model for mask filling. |
| `n_perturbations` | `int` | `10` | Number of perturbed copies per text. |
| `n_perturbation_rounds` | `int` | `1` | Number of perturbation rounds. |
| `span_length` | `int` | `2` | Length of each masked span (in words). |
| `pct_words_masked` | `float` | `0.3` | Fraction of words to mask. |
| `buffer_size` | `int` | `1` | Buffer tokens between masked spans. |
| `mask_top_p` | `float` | `1.0` | Top-p for T5 fill sampling. |
| `chunk_size` | `int` | `20` | Chunk size for batch processing. |
| `criterion` | `str` | `"z"` | `"d"` = raw discrepancy, `"z"` = normalized. |
| `random_fills` | `bool` | `False` | Use random word fills (skip T5). |
| `random_fills_tokens` | `bool` | `False` | Use random token fills (skip T5). |
| `perturber` | `MaskFillPerturber \| None` | `None` | Pre-loaded perturber. |
| `cache_dir` | `str \| None` | `None` | Cache dir for T5 download. |

#### `ada_detect_gpt` — AdaDetectGPT

Extends DetectGPT with conformal prediction. Requires calibration scores for p-value computation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.1` | Significance level for conformal p-value. |
| `calibration_scores` | `np.ndarray \| None` | `None` | Pre-computed calibration scores. |
| *(plus all `detect_gpt` parameters)* | | | |

#### `lrr` — LRR (Log-Rank Ratio)

Perturbation-based log-rank ratio. No own parameters beyond the base.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |

#### `npr` — NPR (Normalized Perturbation Rank)

Mean perturbed log-rank divided by original log-rank.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `mask_filling_model_name` | `str` | `"t5-small"` | T5 model for mask filling. |
| `n_perturbations` | `int` | `5` | Number of perturbed copies. |
| `span_length` | `int` | `2` | Mask span length (words). |
| `pct_words_masked` | `float` | `0.3` | Fraction of words masked. |
| `buffer_size` | `int` | `1` | Buffer tokens between spans. |
| `mask_top_p` | `float` | `1.0` | Top-p for T5 fill sampling. |
| `chunk_size` | `int` | `20` | Chunk size for batching. |
| `perturber` | `MaskFillPerturber \| None` | `None` | Pre-loaded perturber. |
| `random_fills` | `bool` | `False` | Use random word fills (skip T5). |
| `random_fills_tokens` | `bool` | `False` | Use random token fills (skip T5). |
| `cache_dir` | `str \| None` | `None` | Cache dir for T5 download. |

#### `binoculars` — Binoculars

Dual-model (observer + performer) perplexity + cross-entropy. Score is negated so higher = more AI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `observer_name_or_path` | `str` | `"tiiuae/falcon-7b"` | Observer model name. |
| `performer_name_or_path` | `str` | `"tiiuae/falcon-7b-instruct"` | Performer model name. |
| `observer_model` | `AutoModelForCausalLM \| None` | `None` | Pre-loaded observer model. |
| `performer_model` | `AutoModelForCausalLM \| None` | `None` | Pre-loaded performer model. |
| `use_bfloat16` | `bool` | `True` | Use bfloat16 precision. |
| `max_token_observed` | `int` | `512` | Max tokens per text. |
| `mode` | `str` | `"low-fpr"` | Threshold mode: `"low-fpr"` or `"high-accuracy"`. |

#### `dna_gpt` — DNA-GPT

N-gram overlap analysis: truncate → regenerate → compare with source model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name for regeneration. |
| `truncate_ratio` | `float` | `0.5` | Fraction of text to keep before regenerating. |
| `max_new_tokens` | `int` | `300` | Max new tokens to generate. |
| `n_perturbations` | `int` | `1` | Number of regenerations. |
| `temperature` | `float` | `0.7` | Sampling temperature. |

#### `ghostbuster` — Ghostbuster

Spectral features from multi-model log-probability standard deviation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `models` | `list[HFSourceModel] \| None` | `None` | Pre-loaded source models. |
| `model_names` | `list[str] \| None` | `None` | Defaults to `["gpt2", "gpt2-medium", "gpt2-large"]`. |

#### `gecscore` — GECScore

Grammatical-error-correction via LLM + ROUGE-2 F-score. Score is negated (higher = more AI).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded GEC model. |
| `model_name` | `str` | `"gpt2"` | HF model name for GEC. |
| `gec_prompt` | `str \| None` | `None` | Custom GEC prompt template. |
| `use_openai` | `bool` | `False` | Use OpenAI API instead of HF. |
| `openai_model` | `str` | `"gpt-4o-mini"` | OpenAI model name. |
| `openai_api_key` | `str \| None` | `None` | OpenAI API key. |

#### `biscope` — BiScope

Forward/backward cross-entropy loss features across 9 text segments.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `sample_clip` | `int` | `2000` | Max tokens per text. |
| `use_summary` | `bool` | `False` | Use summary model for long texts. |
| `summary_model` | `* \| None` | `None` | Pre-loaded summary model. |
| `summary_tokenizer` | `* \| None` | `None` | Tokenizer for summary model. |

#### `repreguard` — RepreGuard

PCA-based representation reading on hidden states.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `rep_token` | `int` | `-1` | Token position for representation extraction. |
| `n_difference` | `int` | `1` | Number of difference layers. |
| `direction_method` | `str` | `"pca"` | Direction extraction method. |
| `n_train` | `int` | `128` | Number of training samples for PCA. |
| `train_data` | `list \| None` | `None` | Pre-loaded training texts. |
| `ai_weight` | `float` | `1.0` | Weight for AI direction. |
| `human_weight` | `float` | `1.0` | Weight for human direction. |

#### `seqxgpt` — SeqXGPT

Log-probability features (entropy, top-k) from source model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `model_path` | `str \| None` | `None` | Local model path (overrides `model_name`). |
| `max_length` | `int` | `512` | Max token length. |

#### `gpt_who` — GPT-who

Uniform Information Density (UID) variance. Score = negated UID variance (lower variance = more uniform = more AI).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2-xl"` | HF model name (official default). |

#### `llmdet` — LLMDet

Max mean log-likelihood across multiple reference LLMs. Higher = more AI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference_models` | `list[HFSourceModel] \| None` | `None` | Pre-loaded reference models. |
| `reference_model_names` | `list[str] \| None` | `["gpt2", "facebook/opt-125m"]` | HF model names. |
| `model` | `HFSourceModel \| None` | `None` | Single pre-loaded model (alias). |
| `model_name` | `str \| None` | `None` | Single HF model name (alias). |

#### `gltr` — GLTR

Token-rank based detection. Score = fraction of tokens in the top-k (higher = more AI).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `topk` | `int` | `40` | Number of top predictions considered "expected". |

#### `stylometric` — Stylometric

Pure feature-based detector (no LM). Character/word/sentence/punctuation/function-word statistics → sigmoid score in `[0, 1]`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `baseline` | `dict \| None` | `None` | Override human baseline (mean/std/sign) per feature. |
| `weights` | `dict \| None` | `None` | Override per-feature weights. |
| `temperature` | `float` | `2.0` | Sigmoid sharpness. |

#### `sniffer` — Sniffer

Contrastive multi-model discrepancy features (Pearson/Spearman per model pair). Trained MLP or zero-shot correlation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `models` | `list[HFSourceModel] \| None` | `None` | Pre-loaded models (>= 2). |
| `model_names` | `list[str] \| None` | `["gpt2", "EleutherAI/gpt-neo-125M"]` | HF model names. |
| `checkpoint_path` | `str \| None` | `None` | Trained Sniffer MLP state dict. |
| `class_num` | `int \| None` | `None` | Classes in checkpoint (required with `checkpoint_path`). |
| `max_length` | `int` | `1024` | Token truncation length. |

#### `polish_ratio` — Polish Ratio

Perplexity-based (ChatGPT-polished text has lower perplexity). Score = negated perplexity.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded scoring model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |

#### `text_fluoroscopy` — Text Fluoroscopy

Layer-wise KL divergence to find the "max-KL layer"; score = negated mean entropy at that layer.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `HFSourceModel \| None` | `None` | Pre-loaded source model. |
| `model_name` | `str` | `"gpt2"` | HF model name (official: `gte-Qwen1.5-7B-instruct`). |
| `max_length` | `int` | `512` | Max token length. |
| `layer_selection` | `str \| int` | `"max_kl"` | `"max_kl"` or a fixed hidden-state layer index. |

---

### Supervised / Neural Detectors

All supervised detectors share these common parameters from their base classes:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str \| None` | class default | HF model name (each detector has a `DEFAULT_MODEL_NAME`). |
| `max_length` | `int` | `512` | Max token length. |
| `batch_size` | `int` | `16` | Batch size for inference. |

#### `roberta_openai` — RoBERTa-base OpenAI Detector

| Default model | `roberta-base-openai-detector` |
|---|---|
| AI class index | 0 |

#### `roberta_large_openai` — RoBERTa-large OpenAI Detector

| Default model | `roberta-large-openai-detector` |
|---|---|
| AI class index | 0 |

#### `deberta_classifier` — DeBERTa Classifier

| Default model | `microsoft/deberta-v3-base` |
|---|---|

#### `hf_classifier` — Generic HF Classifier

| Default model | `roberta-base` |
|---|---|

Accepts an `ai_class_index` parameter to specify which output class is "AI".

#### `grover` — GROVER

| Default model | `rowanz/grover-base` |
|---|---|
| AI class index | 0 |

#### `damage` — DAMAGE

| Default model | `facebook/xlm-roberta-base` |
|---|---|
| AI class index | 1 |

#### `radar` — RADAR

| Default model | `TrustSafeAI/RADAR-Vicuna-7B` |
|---|---|
| AI class index | 0 |

#### `radar_mix` — RADAR-Mix

| Default model | `TrustSafeAI/RADAR-Mix` |
|---|---|
| AI class index | 0 |

#### `detective` — DeTeCtive

SimCSE-RoBERTa embedding + KNN index.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"princeton-nlp/unsup-simcse-roberta-base"` | Embedding model. |
| `model_path` | `str \| None` | `None` | Local checkpoint path. |
| `database_path` | `str \| None` | `None` | Path to KNN database. |
| `embedding_dim` | `int` | `768` | Embedding dimension. |
| `K` | `int` | `5` | Number of nearest neighbors. |
| `max_length` | `int` | `512` | Max token length. |

#### `remodetect` — ReMoDetect

Retrieval-augmented RoBERTa detector.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `retrieval_index` | `* \| None` | `None` | Pre-built retrieval index. |
| `use_retrieval` | `bool` | `False` | Enable retrieval augmentation. |
| *(plus common supervised params)* | | | |
| Default model | `roberta-base` |
| AI class index | 1 |

#### `poger` — POGER

Loss features + HF classifier.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detect_model` | `HFSourceModel \| None` | `None` | Pre-loaded detection model. |
| `detect_model_name` | `str` | `"gpt2"` | HF model name for loss features. |
| `n_feat` | `int` | `7` | Number of loss features. |
| `k` | `int` | `10` | Top-k for feature extraction. |
| `sample_clip` | `int` | `2000` | Max tokens per text. |
| *(plus common supervised params)* | | | |
| Default model | `roberta-base` |
| AI class index | 1 |

#### `mpu` — MPU (Multiscale Positive-Unlabeled)

| Default model | `yuchuantian/AIGC_detector_env1` |
|---|---|
| AI class index | 1 |

#### `deepfake_wild` — Deepfake Text in the Wild

| Default model | `nealcly/detection-longformer` |
|---|---|
| AI class index | 0 |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preprocess` | `bool` | `True` | Apply official text normalization before scoring. |

#### `ood_detection` — OOD Detection

SimCSE-RoBERTa encoder + distance to a human centroid. Requires `fit(reference_texts)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"princeton-nlp/unsup-simcse-roberta-base"` | Encoder model. |

#### `outfox` — OUTFOX

In-context learning few-shot LLM-as-judge. Score = P("ai")/(P("ai")+P("human")).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | HF causal LM (HF path). |
| `use_openai` | `bool` | `False` | Use OpenAI API instead of HF. |
| `openai_model` | `str` | `"gpt-4o-mini"` | OpenAI model name. |
| `examples` | `list \| None` | `None` | Few-shot human/AI examples. |

#### `raidar` — Raidar

Rewrite + word-level edit distance. Score = `-edit_ratio`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | HF causal LM (HF path). |
| `use_openai` | `bool` | `False` | Use OpenAI API instead of HF. |
| `max_new_tokens` | `int` | `200` | Max rewrite tokens. |

#### `coco` — CoCo

| Default model | `YichenZW/Coh-MGT-Detection` |
|---|---|
| AI class index | 1 |

#### `conda` — ConDA

| Default model | `AmritaBh/ConDA-RoBERTa-large-AI_Human-Detection` |
|---|---|
| AI class index | 1 |

#### `tweepfake` — TweepFake

| Default model | `tizfa/tweepfake_bert` (or a fine-tuned checkpoint via `model_name`) |
|---|---|
| AI class index | 1 |

#### `gpt_paternity_test` — GPT Paternity Test

KNN-embedding detector. Requires `fit(reference_texts, reference_is_ai)` first.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | Reference embedding model. |
| `layer` | `int` | `-1` | Hidden-state layer index. |
| `k` | `int` | `5` | Number of nearest neighbours. |

#### `ship_of_theseus` — Ship of Theseus

Paraphrase + word-level edit distance. Score = `-edit_ratio`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | HF causal LM for paraphrasing. |
| `max_new_tokens` | `int` | `200` | Max paraphrase tokens. |

#### `checkgpt` — CheckGPT

Self-consistency proxy (negative mean log-likelihood under a reference LM).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | Reference scoring LM. |

#### `drgd` — DRGD

Disentangled-representation detector (BERT + information bottleneck + content/domain disentanglement). Loads a trained checkpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | *(required)* | Trained DRGD state dict (`.pth`). |
| `pretrained_model` | `str` | `"bert-base-uncased"` | BERT backbone. |
| `ai_class_index` | `int` | `0` | AI class index (official: `0`). |

#### `notai` — NOTAI.AI

17-feature meta-classifier (curvature + stylometric + readability). Uses a trained classifier if `model_path` is given, else a zero-shot combination.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scoring_model` | `HFSourceModel \| None` | `None` | Reference model for curvature features. |
| `scoring_model_name` | `str` | `"gpt2"` | HF model name for curvature. |
| `model_path` | `str \| None` | `None` | Optional trained meta-classifier (e.g. XGBoost). |

---

### Watermark Detectors

#### `kgw_watermark` — KGW Watermark Detection

Z-score detection over green-list token counts (Kirchenbauer et al.).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `AutoModelForCausalLM \| None` | `None` | Pre-loaded model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `gamma` | `float` | `0.5` | Green-list fraction. |
| `delta` | `float` | `2.0` | Green-list bias. |
| `hash_key` | `int` | `15485863` | Hash key for seeding. |
| `seeding_scheme` | `str` | `"simple_1"` | Watermark seeding scheme. |
| `z_threshold` | `float` | `4.0` | Detection z-score threshold. |
| `ignore_repeated_bigrams` | `bool` | `True` | Ignore repeated bigrams. |

#### `unigram_watermark` — Unigram Watermark Detection

Fixed-partition z-score detection (single-token watermark).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `AutoModelForCausalLM \| None` | `None` | Pre-loaded model. |
| `model_name` | `str` | `"gpt2"` | HF model name. |
| `gamma` | `float` | `0.5` | Green-list fraction. |
| `hash_key` | `int` | `15485863` | Hash key. |
| `z_threshold` | `float` | `4.0` | Detection threshold. |

#### `synthid_text` — SynthID-Text Detection

Bayesian detection using HF transformers >= 4.46 built-in SynthID.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `AutoModelForCausalLM \| None` | `None` | Pre-loaded model. |
| `model_name` | `str` | `"gemma-2b"` | HF model name. |
| `watermark_config` | `* \| None` | `None` | Pre-built watermark config. |
| `key` | `int` | `0` | Watermark key (in `**kwargs`). |
| `ngram_len` | `int` | `5` | N-gram length (in `**kwargs`). |

#### `private_watermark` — Private Watermark

Verification step of the private watermarking scheme (Wu et al., ICLR 2024).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkpoint_path` | `str` | *(required)* | Pre-trained watermark detector checkpoint. |
| `ai_class_index` | `int` | `1` | Index of the watermarked class. |

#### `cater` — CATER

Conditional watermark detector. Checks whether the secret trigger phrase appears.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trigger_phrase` | `str` | *(required)* | Secret trigger phrase used at generation time. |

#### `drw` — DRW

Distillation-resistant watermark detector. Checks for secret trigger phrases.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trigger_set` | `list[str]` | *(required)* | Secret trigger phrases used at watermarking time. |

---

### Black-box / LLM-as-Judge Detectors

#### `chatgpt_as_detector` — ChatGPT as Detector

Prompt an LLM to judge whether a text is human or AI. Score = P("ai")/(P("ai")+P("human")).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | HF causal LM (HF path). |
| `use_openai` | `bool` | `False` | Use OpenAI API instead of HF. |
| `openai_model` | `str` | `"gpt-4o-mini"` | OpenAI model name. |

#### `beat_llms` — Beat LLMs at Their Own Game

Query the LLM multiple times and measure the divergence of continuations. Score = `-divergence` (lower variance = more AI).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"gpt2"` | HF causal LM (HF path). |
| `num_queries` | `int` | `5` | Number of sampled continuations. |
| `temperature` | `float` | `1.0` | Sampling temperature. |
| `max_new_tokens` | `int` | `50` | Max continuation tokens. |
| `use_openai` | `bool` | `False` | Use OpenAI API instead of HF. |
| `openai_model` | `str` | `"gpt-4o-mini"` | OpenAI model name. |

---

## End-to-End Example: DetectGPT on CHEAT

```python
from opendetectrl import load_detector, load_dataset
from opendetectrl.evaluation import Evaluator, split_texts_labels

# 1. Load CHEAT dataset and split into train/test (CHEAT has no official split)
ds = load_dataset("cheat", subsets=["generation"])
texts, labels = ds.load_texts_labels(max_samples=200)
(train_texts, train_labels), (test_texts, test_labels) = split_texts_labels(
    texts, labels, test_size=0.5
)
print(f"Train: {len(train_texts)}  Test: {len(test_texts)}")

# 2. Load a statistical detector
detector = load_detector(
    "detect_gpt",
    model_name="gpt2-medium",       # source LM
    mask_filling_model_name="t5-large",  # T5 for perturbation
    n_perturbations=10,
    criterion="z",
    device="cuda",
)

# 3. Mode 3: calibrate threshold on train, evaluate on test
evaluator = Evaluator()
threshold, metrics = evaluator.evaluate(
    detector,
    train_texts=train_texts, train_labels=train_labels,
    test_texts=test_texts, test_labels=test_labels,
    method="youden",
)
print(f"Calibrated threshold: {threshold:.4f}")
print(f"Test metrics: {metrics}")
# {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.88, 'f1': 0.85,
#  'roc_auc': 0.91, 'avg_precision': 0.90}

# 4. Or get per-text predictions
results = detector.predict(test_texts[:5])
for text, result in zip(test_texts[:5], results):
    print(f"label={result.label}  score={result.score:.4f}  text={text[:60]}...")
```

### Example: Multiple Detectors Comparison

```python
from opendetectrl import load_detector, load_dataset
from opendetectrl.evaluation import Evaluator, split_texts_labels

# 1. Load CHEAT and split
ds = load_dataset("cheat", subsets=["generation"])
texts, labels = ds.load_texts_labels(max_samples=500)
(train_texts, train_labels), (test_texts, test_labels) = split_texts_labels(
    texts, labels, test_size=0.5
)

# 2. Compare statistical detectors (Mode 3: calibrate on train, evaluate on test)
evaluator = Evaluator()
stat_detectors = {
    "log_likelihood": {"model_name": "gpt2"},
    "log_rank":       {"model_name": "gpt2"},
    "entropy":        {"model_name": "gpt2"},
    "fast_detect_gpt": {"scoring_model_name": "gpt2"},
    "detect_gpt":     {"model_name": "gpt2-medium", "random_fills_tokens": True},
}

print("=== Statistical detectors (Mode 3: train→calibrate, test→evaluate) ===")
for name, kwargs in stat_detectors.items():
    detector = load_detector(name, **kwargs)
    threshold, metrics = evaluator.evaluate(
        detector,
        train_texts=train_texts, train_labels=train_labels,
        test_texts=test_texts, test_labels=test_labels,
        method="youden",
    )
    print(f"{name:20s}  threshold={threshold:.4f}  "
          f"AUROC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

# 3. Compare supervised detectors (Mode 2: fixed threshold=0.5)
supervised_detectors = {
    "roberta_openai":      {},
    "roberta_large_openai": {},
    "radar":               {},
}

print("\n=== Supervised detectors (Mode 2: fixed threshold=0.5) ===")
for name, kwargs in supervised_detectors.items():
    detector = load_detector(name, **kwargs)
    metrics = evaluator.evaluate(
        detector,
        test_texts=test_texts, test_labels=test_labels,
        threshold=0.5,
    )
    print(f"{name:20s}  AUROC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")
```
