# OpenDetectRL

An open-source toolkit for **LLM-generated text detection**, built around the [DetectRL](https://github.com/junchaoIU/DetectRL) benchmark (NeurIPS 2024 Datasets & Benchmarks Track).

**Positioning**: **Detection + Attack + Adversarial Evaluation**.
Unlike existing toolkits that mainly reproduce detectors, OpenDetectRL makes **robustness / adversarial evaluation a first-class citizen** — evaluate any detector against a battery of attacks via a **detector × attack matrix**.

> Development roadmap: see [plan.md](plan.md). This checklist tracks implementation progress — tick an item once it is done.

---

## Roadmap Checklist

### Core Framework
- [x] Registry pattern (`load_detector` / `load_attack` factory)
- [x] `DetectionResult` unified output (score / label / confidence / metadata)
- [x] `BaseDetector` interface (`score(texts) -> ndarray`)
- [x] `SourceModel` abstraction (HF models / OpenAI API)
- [x] `HFSourceModel` implementation (tokenizer / logits / log-likelihood)
- [ ] Model hub (auto-download & cache)
- [x] `BaseAttacker` interface (`attack(text) -> text`)
- [x] `Evaluator` (AUROC / AUPRC / Accuracy / F1 / EER)
- [ ] **Detector × Attack matrix evaluation** (core differentiator)
- [ ] Batch inference & device management

### Detectors — Zero-shot / Statistical
- [x] Perplexity / Log-Likelihood / Entropy / Log-Rank
- [x] DetectGPT
- [x] Fast-DetectGPT
- [x] NPR / LRR
- [x] Binoculars
- [x] DNA-GPT
- [x] Ghostbuster
- [x] GECScore
- [x] BiScope
- [x] RepreGuard
- [x] POGER
- [x] AdaDetectGPT
- [x] SeqXGPT
- [x] GPT-who
- [x] LLMDet
- [x] GLTR
- [x] Stylometric
- [x] Sniffer
- [x] Polish Ratio
- [x] Text Fluoroscopy

### Detectors — Supervised / Neural
- [x] RoBERTa-base / RoBERTa-large (OpenAI detector)
- [x] DeBERTa classifier
- [x] Grover
- [x] RADAR
- [x] ReMoDetect
- [x] DeTeCtive
- [x] DAMAGE
- [x] Generic HF classifier (`hf_classifier`)
- [x] MPU
- [x] Deepfake Text in the Wild
- [x] OOD Detection
- [x] OUTFOX
- [x] Raidar
- [x] CoCo
- [x] ConDA
- [x] TweepFake
- [x] GPT Paternity Test
- [x] Ship of Theseus
- [x] CheckGPT
- [x] DRGD
- [x] NOTAI.AI

### Detectors — Watermark
- [x] KGW watermark detection
- [x] Unigram watermark detection
- [x] SynthID-Text detection
- [x] Private Watermark
- [x] CATER
- [x] DRW

### Detectors — Black-box / LLM-as-Judge
- [x] ChatGPT as Detector (Fighting Fire with Fire)
- [x] Beat LLMs at Their Own Game

### Attackers
- [ ] Paraphrase attack (DIPPER)
- [ ] Back-translation
- [ ] Perturbation (typo / homoglyph / insert-delete)
- [ ] Prompt attack
- [ ] Style humanization (MASH)
- [ ] Watermark removal (DE-MARK / Vaporizer)

### Benchmarks / Datasets
- [ ] HC3
- [x] CHEAT
- [ ] M4
- [ ] DetectRL / DetectRL-X
- [ ] RAID
- [ ] M4GT-Bench
- [ ] MGTBench
- [ ] TuringBench
- [ ] MULTITuDE
- [ ] GPABenchmark

---

## MVP Path

Recommended order to reach a working "adversarial evaluation" demo:

1. Core Framework (registry + `BaseDetector` + `DetectionResult`)
2. A few high-value zero-shot detectors (DetectGPT, Fast-DetectGPT, Binoculars)
3. `BaseAttacker` + paraphrase & perturbation attacks
4. Detector × Attack matrix `Evaluator`

## Related

- Paper list & taxonomy: [Awesome LLM-generated Text Detection](../README.md) (parent repo)
- Benchmark: [DetectRL](https://github.com/junchaoIU/DetectRL)
