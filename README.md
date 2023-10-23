# Awesome LLM-generated Text Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![](https://img.shields.io/badge/PRs-Welcome-red)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/junchaoIU/LLM-generated-Text-Detection/main?logo=github&color=blue)

The powerful ability of large language models (LLMs) to understand, follow, and generate complex languages has enabled LLM-generated texts to flood many areas of our daily lives at an incredible rate, with potentially negative impacts and risks on society and academia. As LLMs continue to expand, how can we detect LLM-generated texts to help minimize the threat posed by the misuse of LLMs?

<div align="center">
  <img src="cover.png" alt="Logo" width="800">
</div>
<br>
<!-- **Authors:** -->

**_¹ [Junchao Wu](https://github.com/junchaoIU), ¹ [Shu Yang](https://twitter.com/shuyhere), ¹ [Runzhe Zhan](https://runzhe.me/), ¹ ² [Yulin Yuan](https://fah.um.edu.mo/yulin-yuan/), ¹ [Derek Fai Wong](https://www.fst.um.edu.mo/personal/derek-wong/), ¹ [Lidia Sam Chao]()_**


<!-- **Affiliations:** -->

¹ University of Macau, ² Peking University

## 📢 News
* [2023.10.24] Our survey paper is now available on Arxiv:
* [2023.10.07] : We start **Chinese LLM-generated Essay Detection** Project.
* [2023.05.01] : We began to explore the topic of LLM-generated Text Detection.

## 📃 Papers
### Overview

We mainly analyzed the work related to LLM-generated Text Detection from the following points, and refer to our paper for more details.

### Datasets
#### Benchmarks
| Corpus                                                                 | Use              | Human            | LLMs         |
|------------------------------------------------------------------------|------------------|------------------|--------------|
| [HC3](https://arxiv.org/abs/2301.07597)                                | train            | 58k              | 26k          |
| [HC3-Chinese](https://arxiv.org/abs/2301.07597)                        | train            | 22k              | 17k          |
| [CHEAT](https://arxiv.org/abs/2304.12008)                              | train            | 15k              | 35k          |
| [GROVER Dataset](https://arxiv.org/abs/1905.12616)                     | train valid test | 5k 2k 8k         | 5k 1k 4k     |
| [TweepFake](https://arxiv.org/abs/2008.00036)                          | train            | 12k              | 12k          |
| [GPT-2 Output Dataset](https://github.com/openai/gpt-2-output-dataset) | train            | 250k             | 250k         |
| [TuringBench](https://aclanthology.org/2021.findings-emnlp.172/)       | train            | 10k              | 190k         |
| [MGTBench](https://arxiv.org/abs/2303.14822)                           | train test       | 2k 563           | 13k 3k       |
| [ArguGPT](https://arxiv.org/pdf/2304.07666.pdf)                                                                | train valid test | 3k 350 350       | 3k 350 350   |
| [DeepfakeText-Detect-Dataset](https://arxiv.org/abs/2210.09421)                                           | train valid test | 319k 56.8k 56.8k | -            |
| [M4](https://arxiv.org/abs/2305.14902)                                                                     | train valid test | 122k 500 500     | 122k 500 500 |
| [GPABenchmark](https://arxiv.org/abs/2306.05524)                                                           | train            | 600k             | 600k         |
| [Scientific-articles Benchmark](https://aclanthology.org/2023.trustnlp-1.17/)                                          | train test       | 8k 4k            | 8k 4k        |

#### Corpus

### Detector
![DETECTOR](image.png)

#### Watermark Technology

| Paper                                                                                                         | Link                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  A watermark for large language models.                                               | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2301.10226)                                                                                                                                |
| On the Reliability of Watermarks for Large Language Models                                               | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2306.04634)                                                                                                                                |
| MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions                                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14795) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/princeton-nlp/MQuAKE)                |
| You can’t pick your neighbors, or can you? When and How to Rely on Retrieval in the kNN-LM                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-emnlp.218/)                                                                                                               |
| Nearest Neighbor Zero-Shot Inference                                                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.214/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/swj0419/kNN_prompt)     |
| Memory-assisted prompt editing to improve GPT-3 after deployment                                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.183/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/madaan/memprompt)       |
| Towards Teachable Reasoning Systems: Using a Dynamic Memory of User Feedback for Continual System Improvement | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.644/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://allenai.org/data/teachme)          |
| Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval                                           | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2201.12431) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/neulab/retomaton)                    |
| Memory-Based Model Editing at Scale                                                                           | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2206.06520) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/eric-mitchell/serac)                 |
| Learning to repair: Repairing model output errors after deployment using a dynamic memory of feedback         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-naacl.26/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/allenai/interscript) |
| Efficient Nearest Neighbor Language Models                                                                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.emnlp-main.461/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/jxhe/efficient-knnlm)   |
| BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief                   | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.emnlp-main.697/)                                                                                                                   |
| Generalization through Memorization: Nearest Neighbor Language Models                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=HklBjCEKvH) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/urvashik/knnlm)            |








## 🚩 Citation

If our research helps you, please kindly cite our paper.

```bibtex

```


## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. We appreciate your contributions to making LLM-generated Text Detection work even better.


