# LLM 生成文本检测库 · 开发计划

> **定位**：`检测 + 攻击 + 对抗评估` 三位一体的生产级 LLM 生成文本检测库
> **差异化**：鲁棒性 / 攻击评估是一等公民（DetectZoo / MGTBench / RAID 均未把"攻击"做成一等公民）
> **竞品对标**：DetectZoo（检测器全，但无攻击模块、研究玩具）→ 我们不拼检测器数量，主打对抗评估 + 生产化

---

## 里程碑总览

- [x] **M0** 项目初始化与工程脚手架
- [x] **M1** 核心抽象层（Detector / SourceModel / Result）
- [x] **M2** 检测器实现（统计类 + 监督类）
- [ ] **M3** 攻击器抽象与攻击套件（核心差异化）
- [ ] **M4** 对抗评估框架（检测器 × 攻击 矩阵）
- [ ] **M5** 数据集与基准接入
- [ ] **M6** 训练管线（微调自己的监督检测器）
- [ ] **M7** 生产化（批处理 / CLI / API 服务）
- [ ] **M8** 文档、测试与开源发布

---

## M0 · 项目初始化与工程脚手架

- [x] 确定库名、包名、仓库地址
- [x] 搭建 `pyproject.toml` + 包结构骨架
- [ ] 配置 CI（lint / type-check / 单元测试）
- [x] 确定依赖策略（torch / transformers / 可选 extras）
- [x] 设计注册器（Registry）模式，支持 `load("name")` 工厂加载

## M1 · 核心抽象层

- [x] `DetectionResult`：统一输出（score / label / confidence / metadata）
- [x] `BaseDetector`：统一 `score(texts) -> ndarray` 接口（输出分数而非标签）
- [x] `SourceModel`：白盒方法统一的源模型抽象（兼容 HF 模型 / OpenAI API）
- [x] `get_logprobs / get_hidden_states` 统一取数接口
- [x] `HFSourceModel`：HuggingFace 源模型实现（tokenizer / logits / log-likelihood）
- [ ] 模型 Hub 管理：自动下载、缓存、本地路径覆盖
- [ ] 设备与精度管理：GPU/CPU、批处理、混合精度

## M2 · 检测器实现

> 原则：先实现高价值、高引用的方法，不追求数量

### 统计类（白盒 / 零样本）
- [x] Perplexity / Log-Likelihood / Entropy / Log-Rank
- [x] DetectGPT
- [x] Fast-DetectGPT
- [x] NPR / LRR
- [x] Binoculars（双模型）
- [x] GECScore（黑盒零样本）
- [x] DNA-GPT（黑盒零样本）
- [x] Ghostbuster（多模型频谱特征）
- [x] BiScope（前向/后向交叉熵损失）
- [x] RepreGuard（隐藏表示模式）
- [x] AdaDetectGPT（自适应扰动 + 统计保证）
- [x] SeqXGPT（对数概率特征 + 分类器）
- [x] GPT-who（信息密度）
- [x] LLMDet（跨参考模型最大对数似然）
- [x] GLTR（top-k 概率分布）
- [x] Stylometric（风格特征）
- [x] Sniffer（模型对差异特征）
- [x] Polish Ratio（基于困惑度）
- [x] Text Fluoroscopy（内部特征）

### 监督类
- [x] RoBERTa-base / RoBERTa-large（OpenAI detector）
- [x] DeBERTa 分类器
- [x] 支持加载任意 HF 分类器（`hf_classifier`）
- [x] RADAR（对抗训练 RoBERTa-large）
- [x] GROVER（判别器）
- [x] DeTeCtive（对比学习 + KNN）
- [x] DAMAGE（多语言多领域）
- [x] ReMoDetect（检索增强）
- [x] POGER（代理引导高效重采样）
- [x] MPU（多尺度正无标记学习）
- [x] Deepfake Wild（长文本分类器）
- [x] OOD Detection（SimCSE 距离）
- [x] OUTFOX（上下文学习 few-shot）
- [x] Raidar（重写 + 编辑距离）
- [x] CoCo（连贯性增强）
- [x] ConDA（对比领域自适应）
- [x] TweepFake（推文检测）
- [x] GPT Paternity Test（KNN 嵌入）
- [x] Ship of Theseus（改写稳定性）
- [x] CheckGPT（自一致性）
- [x] DRGD（解耦表征，跨生成器泛化）
- [x] NOTAI.AI（曲率 + 特征归因，XGBoost 元分类器）

### 水印类
- [x] KGW 水印检测（z-score）
- [x] Unigram 水印检测（固定分区 z-score）
- [x] SynthID-Text 检测（贝叶斯显著性检验）
- [x] Private Watermark（私有水印）
- [x] CATER（条件水印）
- [x] DRW（抗蒸馏水印）

### 黑盒 / LLM-as-Judge 类
- [x] ChatGPT as Detector（Fighting Fire with Fire）
- [x] Beat LLMs at Their Own Game（多次查询发散度）

## M3 · 攻击器抽象与攻击套件（核心差异化）

- [x] `BaseAttacker`：统一 `attack(text) -> text` 接口
- [ ] Paraphrase 攻击（DIPPER / 回译）
- [ ] 扰动攻击（拼写错误 / 同形字 / 随机插入删除）
- [ ] Prompt 攻击（改变生成策略）
- [ ] 风格化 / 人性化改写攻击
- [x] 攻击器注册与 `load_attack("name")` 工厂加载

## M4 · 对抗评估框架

- [x] `Evaluator`：标准指标（AUROC / AUPRC / Accuracy / F1 / EER）
- [ ] **检测器 × 攻击 矩阵评估**（杀手锏：clean vs. attacked 性能对比）
- [ ] 鲁棒性报告：性能下降幅度、可视化
- [ ] 结果导出（JSON / 表格 / 图）
- [ ] 对接 DetectRL / DetectRL-X 攻击场景

## M5 · 数据集与基准接入

- [x] 统一 `BaseDataset` 抽象（自动下载 + 缓存）
- [x] 接入 CHEAT（HC3 / M4 待补）
- [ ] 接入 DetectRL / DetectRL-X
- [ ] 接入 RAID（可选）
- [ ] 支持从本地目录 / CSV 加载自定义数据

## M6 · 训练管线

- [ ] 监督检测器微调脚本（基于 HF Trainer）
- [ ] 训练 / 验证 / 测试拆分与数据加载
- [ ] 训练配置（YAML / dataclass）
- [ ] 训练产物一键注册为可用检测器

## M7 · 生产化

- [ ] 批处理推理 API（`predict_batch`）
- [ ] 命令行工具（CLI）：一行命令检测
- [ ] API 服务封装（FastAPI，可选）
- [ ] 接入商业 / API 检测器（OpenAI、GPTZero，可选）
- [ ] 检测器集成（ensemble，可选）

## M8 · 文档、测试与开源发布

- [ ] README + Quick Start
- [ ] API 文档（每个检测器 / 攻击器）
- [x] 单元测试 + 集成测试（本地纯数学单测 + tiny GPT-2 烟雾测试，23 passed）
- [ ] 复现脚本（对标已发表论文指标）
- [ ] 发布到 PyPI
- [ ] 撰写技术报告 / 论文（可选）

---

## 备注

- **优先级建议**：M0 → M1 → M2（精简版）→ M3 → M4 是最小可行产品（MVP）路径，先把"对抗评估"这个核心卖点跑通
- **中文 / 多语言**：可作为后续特色（现有库基本英文为主）
- **避免**：不要一开始就追求检测器数量，先跑通核心闭环
