# AIGC Experiments

## [语音转换 (Voice Conversion, VC)](/TooYoungTooSimp/vc_playground)

- **目标**: 一个轻量的语音转换(Voice Conversion, VC)实验流程。以内容特征(Content)与基频(F0)为条件，结合说话人向量进行声学建模，预测对齐的对数 Mel 频谱，再用声码器合成音频。
- **阶段**: 
  - Step 1: 数据准备与特征抽取（见 `explore_step1.ipynb`）。
  - Step 2: 说话人表征学习（XVector，见 `explore_step2.ipynb` / `explore_step2.py`）。
  - Step 3: 内容+F0 → Mel 转换模型（Conv/Attention 两种实现，见 `explore_step3.ipynb`）与评估（`explore_step3_eval.ipynb`）。

**目录说明**
- `tools.py`: 常用工具
  - `to_logmelspec(waveform, sr)`: 采样率 16k，`n_fft=1024, hop=256, n_mels=80`，输出 log10-Mel（时间在最后一维）。
  - `extract_f0_torchcrepe(wav_16k, sr=16000, hop=256)`: 用 `torchcrepe` 估计 F0，并转换为 MIDI 标度；返回形如 `(T,)` 的序列。
- `model_XVector.py`: 基于 Transformer-Encoder 的 XVector 说话人分类/嵌入模型，支持长度掩码与统计池化，输出 `(logits, embedding)`。
- `model_Content2MelConv.py`: 卷积残差堆叠 + FiLM 条件化（以说话人向量调制），输入 `content(=H)` + `f0` 预测 Mel。
- `model_Content2MelAttn.py`: 自注意力堆叠 + 每层 FiLM 条件化，带位置编码与 LayerNorm，输入同上。
- `models.py`: 简单聚合导出（导入 `Content2MelAttn`、`Content2Mel`、`XVector`）。
- `explore_step1.ipynb`: 数据集构建/特征抽取探索。
- `explore_step2.ipynb`、`explore_step2.py`: XVector 训练与验证，保存权重 `xvector_easy.pth`。
- `explore_step3.ipynb`、`explore_step3_eval.ipynb`: Content2Mel 训练与评估/可视化。

**环境依赖**
- 必需: `torch`, `torchaudio`, `numpy`, `tqdm`, `scikit-learn`, `transformers`, `torchcrepe`, `IPython`(notebook 中音频展示)。
- 说明: 
  - 采样率统一为 `16 kHz`，Mel 配置与训练/推理保持一致。
  - 需要 CUDA/GPU 可显著加速（`torchcrepe` 和 Transformer 训练）。
  - 可选声码器: HuggingFace `microsoft/speecht5_hifigan`（示例在笔记本中）。

安装示例（Windows PowerShell）
- 创建环境并安装依赖（根据你本地 CUDA 版本选择正确的 torch/torchaudio 轮子）。
  - `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
  - `pip install numpy tqdm scikit-learn transformers torchcrepe ipywidgets`

**数据与特征**
- 音频: 单声道 16kHz wav。
- Mel 频谱: 由 `tools.to_logmelspec` 计算，配置固定（`n_fft=1024, hop=256, n_mels=80`，log10-Mel）。
- 基频 F0: 由 `tools.extract_f0_torchcrepe` 计算（输出 MIDI 标度），时间对齐至 Mel 帧。
- 说话人标签与向量:
  - 训练 XVector 时，输入 `(logmel, label)`；验证后可对同一说话人的嵌入做均值，得到 256 维说话人质心向量。
  - 预测 Mel 时，使用目标说话人的嵌入（或质心）。

特征抽取示例
- 读取波形并转 Log-Mel：
  - `waveform, sr = torchaudio.load(path)`
  - `logmel = tools.to_logmelspec(waveform, sr)`  → 形如 `(B, T, 80)` 或 `(T, 80)`
- F0 (MIDI)：
  - `wav16k = torchaudio.functional.resample(waveform, sr, 16000)`
  - `f0_midi = tools.extract_f0_torchcrepe(wav16k.squeeze(0))`

**Step 2: 训练 XVector（说话人嵌入）**
- 脚本: `02-aigc/vc/explore_step2.py`
- 期望输入: 当前脚本默认从 `dataset_slim.pt` 加载数据（`torch.load`），并假设：
  - 存在 `dataset.label` 字段（可用于构建标签词表）。
  - DataLoader 迭代返回 `(logmel_tensor, label_str)`，内部会做 `pad_sequence` 与长度对齐。
- 运行（仓库根目录）：
  - `python .\02-aigc\vc\explore_step2.py`
- 训练细节: AdamW、Cosine LR、交叉熵分类；自动计算验证集准确率；保存权重到 `xvector_easy.pth`。

最小用法（推理获取嵌入）
- `from vc.model_XVector import XVector`
- `model = XVector(num_classes=N).to(device); model.load_state_dict(torch.load('xvector_easy.pth'))`
- `logits, emb = model(logmel_batch, lengths)` → `emb` 形如 `(B, 256)`

**Step 3: 内容+F0 → Mel（两种模型）**
- 卷积版 `Content2Mel`（`model_Content2MelConv.py`）
  - 输入: `H:(B,T,content_dim)`、`f0:(B,T)`、`spk:(B,256)`；默认 `content_dim=1024`。
  - 结构: 线性投影 → FiLM 条件化 → 多层扩张卷积残差块 → 线性到 `n_mels=80`。
- 注意力版 `Content2MelAttn`（`model_Content2MelAttn.py`）
  - 输入同上，支持长度掩码；每层含自注意力 + FFN，并在两处子层使用 FiLM 以说话人向量调制；带正弦位置编码与 LayerNorm。

最小用法（前向生成 Mel）
- `from vc.model_Content2MelAttn import Content2MelAttn`
- `net = Content2MelAttn(content_dim=1024, spk_dim=256, n_mels=80).to(device)`
- `mel_pred = net(H, f0, spk, lengths)` → 形如 `(B, T, 80)` 的 log10-Mel

**声码器合成**
- 示例使用 HuggingFace `microsoft/speecht5_hifigan` 作为 Mel 声码器（在 `explore_step2.py` 有加载演示）。
- 建议参考笔记本 Step 3 部分的调用示例，确保 Mel 配置与声码器预期一致（采样率、hop、归一化等）。

**评估与可视化**
- `explore_step3_eval.ipynb` 提供客观与主观评估的示例代码（如频谱对比、嵌入可视化、简单指标等）。

**端到端推理流程（示意）**
- 准备：
  - 内容特征 `H:(B,T,1024)`（如来自自监督语音模型的帧级特征，按 hop 对齐到 Mel）。
  - `f0_midi:(B,T)`，与 `H`/Mel 时间对齐。
  - 目标说话人向量 `spk:(B,256)`（来自 XVector 质心或注册语音）。
- 生成：
  - `mel = Content2Mel(或 Content2MelAttn)(H, f0_midi, spk, lengths)`
  - 使用声码器将 `mel` 解码为波形。

**常见注意事项**
- 采样率与 hop 一致性: 训练/推理阶段需严格保持 Mel/F0 的对齐与参数一致。
- 时长与掩码: 传入 `lengths` 构造 `key_padding_mask`，避免填充对注意力与统计池化造成偏差。
- 说话人条件化: FiLM 依赖 `spk` 的尺度/分布；建议用 XVector 训练集统计的均值向量作为质心，或在目标域上做对齐。
- HuggingFace 访问: 如需国内镜像，可设置 `HF_ENDPOINT` 环境变量或参考脚本中的示例。

**后续方向**
- 数据增强与说话人/内容解耦更强的编码器。
- 更强的声码器（如大规模 HiFi-GAN/BigVGAN）与自回归/扩散式声学建模。
- 主观评测流程（MOS/ABX）与更严格的客观指标（ASV/ASR 迁移等）。

**快速问题排查**
- ImportError: 检查 `torch/torchaudio/torchcrepe/transformers` 是否安装版本兼容；确认 CUDA 环境。
- 声码器失真: 检查 Mel 配置、尺度（log10/log-e）、对齐、归一化策略与声码器预期是否匹配。
- 训练不收敛: 减小学习率或梯度裁剪，提高 batch size，检查特征统计与标签数据质量。

**参考运行入口**
- `python explore_step2.py` 训练 XVector；更多训练/推理细节见对应笔记本。
