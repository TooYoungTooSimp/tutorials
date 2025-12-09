# AIGC Experiments

## [Reasoning & Tool Call SFT](finetune.ipynb)

目标与概览

-   目标: 让小模型具备“先思考再调用工具”的能力，学习在生成 `<think>...</think>` 规划后再产生结构化 `<tool_call>...</tool_call>` 与工具响应 `<tool_response>...</tool_response>` 的序列。
-   数据集: `Jofthomas/hermes-function-calling-thinking-V1`（HF Datasets）。将原 `conversations` 重命名为 `messages`，并把首条 `system` 内容合并进首条 `user`，在内容中附加“先在 <think></think> 中规划再调用函数”的提示。
-   基座模型: `google/gemma-2-2b-it`。

特殊标记与模板

-   额外特殊标记: `<tools>`, `</tools>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<pad>`, `<eos>`。
-   Tokenizer 设置: 将上述标记注册为 `additional_special_tokens`，并将 `pad_token` 设为 `<pad>`；随后调用 `model.resize_token_embeddings(len(tokenizer))`。
-   Chat 模板: 使用 ChatML 风格模板，形如 `"<start_of_turn>{role}\n{content}<end_of_turn><eos>"`，不接受 `system` 角色（因此在预处理时合并）。

LoRA 与训练配置（来自笔记本）

-   LoRA: `r=16, alpha=64, dropout=0.05`，`target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"]`，`task_type=CAUSAL_LM`。
-   优化/调度: `lr=1e-4`, `warmup_ratio=0.1`, `lr_scheduler_type="cosine"`, `weight_decay=0.1`, `max_grad_norm=1.0`。
-   训练形态: `per_device_train_batch_size=1`, `per_device_eval_batch_size=1`, `gradient_accumulation_steps=4`, `bf16=True`, `gradient_checkpointing=True`（`use_reentrant=False`）, `packing=True`, `max_length=1500`, `num_train_epochs=1`。
-   输出目录: `gemma-2-2B-it-thinking-function_calling-V0`（保存 PEFT/LoRA 适配器）。
-   数据切分: 示例中仅选取 `train[:100]`、`test[:10]` 用于快速演示；如需正式训练，删除这两行限制。

环境与安装

-   依赖: `torch`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`（如需 4/8bit 可另装 `bitsandbytes`）。
-   示例安装:
    -   `pip install transformers datasets trl peft accelerate`
    -   `pip install torch --index-url https://download.pytorch.org/whl/cu130` （按你的 CUDA 版本选择）
-   预训练权重访问: `gemma-2-2b-it` 可能要求许可，需先 `huggingface-cli login` 并在网页同意条款。

运行方式

-   打开并顺序执行 `finetune.ipynb`。关键步骤包括: 加载与预处理数据集 → 构建含特殊标记的 tokenizer → 加载基座模型（`device_map="auto"`, `torch_dtype=bfloat16`）并 `resize_token_embeddings` → 配置 LoRA 与 `SFTConfig` → `SFTTrainer(...).train()` → `trainer.save_model()`。

最小推理用法（加载 LoRA 适配器）

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "google/gemma-2-2b-it"
adapter = "./02-aigc/gemma-2-2B-it-thinking-function_calling-V0"

# 与训练一致的特殊标记与模板
special_tokens = ["<tools>","</tools>","<think>","</think>","<tool_call>","</tool_call>","<tool_response>","</tool_response>","<pad>","<eos>"]
tokenizer = AutoTokenizer.from_pretrained(base, pad_token="<pad>", additional_special_tokens=special_tokens)
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, adapter).eval()

messages = [
    {"role": "user", "content": (
        "<tools>{\n  \"tools\": [{\n    \"name\": \"get_weather\", \"description\": \"Query weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}]\n}</tools>\n"
        "Please tell me if it will rain in Paris today."
    )}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## [语音转换 (Voice Conversion, VC)](https://github.com/TooYoungTooSimp/vc_playground)

-   **目标**: 一个轻量的语音转换(Voice Conversion, VC)实验流程。以内容特征(Content)与基频(F0)为条件，结合说话人向量进行声学建模，预测对齐的对数 Mel 频谱，再用声码器合成音频。
-   **阶段**:
    -   Step 1: 数据准备与特征抽取（见 `explore_step1.ipynb`）。
    -   Step 2: 说话人表征学习（XVector，见 `explore_step2.ipynb` / `explore_step2.py`）。
    -   Step 3: 内容+F0 → Mel 转换模型（Conv/Attention 两种实现，见 `explore_step3.ipynb`）与评估（`explore_step3_eval.ipynb`）。

**目录说明**

-   `tools.py`: 常用工具
    -   `to_logmelspec(waveform, sr)`: 采样率 16k，`n_fft=1024, hop=256, n_mels=80`，输出 log10-Mel（时间在最后一维）。
    -   `extract_f0_torchcrepe(wav_16k, sr=16000, hop=256)`: 用 `torchcrepe` 估计 F0，并转换为 MIDI 标度；返回形如 `(T,)` 的序列。
-   `model_XVector.py`: 基于 Transformer-Encoder 的 XVector 说话人分类/嵌入模型，支持长度掩码与统计池化，输出 `(logits, embedding)`。
-   `model_Content2MelConv.py`: 卷积残差堆叠 + FiLM 条件化（以说话人向量调制），输入 `content(=H)` + `f0` 预测 Mel。
-   `model_Content2MelAttn.py`: 自注意力堆叠 + 每层 FiLM 条件化，带位置编码与 LayerNorm，输入同上。
-   `models.py`: 简单聚合导出（导入 `Content2MelAttn`、`Content2Mel`、`XVector`）。
-   `explore_step1.ipynb`: 数据集构建/特征抽取探索。
-   `explore_step2.ipynb`、`explore_step2.py`: XVector 训练与验证，保存权重 `xvector_easy.pth`。
-   `explore_step3.ipynb`、`explore_step3_eval.ipynb`: Content2Mel 训练与评估/可视化。

**环境依赖**

-   必需: `torch`, `torchaudio`, `numpy`, `tqdm`, `scikit-learn`, `transformers`, `torchcrepe`, `IPython`(notebook 中音频展示)。
-   说明:
    -   采样率统一为 `16 kHz`，Mel 配置与训练/推理保持一致。
    -   需要 CUDA/GPU 可显著加速（`torchcrepe` 和 Transformer 训练）。
    -   可选声码器: HuggingFace `microsoft/speecht5_hifigan`（示例在笔记本中）。

安装示例

-   创建环境并安装依赖（根据你本地 CUDA 版本选择正确的 torch/torchaudio 轮子）。
    -   `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130`
    -   `pip install numpy tqdm scikit-learn transformers torchcrepe ipywidgets`

**数据与特征**

-   音频: 单声道 16kHz wav。
-   Mel 频谱: 由 `tools.to_logmelspec` 计算，配置固定（`n_fft=1024, hop=256, n_mels=80`，log10-Mel）。
-   基频 F0: 由 `tools.extract_f0_torchcrepe` 计算（输出 MIDI 标度），时间对齐至 Mel 帧。
-   说话人标签与向量:
    -   训练 XVector 时，输入 `(logmel, label)`；验证后可对同一说话人的嵌入做均值，得到 256 维说话人质心向量。
    -   预测 Mel 时，使用目标说话人的嵌入（或质心）。

特征抽取示例

-   读取波形并转 Log-Mel：
    -   `waveform, sr = torchaudio.load(path)`
    -   `logmel = tools.to_logmelspec(waveform, sr)` → 形如 `(B, T, 80)` 或 `(T, 80)`
-   F0 (MIDI)：
    -   `wav16k = torchaudio.functional.resample(waveform, sr, 16000)`
    -   `f0_midi = tools.extract_f0_torchcrepe(wav16k.squeeze(0))`

**Step 2: 训练 XVector（说话人嵌入）**

-   脚本: `02-aigc/vc/explore_step2.py`
-   期望输入: 当前脚本默认从 `dataset_slim.pt` 加载数据（`torch.load`），并假设：
    -   存在 `dataset.label` 字段（可用于构建标签词表）。
    -   DataLoader 迭代返回 `(logmel_tensor, label_str)`，内部会做 `pad_sequence` 与长度对齐。
-   运行（仓库根目录）：
    -   `python .\02-aigc\vc\explore_step2.py`
-   训练细节: AdamW、Cosine LR、交叉熵分类；自动计算验证集准确率；保存权重到 `xvector_easy.pth`。

最小用法（推理获取嵌入）

-   `from vc.model_XVector import XVector`
-   `model = XVector(num_classes=N).to(device); model.load_state_dict(torch.load('xvector_easy.pth'))`
-   `logits, emb = model(logmel_batch, lengths)` → `emb` 形如 `(B, 256)`

**Step 3: 内容+F0 → Mel（两种模型）**

-   卷积版 `Content2Mel`（`model_Content2MelConv.py`）
    -   输入: `H:(B,T,content_dim)`、`f0:(B,T)`、`spk:(B,256)`；默认 `content_dim=1024`。
    -   结构: 线性投影 → FiLM 条件化 → 多层扩张卷积残差块 → 线性到 `n_mels=80`。
-   注意力版 `Content2MelAttn`（`model_Content2MelAttn.py`）
    -   输入同上，支持长度掩码；每层含自注意力 + FFN，并在两处子层使用 FiLM 以说话人向量调制；带正弦位置编码与 LayerNorm。

最小用法（前向生成 Mel）

-   `from vc.model_Content2MelAttn import Content2MelAttn`
-   `net = Content2MelAttn(content_dim=1024, spk_dim=256, n_mels=80).to(device)`
-   `mel_pred = net(H, f0, spk, lengths)` → 形如 `(B, T, 80)` 的 log10-Mel

**声码器合成**

-   示例使用 HuggingFace `microsoft/speecht5_hifigan` 作为 Mel 声码器（在 `explore_step2.py` 有加载演示）。
-   建议参考笔记本 Step 3 部分的调用示例，确保 Mel 配置与声码器预期一致（采样率、hop、归一化等）。

**评估与可视化**

-   `explore_step3_eval.ipynb` 提供客观与主观评估的示例代码（如频谱对比、嵌入可视化、简单指标等）。

**端到端推理流程（示意）**

-   准备：
    -   内容特征 `H:(B,T,1024)`（如来自自监督语音模型的帧级特征，按 hop 对齐到 Mel）。
    -   `f0_midi:(B,T)`，与 `H`/Mel 时间对齐。
    -   目标说话人向量 `spk:(B,256)`（来自 XVector 质心或注册语音）。
-   生成：
    -   `mel = Content2Mel(或 Content2MelAttn)(H, f0_midi, spk, lengths)`
    -   使用声码器将 `mel` 解码为波形。

**常见注意事项**

-   采样率与 hop 一致性: 训练/推理阶段需严格保持 Mel/F0 的对齐与参数一致。
-   时长与掩码: 传入 `lengths` 构造 `key_padding_mask`，避免填充对注意力与统计池化造成偏差。
-   说话人条件化: FiLM 依赖 `spk` 的尺度/分布；建议用 XVector 训练集统计的均值向量作为质心，或在目标域上做对齐。
-   HuggingFace 访问: 如需国内镜像，可设置 `HF_ENDPOINT` 环境变量或参考脚本中的示例。

**后续方向**

-   数据增强与说话人/内容解耦更强的编码器。
-   更强的声码器（如大规模 HiFi-GAN/BigVGAN）与自回归/扩散式声学建模。
-   主观评测流程（MOS/ABX）与更严格的客观指标（ASV/ASR 迁移等）。

**快速问题排查**

-   ImportError: 检查 `torch/torchaudio/torchcrepe/transformers` 是否安装版本兼容；确认 CUDA 环境。
-   声码器失真: 检查 Mel 配置、尺度（log10/log-e）、对齐、归一化策略与声码器预期是否匹配。
-   训练不收敛: 减小学习率或梯度裁剪，提高 batch size，检查特征统计与标签数据质量。

**参考运行入口**

-   `python explore_step2.py` 训练 XVector；更多训练/推理细节见对应笔记本。
