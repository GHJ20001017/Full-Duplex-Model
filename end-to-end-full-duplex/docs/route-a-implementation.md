# Route A：按 Moshi 训练流程构建 Qwen3 全双工模型

## 目标与边界

本路线不是“Qwen3 加一个 ASR 适配器，再接一个 TTS”。目标是复现 **Moshi 的训练顺序、数据组织、联合损失、延迟机制和全双工建模方式**，只把原版 Moshi 的文本骨干 Helium 替换为 Qwen3，并且**不重新训练 Mimi 编解码器**。

因此必须遵守以下边界：

- Mimi 使用已经发布的 checkpoint，冻结其 Encoder/Decoder 和量化器；只做加载、流式 encode/decode、码本形状和重建质量验证。
- 不增加独立 ASR、VAD、轮次检测、LLM、TTS 串联路径。用户音频直接作为用户音频流进入模型；助手文本是 Inner Monologue，同时作为助手输出文本流监督。
- 不把“先训练 Qwen3 读音频、再训练一个独立语音生成器”当作 Moshi 训练流程。可以做小规模诊断，但不能替代 Moshi 的四阶段联合训练。
- Qwen3 是本项目相对于原版 Moshi 的唯一核心骨干替换。由于 Qwen3 的隐藏维度、词表和 tokenizer 与 Helium 不同，所有投影、embedding、输出头都从 Qwen3 config 和实际 Mimi 配置读取，不能照抄 Helium 的维度常数。

原版 Moshi 的四个模型训练阶段是：

```text
0. 准备并冻结 Mimi（本项目跳过 Mimi 训练，只验证）
1. Moshi pre-training：单音频流 + 文本保留训练
2. Moshi post-training：基于 diarization 的模拟双流
3. Fisher fine-tuning：真实双流会话，获得全双工能力
4. Instruction fine-tuning：合成语音指令数据，固定助手声音
```

此外，Moshi 还训练一个**辅助的 streaming multi-stream TTS**，用来生成第 4 阶段的合成指令语音；它不是把 Moshi 变成串联式 TTS，且不应与 Moshi 主模型混为一个 checkpoint。

本文的训练数值首先记录 Moshi 论文中的设置；中文首版因数据规模、GPU 和 Qwen3 规模不同，可以等比例缩短，但必须在实验表中标明“缩短版”，不能把缩短版称为复现原训练预算。

参考依据：Moshi 论文 [§3.4、§4.2–§4.4](https://arxiv.org/html/2410.00037v2)、[Moshi 官方实现](https://github.com/kyutai-labs/moshi)、[官方微调仓库](https://github.com/kyutai-labs/moshi-finetune)。

---

## 第 0 步：准备并冻结 Mimi，不训练编解码器

### 0.1 固定模型资产

| 资产 | 用途 | 本路线是否训练 |
| --- | --- | --- |
| Qwen3 权重、配置、tokenizer | Temporal Transformer 初始化 | 后续阶段训练/适配 |
| 已发布 Mimi checkpoint | 24 kHz 音频 ↔ 12.5 Hz、8 路 codec token | 否，全部冻结 |
| Moshi PyTorch loader 与 delay 工具 | Mimi 加载、流式状态、延迟/反延迟 | 复用并适配 |

当前仓库的下载脚本以 ModelScope 为入口，模型来源和实际文件名以 `scripts/download_models.py` 为准。训练运行前要保存：模型仓库 revision、文件 SHA、Python/PyTorch/Moshi 版本、Mimi `num_codebooks`、采样率和帧率。

### 0.2 只做三项验收

1. `24 kHz waveform → Mimi.encode → Mimi.decode` 能正常重建；Mimi 处于 `eval()`，不建立梯度。
2. token 形状为 `[B, 8, T]`，帧率为 12.5 Hz，即每帧 80 ms；确认每个码本的 cardinality 和 padding 约定。
3. 在 20 条未参与训练划分的音频上保存重建音频、波形图和 token 统计。

如果这一步失败，修复采样率、声道、流式缓存或 loader；不要通过训练 Moshi 主模型来掩盖 codec 问题。

---

## 第 1 步：按 Moshi 结构搭建 Qwen3-Moshi

### 1.1 训练时的三类流

每个 80 ms 时间步包含：

- `U_t`：用户音频 8 路 Mimi token，训练时来自用户音频，推理时来自麦克风；
- `A_t`：助手音频 8 路 Mimi token，训练时是真值，推理时由模型采样；
- `W_t`：助手文本 token，训练时是真值对齐文本，推理时由文本头采样。

Temporal Transformer 沿时间建模；Depth Transformer 在同一个时间步内按码本顺序建模。不能把 8 个码本简单展开为 8 倍时间，也不能用 8 个互相独立的并行分类头替代 Depth。

### 1.2 与 Moshi 对齐的网络结构

```text
用户音频 U + 助手历史音频 A + 助手历史文本 W
        ↓ 各流独立 embedding，再按时间步求和
Qwen3 Temporal Transformer（跨时间因果建模）
        ├── Text head → W_t（Inner Monologue）
        └── projection → Depth Transformer（同一步内按码本自回归）
                              ↓
                    助手音频 A_t 的 8 路 logits
        ↓ delay / undelay 后送入冻结 Mimi Decoder
助手音频
```

必须实现并单元测试：

- 每一路的输入 embedding、输出 projection、文本 head 和音频 head；
- Temporal KV cache；
- 2 帧 acoustic delay 的 pre-training 约定，以及后续阶段 1 帧 delay；
- 文本与音频的时间戳对齐、shift、mask，确保当前目标 token 不泄漏到 Temporal 输入；
- 用户流、助手流、文本流的 reset；一次会话结束时同时清空 codec state、KV cache 和播放队列。

### 1.3 初始化方式

- Temporal Transformer：加载 Qwen3 预训练权重。若 Qwen3 的 attention、RoPE、RMSNorm、SwiGLU 实现与训练框架不兼容，先写显式权重映射和等价性测试，不静默改变结构。
- Depth Transformer：随机初始化。Moshi 的参考配置是 6 层、hidden 1024、FFN 4096、16 heads；Qwen3-Moshi 首版沿用该 Depth 配置，显存不足时只能作为标明的缩小实验。
- 音频 embedding、音频输出头、Temporal→Depth 投影：随机初始化；音频码本 cardinality 从 Mimi 配置读取，不能假设所有码本都与 Qwen3 词表共用。
- 文本 embedding 和文本 head：从 Qwen3 初始化，并保留 Qwen3 原词表语义。音频 ID 绝不能直接送进 Qwen3 原词表。

---

## 第 2 步：Moshi pre-training——单流音频预训练

这是第一段真正的 Moshi 训练，不是单独的 ASR 预热任务。

### 2.1 数据

原版使用约 700 万小时的无监督音频，主要为英语语音，并用 Whisper large-v3 得到转写。中文首版应建立对应的数据版本：

- 大规模中文公开/授权音频，统一重采样为 24 kHz、单声道；
- Whisper 或其他离线转写只用于生成训练监督文本和时间戳，不作为模型推理输入；
- 当前阶段把所有说话人混合成**一条音频流**，不做用户/助手双流；文本流包含这条音频中的词；
- 训练/验证/测试按原始节目、说话人或会话划分，不能把同一录音切片分到两侧；
- 每个 batch item 先按 5 分钟音频组织，Moshi 论文的 audio batch 总量约 16 小时。

同时准备与 Qwen3 语言能力匹配的纯文本数据。原版 Moshi pre-training 中有一半时间继续训练纯文本 batch，以避免 Temporal Transformer 忘记原语言能力；中文首版也必须保留这条支路。

### 2.2 输入和目标

单流阶段仍使用 Moshi 的 Inner Monologue：

- 约 30% 的文本 token 随机 mask；
- 文本和音频之间的 delay 在 `-0.6 s` 到 `+0.6 s` 之间随机化；
- acoustic delay 使用 2 帧；
- 文本 padding 占比很高，padding loss 权重降为 50%；
- 每个训练 step 同时产生文本 logits 和 8 路音频 logits；不调用外部 ASR/TTS。

Moshi 论文的主要训练设置：

| 项目 | Moshi pre-training 参考值 |
| --- | ---: |
| 训练步数 | 1,000,000 |
| audio batch | 约 16 小时 |
| Temporal 学习率 | `3e-5`，linear warmup + cosine |
| Depth 学习率 | `2e-4`，linear warmup + cosine |
| 文本/音频混合 | audio 与纯文本各约 50% |
| 文本 embedding/head 学习率 | audio batch 中乘 `0.75` |
| 优化器 | AdamW，weight decay `0.1`，betas 约 `(0.9, 0.95)` |
| 训练技术 | H100、FSDP、activation checkpointing |

纯文本 batch 使用独立 optimizer state，使文本 batch 和音频 batch 的更新尺度平衡。不能简单把两个 loss 相加后共用一套不受控制的 optimizer state。

### 2.3 联合损失

对每个时间步，文本 token 作为第 1 项，音频 8 路作为后续项。损失按 Moshi 的定义实现：

```text
L = 1/S · Σ_s [ CE(text_s)
    + 1/(Σ_{k=2..K} α_k) · Σ_{k=2..K} α_k CE(audio_{s,k}) ]
```

其中第 1 个音频码本是语义码本，参考权重为 `α_2 = 100`；其余 acoustic codebook 的权重为 `1`。文本 loss 与合并后的音频 loss 同量级，不使用当前文档旧版的 `100:1` 作为整个音频 loss 的比例。padding 和无效时间步必须显式 mask。

### 2.4 阶段验收

- 训练集/验证集文本 CE、音频 CE 分别下降；记录每个码本的 CE，不能只看总 loss；
- 由真值文本或模型文本驱动 Depth 生成音频，再经冻结 Mimi Decoder 检查可懂度；
- 验证文本与音频是否对齐，检查 delay/undelay 后的首尾静音和长度；
- 纯文本能力相对 Qwen3 基线的回归必须单独报告；
- 不能用 CER ≤ 某个预设值作为唯一通过线。该阶段的结论是“联合 token 建模已稳定”，不是“已经具备自然对话能力”。

---

## 第 3 步：Moshi post-training——从单流变成模拟双流

### 3.1 构造模拟双流

对无监督音频数据运行 diarization（原版使用 PyAnnote）：

1. 随机选一个说话人作为主说话人，即未来的 Moshi stream；
2. 根据 diarization mask 提取主说话人 waveform；
3. 其余声音作为用户 stream；两条流分别经过同一个冻结 Mimi Encoder；
4. 文本流只保留主说话人的时间对齐文本；
5. 文本 delay 固定为 0；acoustic delay 改为 1 帧；
6. 不使用轮次门控，不因一方暂时静音而截断另一方的上下文。

该阶段的目的不是获得自然双工，而是让模型学会同时接收用户流并持续生成助手流。它必须保留重叠和非主说话人残留，不能用“主说话人一段、用户一段”的纯轮流数据替代。

### 3.2 参考训练设置

| 项目 | Moshi post-training 参考值 |
| --- | ---: |
| 训练步数 | 100,000 |
| audio batch | 约 8 小时 |
| Temporal 学习率 | `3e-6` |
| Depth 学习率 | `5e-5` |
| 纯文本 batch | 约 10% |
| text/audio delay | 固定 0；acoustic delay 1 帧 |

从第 2 步 checkpoint 继续训练，不重新初始化 Qwen3、Depth 或音频头。每轮验证同时做：主说话人语音生成、用户流变化响应、静音期间的正常继续生成、两流重叠时的输出稳定性。

---

## 第 4 步：Fisher fine-tuning——真实双流会话

### 4.1 数据格式

原版使用 Fisher 约 2000 小时电话会话，每个参与者有独立声道。中文版本必须优先寻找或采集**双麦克风、独立声道、带时间戳的中文对话**；不能把混音后再分离的结果当作 ground-truth 双流。

每条样本保存：

```text
conversation_id
main_speaker_audio.wav
user_audio.wav
main_speaker_timestamped_transcript.json
sample_rate = 24000
speaker_assignment_seed
```

随机选择一方作为 main/Moshi stream，另一方作为 user stream；同一会话的所有切片必须保持在同一个数据划分。文本只监督 main speaker。保留自然停顿、重叠、打断和 backchannel。

### 4.2 训练

- 从 post-training checkpoint 继续；
- acoustic delay 1 帧，text delay 0；
- 参考训练量 10,000 batches，每 batch 约 40 分钟音频；
- Temporal 学习率 `2e-6`，Depth 学习率 `4e-6`；
- 不再混入纯文本 batch；
- 仍使用联合文本/音频 CE 和语义码本加权；
- 验证时必须按会话独立留出，测重叠、打断、附和、静音和正常轮替，而不是只测单人语音重建。

这一阶段才训练模型从真实用户流中持续听、持续说，不能提前加 VAD/turn detector 把全双工问题改成轮流对话。

---

## 第 5 步：Instruction fine-tuning——固定助手声音和行为

### 5.1 先训练辅助 streaming multi-stream TTS

为了生成 Moshi 的 instruction 数据，先训练独立的 streaming multi-stream TTS：

1. audio pre-training 部分与 Moshi 共用单流音频预处理；
2. post-training 使用文本领先音频约 2 秒的 delay，使文本能稳定控制音频；
3. 使用约 170 小时高质量、多说话人、独立声道的 supervised multi-stream conversation 做 TTS 微调；
4. 用它把文本对话脚本渲染成双流语音。

该 TTS 是数据生成工具，不是最终在线推理路径；Moshi 主模型不直接在这 170 小时 supervised multi-stream 数据上训练。

### 5.2 生成 speech-text instruct 数据

原版先用经过 Open Hermes 和真实会话 transcript 微调的 Helium 生成自然口语脚本，再用 multi-stream TTS 合成超过 20k 小时语音。中文首版应保持同样的数据逻辑，而不是直接把文本指令套聊天模板：

- 普通知识问答、短轮次、backchannel；
- 询问声音风格、情绪和角色；
- 错别字/误听后的澄清与重复请求；
- 错误事实纠正；
- 简单数学、语法、常识；
- 安全拒答；
- 介绍模型自己和项目；
- 用户声音随机变化，助手声音固定为一个授权说话人的声音和多种表达风格。

生成文本不能含难以自然朗读的 URL、长列表和不自然的书面格式。每条样本必须包含两路音频、助手时间戳文本、用户音频增强配置和会话划分信息。

### 5.3 训练与用户流增强

从 Fisher checkpoint 继续训练：

| 项目 | Moshi instruction fine-tuning 参考值 |
| --- | ---: |
| 训练步数 | 30,000 |
| audio batch | 约 2.7 小时 |
| Temporal 学习率 | `2e-6` |
| Depth 学习率 | `2e-6` |
| acoustic delay | 1 帧 |

训练中对 user stream 做与 Moshi 对齐的鲁棒性增强：

- 50% 概率随机增益 `-24` 至 `+15 dB`；
- 30% 概率加入 Deep Noise Suppression 噪声，并随机插入最长 30 秒静音段；
- 30% 概率加入助手声音的回声，回声增益取 `[0, 0.2]`，延迟取 `100–500 ms`；
- 30% 概率一起使用 echo/reverb；
- 保留用户说话、静音、打断和助手继续说的目标，不用增强后的 VAD 标签替代真实目标。

助手声音的一致性来自 instruction 阶段固定 speaker 条件，而不是推理时接一个 voice-cloning/TTS 模块。

---

## 第 6 步：训练过程中的统一验证

每个阶段都保存可复现的 checkpoint、数据 revision、配置和 optimizer state，并至少报告：

1. 文本 CE、8 个音频码本 CE、padding loss、梯度范数；
2. 文本流与音频流的时间对齐、实际生成延迟、首帧延迟；
3. 冻结 Mimi 重建质量和生成语音可懂度；
4. Qwen3 原有纯文本能力回归；
5. 单流、模拟双流、真实双流、instruction 数据四套独立验证；
6. 正常轮替、重叠、打断、backchannel、静音和错误事实纠正；
7. GPU 显存、tokens/秒、音频实时率、p50/p95 端到端延迟。

禁止用以下结果宣称 Moshi 训练完成：

- 只有单句 ASR/CER 变好；
- 只有 Mimi 重建质量好；
- 先完整 ASR，再文本 LLM，再独立 TTS 的串联 demo；
- 只有合成的严格轮流对话；
- 只有模型停止生成的延迟，没有实际播放停止延迟；
- 只报告总 loss，不报告文本、语义码本和 acoustic codebook 的分项 loss。

---

## 第 7 步：流式推理闭环

最终在线路径必须是：

```text
麦克风
  → 冻结 Mimi Encoder
  → 用户 8 路音频流
  → Qwen3 Temporal + Text head + Depth
  → 助手文本与助手 8 路音频流
  → delay/undelay
  → 冻结 Mimi Decoder
  → 播放
```

推理时：

- 每 80 ms 接收一帧用户音频；
- 用户流直接进入 history，不经过外部 ASR；
- 只采样助手文本和助手音频；
- 维护 Temporal KV cache、Depth 当前步状态、Mimi 编解码器状态和播放队列；
- 连续运行至少 10 分钟，确认实时率小于 1、队列不持续增长；
- 单独测“模型决定停”和“扬声器实际停”的时间差；
- 测试用户打断时，助手能停止/调整而不是由 VAD 直接替模型做决定。

---

## 本项目的实际落地顺序

为了控制工程风险，按下面顺序实现，但不要改变 Moshi 的训练定义：

1. **Mimi 验证和流式状态**：完成第 0 步，不训练 codec。
2. **Qwen3-Moshi 架构单元测试**：完成第 1 步，先用极小 batch 验证 shift、delay、Depth 和 loss。
3. **单流 pre-training 小规模复现**：完成第 2 步的短跑，确认文本保留训练和联合 loss 正常。
4. **完整单流 pre-training**：扩大无监督中文音频和纯文本数据。
5. **diarization 模拟双流 post-training**：完成第 3 步。
6. **真实中文双流 Fisher 等价数据 fine-tuning**：完成第 4 步。
7. **辅助 streaming multi-stream TTS 和 instruction 合成**：完成第 5 步前半。
8. **instruction fine-tuning 与噪声/回声增强**：完成第 5 步后半。
9. **流式推理、延迟和全双工事件验收**：完成第 6–7 步。

首个可执行里程碑不是 E1/E2 的音频转写实验，而是：**在冻结 Mimi 的前提下，Qwen3-Moshi 能在单流数据上同时降低文本 CE 和 8 路音频 CE，并经 delay/undelay 与 Mimi Decoder 生成可对齐语音**。达到这个里程碑后，才进入模拟双流和真实全双工训练。
