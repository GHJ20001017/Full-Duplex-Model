# 历史方案：VoxBase-omni 自建输入侧模型

> 本文归档路线切换前的 README，保留已有实验的设计和调用说明。当前项目直接采用预训练 Qwen2.5-Omni，不再执行下述自建 Omni 预训练流程；当前路线见 [项目 README](../README.md)。下文“当前”“本次”等措辞均指旧实验，代码路径和安装命令均以 `end-to-end-full-duplex/` 为工作目录。旧 Qwen3 权重需自行准备，当前下载脚本已改为下载 Qwen2.5-Omni。

**VoxBase-omni** 是一个以 **Qwen3-4B-Instruct-2507** 为文本与推理骨干、面向文本和音频的端到端 Omni 模型项目。最终目标是在统一的 Omni 基础模型之上，构建能够边听、边说的全双工交互模型。

项目分两步推进：

1. **先训练 Omni 基础模型**：建立文本和音频之间的语义对齐与理解、响应能力。
2. **再训练全双工交互模型**：在 Omni 基础模型上引入流式输入输出、并行感知与生成，以及打断、附和和话轮交接能力。

音频的核心建模路线是：**先学习离散表征，再引入连续表征，最终联合使用两者。** 离散分支优先建立语义结构，连续分支进一步补充细粒度的声学与时序信息，而不是在两种表征之间二选一。

> 当前已实现输入侧预训练模型：Mimi 八码本表征、可选 Qwen3-ASR AuT 连续特征、projector、逐帧融合，以及与文本混排后送入 Qwen3 的因果语言建模链路。尚未实现语音生成、训练数据流水线及有状态全双工运行时。

## 1. 模型定位


| 项目   | 设计方向                       |
| ---- | -------------------------- |
| 模型名称 | **VoxBase-omni**           |
| 文本骨干 | **Qwen3-4B-Instruct-2507** |
| 输入模态 | 文本、音频                      |
| 响应目标 | 文本与语音                      |
| 音频表征 | 离散表征与连续表征结合，按训练阶段逐步引入      |
| 第一步  | 训练具备音频理解与响应能力的 Omni 基础模型   |
| 第二步  | 将 Omni 基础模型扩展为端到端全双工交互模型   |


这里的“端到端”指向统一模型直接处理文本和音频输入并产生响应的目标，而不是将独立 ASR、语言模型和 TTS 串联作为最终架构。音频编码器、模态适配器与输出解码器仍可以作为模型的组成部分；是否冻结或联合训练这些模块，需要在具体方案中确定。

## 2. 核心思路：先离散，后连续，再联合



### 2.1 离散表征：优先建立语义结构

第一阶段先使用音频的离散表征，与文本骨干建立联系，让模型优先学习输入内容的语义及文本—音频之间的对应关系。

- **音频**：重点建模语音内容、声音事件及其与文本语义的对应关系。
- **跨模态**：让文本与音频中的相关内容能够进入共享的语义建模过程。

离散 tokenizer 选用 **Mimi（`kyutai/mimi`）**，运行编码时显式设置 `num_quantizers=8`：第 1 层为语义码本，后 7 层为声学码本，每层索引范围为 `0..2047`。八层不是该 checkpoint 的默认码本数。这里使用的是 Mimi 的音频编码与量化结果；输入侧的索引 embedding 不调用 Mimi 波形解码器，也不直接复用 Mimi 内部的码本向量。

### 2.2 连续表征：补充细粒度信息

当离散分支建立较稳定的语义对齐与任务能力后，再引入连续特征，补充离散表示可能压缩或遗漏的信息。


| 模态  | 离散分支的优先目标      | 连续分支的补充目标              |
| --- | -------------- | ---------------------- |
| 音频  | 语音内容、声音事件与语义结构 | 音色、音高、韵律、情绪、发声细节与环境声细节 |


这是预期的功能分工，不是严格的信息隔离：离散特征也可能保留声学细节，连续特征也可能包含语义。项目要验证的是两者是否形成有效互补，而不是预设某一种表征只能承载某一类信息。

### 2.3 联合建模：保留离散分支，而非替换

“加入连续特征”不是丢弃离散 token，也不是将原来的离散特征转换成浮点 embedding 后重新命名为连续分支。新增分支应提供来自音频编码过程的、未经该离散量化瓶颈压缩的连续特征。

联合阶段需要满足以下设计要求：

- **保留离散语义路径**，从已有训练结果继续学习。
- **引入连续细节路径**，让模型能利用新增信息，而不只是增加输入长度。
- **保持时间对应关系**，明确两类表征所覆盖的音频片段或时间窗口。
- **支持模态缺失**，不要求每条样本都同时具有文本和音频。
- **保留文本能力**，避免多模态训练明显破坏骨干已有的语言理解与指令遵循能力。



## 3. 当前模型结构：到 LLM 输入为止

```text
24 kHz 音频 → Mimi.encode(num_quantizers=8) → [B, 8, T_d] 索引
                   ├─ 第 1 层 → 独立语义 embedding ────────────────────┐
                   └─ 后 7 层 → 共享声学 embedding → 各层独立 adapter ─┤
                                    语义 + 七路声学均值 → LayerNorm
                                                     ↓ [B, T_d, 2560]
                                                     离散表征 d ───────────┐
                                                                          │
16 kHz 音频 → log-mel → Qwen3-ASR-0.6B 的 AuT → [B, T_c, 1024]             │
                           显式时间索引对齐 → projector(1024 → 2560)       │
                                                     连续表征 c ───────────┤
                                                                          ↓
                                            有 c：门控残差融合；无 c：直接 d
                                                                          ↓
文本 → Qwen3 tokenizer / 原有 embedding → 与音频帧按位置混排成单一序列
                                                                          ↓
                                              Qwen3-4B-Instruct-2507
                                                                          ↓
                                              原有文本 LM head / causal loss
```

### 3.1 离散分支：两张 embedding 表、七个 adapter

令 `D` 为 LLM hidden size；所选 Qwen3-4B 的 `D=2560`，代码从骨干 embedding 实际维度读取，不硬编码。每帧包含八个索引 `q₁…q₈`：

- `E_sem: [2048, D]`：仅第 1 层使用。
- `E_ac: [2048, D]`：后 7 层共用同一张表，不创建七张声学表。
- `A₂…A₈`：七个独立残差 adapter，结构为 `x + Linear(SiLU(Linear(LayerNorm(x))))`，瓶颈默认 256 维。
- 每帧汇聚：`d = LayerNorm(E_sem(q₁) + mean(Aᵢ(E_ac(qᵢ)), i=2…8))`。

同一帧最终只占 **一个 LLM 序列位置**，而不是把八层码本展开成八个 token。声学均值汇聚、adapter 瓶颈和下述门控融合是本次实现采用的可调整默认值，不代表已通过实验选优。

### 3.2 连续分支：AuT 后接 projector

选用 [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 的 `thinker.audio_tower`，不将其 ASR 文本模型接入 VoxBase 的前向链路。AuT 内部 `d_model=896`，**最终输出为 1024 维**；projector 使用 `LayerNorm → Linear(1024, D) → GELU → Linear(D, D)`。

可直接输入缓存的 AuT 特征，也可安装连续分支依赖后输入 log-mel，由模型内的 AuT 在线计算。离散阶段不加载 AuT，projector 与融合模块也不会执行。当前加载器通过官方完整 ASR checkpoint 类加载，再仅保留 AuT，因此加载瞬时内存仍包含 ASR 文本模型。

**对齐与流式边界：** Mimi 是 24 kHz、12.5 帧/秒；AuT 使用 16 kHz、128-bin log-mel，其一秒卷积块通常产生 13 帧，不能按数组下标直接一一对应。调用方必须提供 `continuous_frame_indices[B,T_d]`，指出每个 Mimi 帧对应的 AuT 帧，`-1` 表示该帧没有可用连续特征。代码不做会隐含未来信息的整段插值，也不因两路长度相同就认定已经对齐。

AuT 原生 Transformers 接口没有增量缓存，且块内注意力是双向的。当前支持片段/分块特征接入，**不声称已经实现有状态、严格因果的流式 AuT**。后续流式运行时须按实际接收的音频块构建特征，并根据整块可用时刻而非仅帧中心时间建立对齐索引；自动时间对齐、缓存、重叠块处理与延迟验证不在本次实现范围。

### 3.3 可选融合：缺失连续特征就旁路

有连续特征的帧采用逐维门控残差：

`g = sigmoid(Linear(concat(d, c)))`，`h = d + g * LayerNorm(c)`。

门控线性层权重初始化为 0、bias 为 -2，使初始融合偏向保留已训练的离散分支。这不是严格等于离散分支的初始化。有以下旁路行为：

- 完全不传连续特征：直接返回 `d`，不调用 AuT、projector 或融合模块。
- 一个 batch 中仅部分帧/样本有连续特征：仅对可用帧执行 projector 和融合，其他帧保持 `d` 原值。
- 不支持仅连续、无离散的音频路径；本阶段保留离散分支作为音频输入基准。

### 3.4 与文本混排及预训练损失

`audio_positions[B,T_d]` 指定每个音频帧在最终序列中的位置，可以表达“文本前缀 → 音频 → 文本回答”及多段文本/音频交替。各样本的位置须唯一、递增，`-1` 表示填充帧。最终序列先预留每帧一个位置，模型以音频向量替换该位置的 embedding，通过 `inputs_embeds` 送入 Qwen3；不增加或借用未经定义的音频词表 ID。

- 输入形状：`input_ids[B,S]`、可选 `attention_mask[B,S]`、`audio_codes[B,8,T_d]`；纯文本输入不需要音频参数。
- 音频占位处的 `input_ids` 不被查表，可以填写 `-1`；其他有效位置必须是合法文本 token。音频帧不可放在 `attention_mask=0` 的序列填充处。
- `labels[B,S]` 按**最终混排序列**对齐；模型自动将音频位置和 padding 设为 `-100`。调用方另外将不需监督的 prompt 设为 `-100`。
- 保持标准 next-token shift，只监督文本词表目标。全忽略标签的 batch 返回可反传的零损失而非 NaN；它本身不会提供训练信号。
- 当前预训练目标是**以音频为条件的文本因果语言建模**，不包含预测 Mimi 码本、语音重建损失或波形生成。

输入表征与输出生成是不同模块：本次到 LLM 输入和现有文本 LM head 为止，不实现语音生成头、Mimi 波形解码或全双工交互调度。

## 4. 第一步：训练 VoxBase-omni 基础模型

这一步先解决“能理解音频内容并给出响应”，不以自然全双工交互作为前置要求。内部按以下顺序推进。

### 阶段 A：离散表征对齐与建模

以 Qwen3-4B-Instruct-2507 为起点，接入音频的离散分支，先学习语义层面的文本—音频对应关系。

- 使用音频—文本以及具备时间对应关系的语音数据建立对齐。
- 训练音频适配与多模态任务能力，逐步建立理解与响应路径。
- 在多模态训练中保留文本任务，跟踪语言与指令能力的变化。
- 先形成可检查的离散表征基线，再引入连续分支。



### 阶段 B：引入连续特征并联合训练

从阶段 A 的模型继续训练，为音频加入连续特征路径，并保留已有离散分支。

- 对齐同一语音内容的离散与连续特征，处理不同采样率及序列长度。
- 训练连续适配模块与融合模块，使新增信息能够参与理解和响应。
- 关注音频的音色、音高、韵律与发声细节。
- 记录引入连续分支前后的语义能力、细节感知能力、计算开销与训练稳定性。

**阶段切换以验证结果为依据，不先写死训练步数。** 具体需要达到的稳定性、语义对齐与任务能力标准，应在数据和训练配置确定后制定。

### 阶段 C：统一的多模态指令与响应训练

在离散与连续表征联合工作的基础上，训练面向实际任务的 Omni 模型。

- 处理文本、音频及其组合输入。
- 根据输入内容与指令生成文本或语音响应。
- 建立跨模态内容一致性，并检查单模态输入下的能力。
- 形成可独立评估的 VoxBase-omni 基础模型，作为后续全双工训练的起点。



## 5. 第二步：扩展为全双工交互模型

全双工不是简单地给 Omni 模型增加流式接口，而是让模型在持续接收输入的同时生成输出，并根据新到达的信息调整交互行为。

这一阶段的目标包括：

- **边听边说**：生成语音时仍持续接收用户音频，不因助手发言而停止感知。
- **自然打断与附和**：区分用户的简短反馈、补充信息和真正的打断意图。
- **话轮交接**：学习何时发言、继续、等待或让出话轮。
- **流式状态管理**：维护跨模态时间线、上下文缓存和生成状态，处理暂停、打断与会话重置。

训练数据需要从片段级、多轮指令样本进一步扩展到保留真实时间关系的交互数据，包含双方独立音频流、停顿、重叠发言、附和和打断。

编码、融合与生成路径都需要满足流式因果约束，不能在训练或评估中依赖实际交互时尚未到达的未来信息。第一步中的表征选型应提前考虑这一兼容性，但全双工行为的学习与验收放在第二步完成。

## 6. 尚待确定的技术选型

以下内容仍需在训练与后续全双工阶段确定：

- Mimi 离散表征在目标数据上的语义覆盖与训练效果。
- 分块 AuT 的时间可用性、自动时间对齐、流式状态与延迟预算。
- 当前 adapter、projector 与门控融合默认值的消融，以及上下文预算。
- 各阶段的冻结策略、训练范围、附加损失与数据配比。
- 语音生成模块、音色与韵律条件传递方式、解码与流式输出机制。
- 阶段切换标准，以及基础 Omni 与全双工交互各自的验收指标。



## 7. 实现与使用

当前新增的实现位于 `src/voxbase_omni/`；`model/` 仍仅用于下载的模型资产，继续保持 Git 忽略。此前 Qwen3-Moshi 实现已移除，本次不恢复旧训练脚本。

| 文件 | 职责 |
| --- | --- |
| `src/voxbase_omni/audio.py` | 两张码本 embedding、七个 adapter、projector、显式帧索引对齐及门控融合 |
| `src/voxbase_omni/encoders.py` | 提取 Qwen3-ASR AuT，处理变长 log-mel 并返回特征与 mask |
| `src/voxbase_omni/modeling.py` | 加载 Qwen3、组装混排序列、attention/position/label 处理与预训练 forward |
| `tests/test_modeling.py` | 小尺寸真实 Qwen3 的输入、旁路、损失与梯度测试 |
| `tests/test_encoders.py` | 小尺寸真实 AuT 变长输入及接入 Qwen3 的梯度测试 |

### 安装与本地验证

在本目录下，建议使用独立环境，避免与仓库根目录其他项目的 Transformers 5.x 依赖混装：

```bash
python -m pip install -e .
# 仅需要在线计算连续特征时安装；缓存特征不需要 qwen-asr。
python -m pip install -e '.[continuous]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

依赖固定兼容上游 `qwen-asr==0.0.6` 的 `transformers==4.57.6`。测试使用随机初始化的小尺寸真实 Qwen3/AuT，不下载预训练权重；未安装 `qwen-asr` 时跳过 AuT 测试，不影响离散与缓存特征测试。已验证环境为 Python 3.11、PyTorch 2.10.0、Transformers 4.57.6、qwen-asr 0.0.6。

### 离散阶段的调用契约

初始化使用 `VoxBaseOmni.from_pretrained()`，默认只加载 Qwen3，不加载 AuT。`llm_name_or_path` 也可传 `model/Qwen3-4B-Instruct-2507`；旧 Qwen3 权重需自行准备，当前 `script/download_models.py` 仅下载 Qwen2.5-Omni，不能用于此实验。新增的 embedding、adapter、projector、融合层均需训练。

一次训练 forward 传入：

| 参数 | 内容 |
| --- | --- |
| `input_ids` | 已按最终序列预留音频位置的 `[B,S]` 文本 token 与占位值 |
| `audio_codes` | Mimi `encode(..., num_quantizers=8).audio_codes`，形状 `[B,8,T_d]` |
| `audio_positions` | 每帧在最终序列中的位置 `[B,T_d]`，填充帧为 `-1` |
| `attention_mask` | 最终混排序列有效位置为 1，padding 为 0 |
| `labels` | 与最终序列同长；文本目标为 token ID，prompt 等不监督部分为 `-100` |

返回原有 Qwen3 causal-LM output，使用 `output.loss.backward()` 即可将梯度传至离散分支与未冻结的 LLM。码本 padding 可用 `-1`，但必须同时将对应 `audio_positions` 设为 `-1`，不得把填充帧作为有效音频输入。Mimi 波形预处理需使用其 24 kHz 单声道输入规范；变长波形编码后，调用方负责标记真正有效的码本帧。

### 联合阶段的两种输入

1. **缓存 AuT 特征**：额外传 `continuous_features[B,T_c,1024]`、`continuous_frame_indices[B,T_d]`，以及可选 bool `continuous_feature_mask[B,T_c]`。不要把 projector 之后的 2560 维向量当作 AuT 原始特征。
2. **模型内计算 AuT**：初始化时设置 `with_continuous_encoder=True`；通过 `Qwen3ASRAudioEncoder.load_feature_extractor()` 获取对应预处理器，对 **16 kHz 单声道**音频使用 `padding=True, truncation=False, return_attention_mask=True, return_tensors="pt"`。将输出的 `input_features` 与 `attention_mask` 分别放入 `audio_inputs` 的 `input_features` 和 `feature_attention_mask` 键，另外传入对齐索引。不可同时传 `audio_inputs` 和缓存特征。

两个采样率的音频应来自相同原始时间线，不能只修改采样率数字而不实际重采样。模型内不执行重采样，不自动推断片段边界或伪造时间索引。序列各参数及缓存特征须在同一设备上；常规做法是将整个模型 `.to(device)`，并将新连续编码器移动到同一设备。本次未适配 `device_map` 多设备切分与权重 offload。

冻结策略留给训练配置选择，默认不冻结参数。阶段 A 可显式冻结未使用的 projector/fusion；使用 DDP 时须冻结这些未使用参数或配置 unused-parameter 检测，阶段 B 再解冻并更新 optimizer。若冻结 AuT，训练时还应按需要保持其 eval 状态。保存使用标准 `state_dict`，并保存初始化参数及两个 backbone 的名称/版本；恢复时重建相同结构后 `load_state_dict`。`from_pretrained` 只初始化上游骨干，不是已训练 VoxBase checkpoint 的专用加载器。

### 上游依据与保留内容

- [Qwen3-4B-Instruct-2507 配置](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)：文本 hidden size。
- [Qwen3-ASR-0.6B 配置](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json)、[AuT 官方实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)：输出维度、卷积分块及前向接口。
- [Mimi 配置](https://huggingface.co/kyutai/mimi/blob/main/config.json)、[Transformers 4.57.6 Mimi 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mimi/modeling_mimi.py)：码本及编码接口。

目录中保留的历史文档、架构图和 Mimi 重建样例仅作参考，不代表 VoxBase-omni 的现有能力。其中 [docs/route-a-implementation.md](route-a-implementation.md) 是旧版 Moshi 复现路线，其模型选择、训练顺序、代码路径和命令不再作为当前项目的实现或使用说明。
