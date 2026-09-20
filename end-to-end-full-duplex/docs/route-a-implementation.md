# Qwen3 + Mimi 小型全双工模型：总路线与首轮实验

## 总的做法

**不是直接把 Qwen3 换进 Moshi 就能训练对话。先让 Qwen3 读懂 Mimi token，再训练它生成 Mimi token，最后训练双流对话。**

```text
1. 下载 Qwen3 和 Mimi，检查文本推理及音频重建
   ↓
2. 音频输入适配：Mimi token → Qwen3 → 转写文本
   验证 Qwen3 是否真的利用音频，而不是靠文字猜答案
   ↓
3. 音频输出适配：Qwen3 + Depth → Mimi token → 语音
   先验证受控文本条件生成，再联合预测文本和语音
   ↓
4. 语音问答：用户语音 → 助手文本 + 助手语音
   ↓
5. 真正双工：持续双轨输入，学习轮替、附和和打断
   ↓
6. 实时推理：麦克风 → 模型 → 播放，测质量和延迟
```

第 2 步是本项目增加的**输入适配诊断/预热任务**，不是原 Moshi 论文的完整训练阶段；转写成功不等于能理解问答。最终在线路径不串接外部 ASR 或 TTS，仍保留 Moshi 的多流 Temporal/Depth、对齐助手文本、声学延迟与流式状态。

以下按**中文首版、Qwen3-1.7B-Base**给出可执行的建议。数据量、超参数和通过线是首轮工程设置，不是实测结果。本文只更新方案，不执行下载或训练。

## 第一步：下载两个预训练模型，确认能单独运行

### 下载清单

| 资产 | 下载来源 | 用途 |
| --- | --- | --- |
| Qwen3 权重、配置、tokenizer | `Qwen/Qwen3-1.7B-Base` | 作为跨时间建模的 Temporal Transformer |
| Mimi 权重 | `kyutai/moshiko-pytorch-bf16` 中的 `tokenizer-e351c8d8-checkpoint125.safetensors` | 同一个模型提供 Encoder 和 Decoder |
| Moshi 实现 | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | 复用 Mimi loader、Depth 和延迟机制；适配到 Qwen3 |

**不下载原版 7B Moshi LM 作为训练起点，不使用其 SentencePiece tokenizer。** Depth、音频 embedding 和音频输出头是新训练的。Mimi 不是“mini”。

以下是在模块目录执行的下载示例，需要先有含 Hugging Face CLI 的环境；正式运行前把 revision 环境变量设置为选定仓库的 commit SHA，不使用浮动版本训练。

```bash
cd "/Users/guhj/Documents/Full-Duplex Model/end-to-end-full-duplex"
: "${QWEN_REV:?设置 Qwen3 仓库 commit SHA}"
: "${MIMI_REV:?设置 Mimi 所在仓库 commit SHA}"
hf download Qwen/Qwen3-1.7B-Base \
  --revision "$QWEN_REV" --local-dir checkpoints/base/qwen3-1.7b
hf download kyutai/moshiko-pytorch-bf16 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  --revision "$MIMI_REV" --local-dir checkpoints/base/mimi
```

实现时用固定版本 `moshi.models.loaders.get_mimi` 加载这个 Mimi 文件，设置 `num_codebooks=8`，不调用会一并下载 Moshi LM 的完整 checkpoint 加载入口。Qwen3 使用支持该模型的 Transformers 版本；安装后锁定实际依赖版本。

### 做两个检查

1. Qwen3 完成文本续写，保存 tokenizer/config，确认模型层和文本输出头可加载。
2. 从 AISHELL-3 取 20 条录音，转为 24 kHz，执行 `waveform → Mimi.encode → Mimi.decode → waveform`。设置 Mimi 为 eval，关闭梯度。核对每侧 token 形状约为 `[B, 8, T]`、帧率 12.5 Hz、普通码本 ID 范围 0–2047；比较原声与重建音频。

**产物**：两套本地权重、版本记录、20 条重建音频。重建有明显问题就先修采样率/码本配置，不进入训练。

## 第二步：做“Qwen3 能否读懂音频 token”的适配实验

### 2.1 用什么数据

- 主数据：[AISHELL-1](https://www.openslr.org/33/)，使用录音和汉字转写。先从官方训练集抽 **10h**，另从开发集取 **1h** 做验证；官方测试集最后再使用。
- 音频重采样到 24 kHz，保留说话人/原始文件 ID。重采样不代表恢复原本缺失的高频。
- 对转写做强制对齐，再用 Qwen3 tokenizer 转成子词，分配到 80 ms 时间格。快速语速造成同帧多 token 时顺延，并统计实际滞后；不得丢字。
- 缓存 Mimi 的 8 路 token、对齐文本 token、文本有效 mask。按原始文件划分，不能把同一句的不同切片放到训练和验证两侧。

### 2.2 模型具体怎么接

每个码本独立建一个可训练 embedding，大小为 `2048 × Qwen3 hidden_size`，特殊码本标记另留位置。hidden size 从 Qwen3 config 读取。将同一帧的 8 个 embedding 求和作为该帧输入：

```text
8 路 Mimi token [B, 8, T]
    → 8 个音频 embedding，按码本求和
    → 音频帧表示 [B, T, H]
    → Qwen3 inputs_embeds
    → 原 Qwen3 文本输出头
    → 每帧一个文本 token 或文本对齐 PAD
```

- **不把 2048 个 codec ID 当作 Qwen3 原词表 ID，也不把 8 个码本展开成 8 倍时间长度。**
- 这是帧级因果转写任务，不走聊天模板，不加入 Depth，不预测音频。
- 输入不提供参考转写，也不提供上一帧真实文字，只给音频帧；避免模型靠 teacher-forced 文本历史把实验做“成功”。
- 每帧只能看当前和过去音频。转写标签从对应发声结束后再延迟 2 帧（160 ms）开始排布，拥挤时继续顺延；这只是诊断任务的初始延迟设置，不是最终助手回复延迟。
- 为片尾排不下的文字追加真实静音编码帧，并记录额外等待时间。验证必须报告实际转写滞后，不能只报告 160 ms。
- loss 为文本 CE，对齐 PAD 先设权重 0.1，普通文本权重 1；batch padding 不计入 loss。添加专用 PAD/EPAD ID，不覆盖 Qwen3 原有 token 语义。

### 2.3 先跑小样本，再跑两组对照

先用 32 条训练样本检查 loss、梯度和自由转写；这一轮只排查工程问题。随后做以下 10h 数据实验，两组从同一 Qwen3 和同一新增 embedding 初始化出发：

| 实验 | 训练哪些参数 | 回答的问题 |
| --- | --- | --- |
| E1：冻结骨干 | 8 个音频 embedding、新增文本 token 参数；其余 Qwen3 冻结 | 只映射输入是否足够？ |
| E2：LoRA 适配 | E1 + Qwen3 attention 的 q/k/v/o projection LoRA，建议 rank 16、alpha 32 | 骨干需要适配时，轻量更新是否足够？ |
| E3：条件性追加 | E1 + Qwen3 最后 8 个 block 解冻；不叠加 E2 的 LoRA | 仅当 E1/E2 均不行且数据/实现正确时，检查 LoRA 容量是否限制学习 |

**首轮训练设置**：AdamW；新参数学习率 `1e-4`，LoRA `1e-4`，解冻 block `1e-5`；warmup 占更新数 5%，梯度裁剪 1.0。新参数组、LoRA 和 block 分别记录梯度范数。新增文本 token 的参数训练要避免误解冻整个词表。

每组先跑 3 个训练集 epoch，每半个 epoch 验证；同样本顺序、同有效音频时长、同最大上下文。每次 optimizer 更新累计约 120 秒有效音频，通过 microbatch/梯度累积适配显存。上述值用于起跑，不保证收敛；比较学习曲线后再追加同等预算。

### 2.4 怎么判断它真的读懂了音频

验证时不喂参考文字，去掉输出 PAD 后计算中文 CER，并报告插入/删除/替换错误。对同一 checkpoint 再做两项诊断：

1. 把输入换成相同长度的静音 codec token。
2. 把输入换成其他句子的音频，保持待评分的原转写不变。

**进入下一步的建议门槛**：开发集 CER ≤ 30%，且比静音、错配音频两种诊断各降低至少 30% 相对 CER；按原始录音配对 bootstrap 的 CER 差值 95% 区间应支持真实音频更好。主候选重复 3 个随机种子，报告均值/波动。门槛是排除“还没学会输入”的工程线，不是最终 ASR 质量目标。

如果 loss 降但 CER 不降，先查 PAD 占比、标签泄漏和文本错位；如果真实音频与静音差不多，不进入双工训练。达标后将数据扩到 AISHELL-1 训练集，并记录扩量收益。

**产物**：`audio-input-adapted` checkpoint，以及 E1/E2 的 CER、静音/错配 CER、转写滞后、GPU-hours。此时只证明语音内容输入可用，尚未证明问答能力。

## 第三步：让模型能说——训练 Depth 与音频输出

### 数据和初始化

- 数据：[AISHELL-3](https://www.openslr.org/93/)，约 85h 多说话人普通话及转写。先选一名数据量充足的说话人，用其不重复句子的训练/验证划分验证固定音色；再扩到多说话人训练部分。
- 继承第二步 Qwen3 和输入 embedding。为助手音频建立独立 embedding，可复制输入 embedding 初始化，之后不绑定权重。
- 新建 `Qwen3→Depth` 投影、Depth（首试 4 层、hidden 512、8 heads、FFN 2048）和 8 个码本输出头，保留 Depth 按位置区分的参数化。Mimi 继续冻结。

### 分成两个实验，不一开始就做全双工

| 实验 | 输入 | 预测目标 | 训练方式 |
| --- | --- | --- | --- |
| E4：受控发音 | 当前对齐文本 token + 历史助手音频 | 当前调度位置的 8 个助手音频 token | 先固定 Qwen3，训练新增输出模块；验证未训练句子的文本能否变成可懂语音 |
| E5：文本—语音联合生成 | 历史助手文本 + 历史助手音频，不提前输入当前目标文字 | 先预测当前文本，再以文本和 Temporal hidden 为条件让 Depth 预测音频 | 继续训练新增模块和 Qwen3 适配参数；训练时对当前文本条件使用 teacher forcing，生成时换成模型自己的文本 |

E4 是接口与发音诊断，不是最终推理。E5 才接近 Moshi 的联合生成训练。两者都按固定帧率运行，不先生成完整回答后接独立 TTS。

**训练细节**：
- Temporal 跨时间，Depth 在同一调度步内自回归预测 8 个码本；不能用 8 个独立并行头替代 Depth。
- 复用并测试 delay/undelay，单流参考 2 帧声学延迟；训练目标和输入严格移位，不能把当前音频答案送回 Temporal。
- 音频 loss 使用归一化加权 CE，语义码本/其余码本先用 100:1；E5 再加文本 CE。分别记录各码本和非 PAD 文本 loss。
- E5 可从 30% 输入文本遮蔽开始，保留音频历史训练；穿插第二步音频转写任务和合法纯文本任务，起点按更新数 80%/10%/10%（联合生成/转写/纯文本），监测听觉与文本遗忘。

**检查**：E4 用 200 条未训练句子的对齐文本自由生成全部音频，不喂真实历史音频；E5 只给短前缀后自由续写。固定同一独立 ASR 评估器，比较原录音、Mimi 重建、生成音频的 CER，并人工盲听可懂度和重复问题。E4 先以生成 CER 不高于重建 CER + 10 个百分点作为工程门槛；E5 检查自身文本与声音一致性，不用开放续写与唯一参考句的 CER 判正确性。

**产物**：`speech-text-joint` checkpoint、固定文本生成样例、自由续写样例及输入能力回归结果。通过后再扩大到清洗后的 AISHELL-3 数据；多说话人生成用短音频前缀指定声音，不能假装无条件混合数据会自动得到固定音色。

## 第四步：让“听”和“说”接起来——语音问答

**数据**：先准备 5,000–10,000 条人工审校的中文口语问答，涵盖问答、指令、澄清和纠正；用获授权 TTS 生成用户音频和固定音色助手音频，得到统一时间轴的双轨样本。源脚本按训练/验证/测试分组，不能泄漏其音色变体。

**做法**：
1. 加入独立用户与助手两侧音频流；输入用户音频、助手历史音频和历史助手文本，监督助手文本与音频。用户转写不作为主任务的输入。
2. 用户说话期间，助手目标是正确的静音和文本 PAD；助手回答期间仍持续读取用户流。双流参考 1 帧声学延迟。
3. 从第三步 checkpoint 训练 Qwen3 适配参数和全部音频新增模块；混入输入/语音任务做回归，避免只学会固定套路。
4. 对相同问题构造只改一个关键词的音频对，如日期、数量、地点或否定条件；正确答案应随输入变化。

**检查**：保留 200 条未训练语音问答，人工评估答案是否正确、是否遵循要求、是否说清。与同一问答适配 checkpoint 的文本输入参考任务区分比较；另做错配用户音频诊断。第二步 CER 合格而这里失败，优先补语义问答与指令数据，不直接宣称全双工已完成。

**产物**：`spoken-qa` checkpoint。此时允许先轮流说话，目的是检查用户音频能驱动正确的助手语音响应。

## 第五步：用真实双轨数据训练打断和附和

**数据**：先采集或取得授权的 10h 同步双轨中文会话，再按学习曲线扩大到 50–100h。左助手、右用户，保留静音和重叠。两人独立麦克风、共享时间原点；包含正常轮替、停顿、同时起说、打断、短附和、发声中补充条件。

**做法**：
- 从第四步继续训练，不用 VAD 的轮次门控切断用户输入。
- 打断样本必须有“助手停说→用户补充→按新要求回答”的真实目标；附和样本则应包含助手继续说。
- TTS 拼接只补内容，不能替代真实交互。说话人分段不能从混音还原干净重叠双轨。
- [DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) 仅作为可选英文格式调试数据，先查许可，不当作中文打断训练的替代品。

**检查**：测试集按会话独立保留，每类至少先标注 20 个事件。对比第四步与第五步 checkpoint 的打断成功率、实际停播延迟、附和误停率，以及打断后的回答正确性；对正常问答做回归。

**产物**：`duplex` checkpoint 与事件级对比结果。没有真实双轨数据就停在语音问答，不把合成轮次训练称为自然双工完成。

## 第六步：实时运行与最终交付

1. 接通麦克风 → Mimi Encoder → Qwen3/Depth → Mimi Decoder → 播放；持续维护 codec 状态、KV cache 和播放队列，会话结束一起 reset。
2. 每帧记录 encode/model/decode 时间、排队时长及显存；在目标机器连续运行至少 10 分钟，检查 RTF < 1、队列不持续增长。
3. 打断同时测模型停生成和实际停止播放；不能用“一检测到人声就停播”掩盖附和误判。
4. 输出质量、任务正确性、打断/附和、p50/p95 延迟、失败次数及 GPU/缓冲配置一起报告。

**最终交付**：完整组合 checkpoint、Qwen3 tokenizer、模型/延迟配置、版本和数据清单、每阶段实验表、固定样例与失败案例。下载权重不提交 Git；训练入口需实现，不把本文的实验标签当作已有脚本。

## 现在先做哪一件事

**先只完成第一步和第二步的 E1/E2。** 它们回答当前最关键的问题：在 10h 普通话试验上，Qwen3 通过新增音频 embedding 加轻量骨干适配，是否确实能读取 Mimi token。拿到这个结果，再投入 Depth、语音问答和真实双轨数据训练。

参考：[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B-Base)、[Moshi loader（Mimi 文件名与加载方式）](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py)、[Moshi 论文](https://arxiv.org/html/2410.00037v2)、[官方微调数据约定](https://github.com/kyutai-labs/moshi-finetune)。AISHELL 页面列有 Apache-2.0，使用时仍保存下载包实际许可；自采录音和合成声音须取得训练授权。
