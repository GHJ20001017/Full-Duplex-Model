# 适用于 Nano Banana 的 Qwen3-Moshi 中文框架图提示词

> 用途：生成当前 Step3 的科研架构图。以下内容可以整体复制给 Nano Banana。图中文字以中文为主，只有模型名、标准技术名词、张量符号和必要的英文术语保留英文。

## 可直接复制给 Nano Banana 的提示词

请生成一张**论文级、结构准确、中文标注为主的深度学习模型架构图**，主题是：

# Qwen3-Moshi 联合文本-音频建模框架

英文副标题：

**Qwen3-Moshi: Joint Text–Audio Modeling for Full-Duplex Speech**

副标题下方增加一行中文说明：

**基于 Qwen3 Temporal Transformer 与文本条件 Depth Transformer 的联合语音建模**

这是一张方法架构图，不是宣传海报，不是产品界面，不是写实场景。请使用白色背景、横向宽幅布局、扁平矢量风格、清晰的圆角矩形、严格的网格对齐和充足留白。建议画布比例为 16:9 或 2:1，输出高分辨率，适合论文和技术报告。

---

## 一、核心语义和必须表达的主线

图中必须清晰表达以下主线：

**用户音频历史 + 助手音频历史 + 助手文本历史**

经过：

**延迟与历史移位 → 各流独立 Embedding → 按时间步求和 → Qwen3 Temporal Transformer**

然后分成两个联合输出分支：

1. **文本分支**：预测当前助手文本，也就是 Inner Monologue；
2. **音频分支**：将 Temporal hidden state 和当前文本送入 Depth Transformer，按顺序生成 8 路助手音频 codebook token。

最后：

**生成的助手音频 token → Undelay / Frame Re-alignment → 冻结的 Mimi Decoder → 助手语音波形**

必须强调：

- 这是 Moshi 风格的联合文本-音频 token 建模；
- 不是“ASR → LLM → TTS”串联系统；
- 用户音频不经过外部 ASR；
- Mimi 是外部冻结 codec，不参与训练；
- Qwen3 负责跨时间的 Temporal 建模；
- Depth Transformer 负责同一延迟时间步内的 codebook 自回归；
- 8 个音频 codebook 不能画成 8 个互相独立的分类器。

---

## 二、整张图的版式

采用从左到右的主数据流，并在下方增加两个局部放大框。

### 左侧：三类输入流

占画面左侧约 25%。从上到下排列三条平行流：

1. 用户音频流；
2. 助手音频流；
3. 助手文本流。

### 中央：Temporal 主干

占画面中央约 30%。将三条输入流经过预处理和 Embedding 融合后，送入最大的紫色模块：

**Qwen3 Temporal Transformer**

### 右侧：文本和音频输出

占画面右侧约 30%。从 Temporal hidden state 分成：

- 绿色文本输出分支；
- 橙色 Depth 音频输出分支。

### 下方：两个局部细节框

底部放置两个横向放大框：

1. **Depth Transformer 同步骤自回归细节**；
2. **Acoustic Delay、History Shift 与联合 Loss 细节**。

不要让底部细节框遮挡主数据流。

---

## 三、左侧输入流

### 1. 用户音频流

使用蓝色表示用户音频。

从左上角开始：

**用户麦克风音频**

使用简单的 waveform 图标，不使用人物照片，不使用写实场景。

连接到灰色模块：

**冻结的 Mimi Encoder**

模块旁边标注：

- 采样率：24 kHz
- 编码帧率：12.5 frames/s
- 每帧时长：80 ms
- 参数状态：Frozen，不参与训练

输出蓝色的离散 token 网格，标题：

**用户音频 Codec Tokens U**

形状标注：

**[B, 8, T]**

网格有 8 行，每一行代表一个 Mimi codebook：

- 第 1 行：语义码本 Semantic Codebook
- 第 2–8 行：声学码本 Acoustic Codebooks

在网格旁边标注：

**8 路 Mimi 音频 token**

用户音频只作为模型输入条件，不要连接到用户音频输出头。

### 2. 助手音频流

使用橙色表示助手音频。

标题：

**助手音频历史流 A**

形状：

**[B, 8, T]**

使用 8 行橙色 token 网格。

旁边写：

- 训练时：真实助手音频经过冻结 Mimi Encoder 得到的 token
- 推理时：模型已经生成的助手音频历史 token

如果画出助手音频的 Mimi Encoder，必须标注：

**与用户流共享冻结的 Mimi codec 权重**

不要画成独立的 TTS 模块。

### 3. 助手文本流

使用绿色表示助手文本。

标题：

**助手文本历史流 W**

形状：

**[B, T]**

画成一行绿色文本 token 方块，并与音频时间轴对齐。

旁边标注：

- **Inner Monologue：助手内部文本流**
- 训练时：带时间戳的对齐文本
- 推理时：模型生成的文本历史
- 使用 Qwen3 vocabulary

文本对齐模块名称：

**文本时间戳 → 80 ms 音频帧对齐**

不要添加独立 ASR 模块，不要将用户语音转写文本作为 Qwen3 输入。

---

## 四、延迟与历史移位模块

三条流进入一个浅黄色或浅灰色区域，标题：

**延迟时间轴与因果历史准备**

区域内必须明确画出两个不同操作：

### 1. Codebook-specific Acoustic Delay

模块标题使用：

**按码本设置的 Acoustic Delay**

画出延迟向量：

**[0, d, d, d, d, d, d, d]**

用中文说明：

- 第 1 个语义码本不延迟；
- 第 2–8 个声学码本延迟 d 帧；
- Pre-training：d = 2；
- 后续阶段：d = 1；
- 末尾追加 flush frames，不能丢弃音频尾部。

不要把所有 8 个 codebook 统一右移。

### 2. One-step History Shift

模块标题：

**一步 History Shift：只读取上一时间步历史**

标注：

**Previous delayed step → Current prediction**

再标注：

**当前目标 token 不进入 Temporal 输入**

首个时间步使用空 token，标注：

**Absent token −1 → zero embedding**

需要用不同视觉样式区分：

- 无效/缺失 token；
- 合法音频 token；
- 文本 padding token。

不要把 −1 画成合法的 Mimi codebook ID。

---

## 五、独立 Embedding 与逐时间步求和

三条输入流分别连接到三个 Embedding 模块。

### 用户音频 Embedding

蓝色模块：

**用户音频 Embedding**

英文小字：

**8 independent codebook embedding tables**

说明：8 个 codebook 使用独立参数表，先在每个时间步内求和。

输出：

**[B, S, H]**

### 助手音频 Embedding

橙色模块：

**助手音频 Embedding**

英文小字：

**8 independent codebook embedding tables**

说明：不与用户音频 Embedding 共享参数。

输出：

**[B, S, H]**

### Qwen3 文本 Embedding

绿色模块：

**Qwen3 Text Embedding**

说明：

**由预训练 Qwen3 初始化**

输出：

**[B, S, H]**

### 求和节点

三个 Embedding 输出连接到一个圆形求和节点：

**Σ：逐时间步 Element-wise Sum**

输出标注：

**Temporal Input Embeddings [B, S, H]**

旁边放一个红色禁止符号和简短说明：

**不是 Concatenation；不是 8 倍时间展开**

再放一句：

**音频 token ID 不直接进入 Qwen3 原始词表**

在角落定义符号：

- B：batch size
- T：原始音频帧数
- S：延迟并追加 flush frames 后的时间长度
- H：从实际 Qwen3 config 读取的 hidden size

不要写未经确认的固定 H 数值。

---

## 六、中央 Qwen3 Temporal Transformer

这是整张图最重要的模块，使用深紫色边框、浅紫色填充，尺寸最大。

标题：

**Qwen3 Temporal Transformer**

副标题：

**预训练 Qwen3 Backbone，沿时间维度进行因果建模**

模块内部画多层 Transformer blocks，但不需要逐层写出具体层数。

内部标注：

- **Causal Attention Across Time**
- 跨时间因果注意力
- Temporal hidden states
- 保留 Qwen3 的原始结构

输入：

**Temporal Input Embeddings [B, S, H]**

输出：

**h_s ∈ R^H**

在模块旁边画一条时间轴：

**s−2 → s−1 → s → s+1**

这条时间轴表示 Temporal 沿时间建模，不表示 codebook 顺序。

旁边放一个小状态模块：

**Temporal KV Cache**

标注：

- 跨时间步持续保存；
- 新会话开始时清空；
- 不属于新的学习网络。

从 Qwen3 Temporal 输出分为两个粗箭头：

1. 向上的绿色箭头，连接 Text Head；
2. 向下的橙色箭头，连接 Temporal-to-Depth Projection。

必须让读者看出：同一个 Qwen3 Temporal hidden state 同时服务文本预测和音频预测。

---

## 七、绿色文本输出分支

从 h_s 连接到绿色模块：

**Qwen3 Text Head**

小字：

**预训练 output head，可训练适配**

输出：

**Text Logits [B, S, V]**

其中：

**V = Qwen3 vocabulary size**

再连接到：

**当前助手文本 W_s**

旁边标注：

**Inner Monologue**

画一个小选择节点：

**当前文本条件 Current-Text Conditioning**

两条输入说明：

- 训练：Teacher-forced W_s
- 推理：Predicted W_s

当前文本随后进入：

**Depth Text Embedding**

并明确标注：

**新初始化参数表，不直接复用 Qwen3 输入 Embedding**

文本预测结果使用绿色虚线反馈到右侧或上方的未来时间步，标注：

**下一时间步的助手文本历史**

不要让反馈线穿过主干模块。

---

## 八、橙色 Temporal-to-Depth 音频分支

从同一个 h_s 连接到橙色模块：

**Temporal-to-Depth Projection**

标注：

**Linear: H → D**

输出：

**p_s ∈ R^D**

默认 Depth 配置：

- D = 1024
- 6 layers
- 16 attention heads
- FFN size = 4096

强调：

**同一个 Temporal state p_s 注入当前时间步的全部 8 个 Depth positions**

画一条橙色横向 conditioning bus，从 projection 同时连接到下方 8 个 Depth position。不要画成 8 个独立 Temporal 模型。

---

## 九、Depth Transformer 局部放大图

在主图下方放置橙色浅底的大型放大框，标题：

**同一延迟时间步内的 Depth 自回归**

英文副标题：

**Within-Step Autoregressive Depth Modeling**

小字：

**一个 delayed time step s，按 codebook 顺序生成 8 路音频 token**

### 输入位置

从左到右画 8 个连续位置，组成一个横向序列：

1. **当前文本 W_s**
2. **A_s,1**
3. **A_s,2**
4. **A_s,3**
5. **A_s,4**
6. **A_s,5**
7. **A_s,6**
8. **A_s,7**

第一个位置标注：

**Depth Text Embedding(W_s)**

后面 7 个位置标注：

**Depth Audio Embedding(A_s,k)**

每个位置都接收：

- **Temporal condition p_s**；
- **Codebook position embedding**。

在序列上方写：

**Depth inputs = [W_s, A_s,1, A_s,2, …, A_s,7]**

### Causal Depth Transformer

八个位置共同进入一个横跨全序列的模块：

**Causal Depth Transformer**

模块旁标注：

- 6 层
- hidden size 1024
- 16 heads
- FFN size 4096
- 沿 codebook depth 维度使用 causal attention

画一个下三角 Attention Mask，明确表示：

- 第 1 个输出只能看到当前文本和 p_s；
- 第 2 个输出可以看到第 1 个 audio codebook；
- 第 k 个输出可以看到当前文本与前 k−1 个 audio codebook；
- 不能看到自己的目标或未来 codebook。

重点文字：

**Predict A_s,k from h_s, W_s, and A_s,<k — never A_s,≥k**

中文解释：

**第 k 路音频 token 只依赖当前文本、Temporal state 和前面的音频码本。**

### 八路输出头

Depth Transformer 后面连接 8 个对应的输出头：

- **Audio Head 1 → A_s,1：Semantic Codebook**
- **Audio Head 2 → A_s,2：Acoustic Codebook**
- **Audio Head 3 → A_s,3：Acoustic Codebook**
- **Audio Head 4 → A_s,4：Acoustic Codebook**
- **Audio Head 5 → A_s,5：Acoustic Codebook**
- **Audio Head 6 → A_s,6：Acoustic Codebook**
- **Audio Head 7 → A_s,7：Acoustic Codebook**
- **Audio Head 8 → A_s,8：Acoustic Codebook**

输出整体标注：

**8 路 Audio Logits：[B, S, C_k]**

其中：

**C_k：从实际 Mimi 配置读取的第 k 个 codebook cardinality**

不要将 C_k 写成未经验证的固定数字。

底部增加训练/推理区别：

- 训练：Causal Teacher Forcing，支持并行计算 logits；
- 推理：按 codebook 顺序 Sequential Generation。

强调：这里的“同一步”是延迟后的时间步，不一定对应原始波形中的同一物理帧。

---

## 十、右侧音频输出和 Mimi Decoder

8 路输出汇合为：

**Generated Assistant Codec Tokens**

先连接到：

**Undelay / Frame Re-alignment**

标注：

**恢复原始音频 codebook 时间对齐**

再连接到灰色锁定模块：

**冻结的 Mimi Decoder**

输出：

**助手语音波形**

标注：

**24 kHz waveform**

可配一个简单扬声器图标或波形图标，但不要使用真人照片。

必须突出：

**先 Undelay，再进入 Mimi Decoder**

禁止从 Text Head 直接连线到 Mimi Decoder。Mimi Decoder 的输入只能是音频 codec token。

生成的助手音频 token 通过橙色虚线反馈至后续时间步，标注：

**下一时间步的助手音频历史**

不要画成“播放波形必须重新编码才能作为助手历史”的必经路径。

---

## 十一、底部局部框：Delay 与 Shift 时间轴

标题：

**Acoustic Delay ≠ Temporal History Shift**

使用 d = 2 画离散网格。

语义码本：

**原始：s₀ | s₁ | s₂ | s₃**

**延迟后：s₀ | s₁ | s₂ | s₃ | ∅ | ∅**

声学码本：

**原始：a₀ | a₁ | a₂ | a₃**

**延迟后：∅ | ∅ | a₀ | a₁ | a₂ | a₃**

空位置使用浅灰色空心格，不使用数字 0，因为 0 可能是合法 token ID。

再用独立箭头表示：

**Then apply one-step History Shift before Temporal input**

底部写：

**Semantic delay = 0；Acoustic delay = d；History shift = 1**

不要将 d × 80 ms 直接标记为已经测量的端到端系统延迟。

---

## 十二、底部局部框：联合训练目标

使用浅红色框，标题：

**联合 Text–Audio 训练目标**

Text Head 连接：

**Text Cross-Entropy**

8 个 Audio Head 连接：

**Per-Codebook Audio Cross-Entropy**

展示公式：

**L = L_text + L_audio**

**L_audio = (100 × CE₁ + CE₂ + … + CE₈) / 107**

说明：

- 第 1 个语义码本权重：100；
- 其余 7 个声学码本权重：1；
- 有效文本 padding loss 权重：0.5；
- 无效/缺失 token 由 mask 排除。

明确注明：

**100 是音频码本之间的权重，不是整体 Audio Loss 相对于 Text Loss 的 100 倍。**

可以加一条较小的辅助支路：

**纯文本保留训练：Text-only batch → Qwen3 → Text CE**

旁边标注：

**保持 Qwen3 原有文本能力；optimizer state 由训练流程单独管理**

不要将它画成第二个独立语言模型。

---

## 十三、模块初始化和冻结状态图例

在右下角放置图例。

### 预训练初始化、可训练

使用紫色或绿色实线边框：

- Qwen3 Temporal Transformer
- Qwen3 Text Embedding
- Qwen3 Text Head

标签：

**Pretrained initialization, trainable**

中文说明：

**由预训练 Qwen3 初始化，后续训练中可更新。**

### 新初始化、可训练

使用橙色或蓝色实线边框：

- 用户音频 Embedding
- 助手音频 Embedding
- Temporal-to-Depth Projection
- Depth Text Embedding
- Depth Audio Embedding
- Codebook Position Embedding
- Depth Transformer
- 8 个 Audio Heads

标签：

**Newly initialized, trainable**

中文说明：

**为 Qwen3-Moshi 新增的可训练模块。**

### 外部冻结模块

使用浅灰色模块和小锁图标：

- Mimi Encoder
- Mimi Decoder

标签：

**Frozen codec, external to the trainable token model**

中文说明：

**已发布的冻结音频 codec，不参与 Qwen3-Moshi 训练。**

---

## 十四、模型实现边界

用一条清晰的实线大边框圈出以下模块：

**已实现的 Token-Level Qwen3-Moshi Model**

边界内包括：

- 三类 token 输入；
- Delay 与 History Shift；
- 三类 Embedding；
- Element-wise Sum；
- Qwen3 Temporal Transformer；
- Text Head；
- Temporal-to-Depth Projection；
- Depth Transformer；
- 8 个 Audio Heads；
- 联合 loss。

Mimi Encoder、Mimi Decoder、麦克风、扬声器放在边界外，但通过 token 接口连接。

如果画出 codec state、播放队列或在线调度，使用灰色虚线框标注：

**External Runtime Integration**

底部加一句：

**当前实现是 token-level architecture；实时 codec/playback orchestration 不属于本框架图的已实现范围。**

不要宣称：

- 完整全双工系统已经完成；
- Moshi 训练已经完成；
- 已经验证实时运行；
- 已经验证语音质量；
- 已经测得端到端延迟。

---

## 十五、Nano Banana 视觉要求

请严格生成清晰、可读、规整的科研框架图：

- 所有主要模块使用圆角矩形；
- 主要数据流使用粗实线箭头；
- 历史反馈使用虚线箭头；
- 条件注入使用细实线；
- 不要产生交叉混乱的箭头；
- 底部放大框与主图之间使用编号或细连接线；
- 使用中文作为主要说明文字；
- Qwen3、Moshi、Mimi、Temporal Transformer、Depth Transformer、KV Cache、Embedding、Codebook、Logits、Cross-Entropy 等技术名词保留英文；
- 确保中文字体清晰，不出现乱码、错别字、伪造英文单词或无法辨认的小字；
- 不要在图中生成多余的段落文字；
- 每个模块只保留标题和 1–3 行关键说明；
- 复杂说明放在图下方的图例和局部放大框中；
- 主路径缩小后仍然清楚：

**输入流 → Delay/Shift → Embedding Sum → Qwen3 Temporal → Text Head / Depth → Audio Tokens → Undelay → Mimi Decoder**

整体风格应接近顶会论文中的方法图：简洁、准确、克制、可读，不使用夸张光效、3D 透视、卡通人物、机器人、麦克风产品渲染或装饰性神经元图案。

---

## 十六、最终必须突出的核心句

在图的底部或标题下方放置一句简洁中文总结：

**Qwen3 沿时间建模三类历史流；Depth Transformer 在每个延迟时间步内按码本顺序生成助手音频。**

旁边可放英文主线：

**History → Qwen3 Temporal → Current Text → Text-Conditioned Depth → Audio Codebooks**

这句话必须成为整张图的视觉总结。
