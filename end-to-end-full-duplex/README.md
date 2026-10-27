# 基于 Qwen2.5-Omni 的端到端全双工语音模型

本项目直接使用 **Qwen2.5-Omni 已发布的预训练模型**作为基础模型，在其音频理解、文本响应和语音生成能力之上进行全双工适配。**不再先用 Qwen3、Mimi 和自建音频适配器预训练一个 Omni 基础模型。**

当前重点从“建立基础跨模态能力”转为“持续接收输入、并行生成输出、处理打断和话轮交接，以及管理长会话状态”。优先研究文本与音频交互，不把图像、视频接入作为当前前置要求。

> **实现状态：** 当前已切换项目路线和模型下载入口；Qwen2.5-Omni 的项目内推理封装、全双工后训练、增量输入输出调度及会话管理尚未实现。`src/voxbase_omni/` 仍是保留的旧版 Qwen3 输入侧实验，不能加载或代表新的 Qwen2.5-Omni 路线。

## 1. 基础模型与能力边界

| 项目 | 当前选择 |
| --- | --- |
| 基础模型 | Qwen2.5-Omni，直接使用官方已发布权重 |
| 下载默认值 | `Qwen/Qwen2.5-Omni-7B`；可按资源选择 `Qwen/Qwen2.5-Omni-3B` |
| 核心结构 | 复用原生多模态感知、Thinker、Talker 与音频波形生成链路 |
| 当前输入重点 | 用户音频、文本及其历史上下文 |
| 响应目标 | 文本与语音 |
| 不再作为前置任务 | 自建 Omni 基础模型、Mimi 码本对齐预训练、Qwen3-ASR AuT 融合预训练 |
| 后续工作 | 原生能力基线验证、全双工运行时适配及必要的交互后训练 |

**不需要重新做基础模型预训练，不等于完全不需要训练或运行时改造。** 原模型具备多模态理解及文本/语音生成能力；分块处理和流式生成也不等于本项目已经具备有状态、可打断的全双工会话。是否需要微调、训练哪些模块，应由基线表现和目标行为差距决定。

当前路线优先保留原生音频表示与生成链路，不默认替换为 Mimi，也不将原模型拆成独立 ASR → 文本 LLM → TTS 作为最终方案。原版 Moshi 资料仅用于理解全双工建模，不再要求复现其预训练流程。

## 2. 目标架构

下面是**待实现的目标结构**，不是当前已运行的推理链路：

```text
用户麦克风 / 文本
        ↓ 按实际到达时间接入
音频缓冲与原生预处理 → Qwen2.5-Omni 感知模块 → Thinker
        ↑                                          ├─ 文本响应
        │                                          └─ Talker → 原生音频解码 → 播放
        │                                                                            │
        └──────────── 全双工会话运行时：输入持续接收、输出调度与打断协调 ──────────────┘
                              ↓
                 历史上下文、长期摘要、缓存与实际播放进度
```

全双工适配需要解决：

- **边听边说**：输出语音期间持续接收用户音频，并验证新输入何时真正影响后续输出；不能只做到麦克风仍在录音。
- **打断与附和**：区分反馈和打断，停止过期生成与播放，并保留正确的会话历史。
- **话轮交接**：决定何时发言、继续、等待或让出话轮。
- **流式因果性**：编码、模型输入和调度不能依赖实际交互时尚未到达的未来音频。
- **状态一致性**：协调感知状态、Thinker/Talker 缓存、输出队列与播放进度，处理取消、重连和重置。

## 3. 推进顺序：复用基础能力，再适配交互

### 3.1 验证原生模型基线

先用官方模型和处理器验证音频理解、多轮文本上下文，以及文本与语音响应，不更新模型参数。记录所用模型 revision、依赖版本、设备、显存占用、首文本/首音频延迟及输出质量。

标准离线 `generate()` 成功只能说明基础推理可用，不能据此宣称全双工可用。分块输入、增量缓存与输出流接口须对照实际使用的上游实现逐项验证。

### 3.2 实现全双工运行时

建立带时间戳的输入缓冲、增量感知与生成调度、播放队列、取消机制和会话上下文管理。重点验证输出过程中到达的新输入能否参与决策，以及打断后是否还会播出旧回复。

### 3.3 按需进行全双工后训练

如果原生模型配合运行时仍无法达到交互目标，再进行面向打断、附和、等待和话轮交接的适配训练，而不是重新开展基础 Omni 预训练。

训练数据需要保留双方独立音轨、停顿、重叠发言和事件时间关系，不能把普通单轮语音问答直接当作全双工监督。冻结范围、LoRA 或其他适配方式、损失和数据配比留待基线结果与接口约束明确后确定。

## 4. 长会话上下文管理（待实现）

全双工持续交互不能无限追加历史。上下文管理放在实际持有模型会话状态的运行时，而不只是裁剪客户端显示记录。

- **token/序列位置预算为主，轮数为辅**：请求或增量推进前计入系统指令、历史摘要、近期原文、当前输入和音频展开后的有效序列位置；预留输出及安全余量。不要只统计转写文本，也不要沿用旧 Mimi 路线的帧率估算原生模型的输入长度。
- **提前压缩与硬限制**：软阈值触发旧历史摘要；达到硬预算时等待压缩或执行明确的兜底，压缩后重新核验长度。预算以实际模型配置和输入展开结果为准。
- **分层保存**：保留系统指令、关键事实/任务状态、历史摘要和最近若干轮原文；近期必要音频另设窗口。文本摘要不能替代音色、韵律和重叠发言等全部声学信息。
- **异步且一致**：只摘要已稳定的历史快照，保留期间新到达的消息和未结束交互；压缩失败不覆盖原历史。打断历史应反映实际已播放内容，而非把未播完的整段回复当成已说出。
- **缓存协同**：改写历史后，不能默认复用改写前的全部 KV cache；需明确重建或缓存迁移策略，并同步 Thinker、Talker 与播放状态。

这些是后续实现要求，不表示旧实验代码已有上述能力。

## 5. 模型准备与上游推理入口

### 5.1 下载模型

在本目录下执行，模型文件保存在 Git 忽略的 `model/` 中：

```bash
python -m pip install modelscope
python script/download_models.py --help

# 默认下载 7B 的完整官方模型资产，不只是文本 Thinker。
python script/download_models.py

# 资源受限时可改用 3B；不是要求同时下载两个模型。
python script/download_models.py --model-id Qwen/Qwen2.5-Omni-3B

# 可选：指定模型保存位置。
python script/download_models.py --local-dir /path/to/Qwen2.5-Omni-7B
```

默认保存位置为 `model/Qwen2.5-Omni-7B`；选择 3B 时默认保存为 `model/Qwen2.5-Omni-3B`。脚本可在中断后重跑，但下载完成本身不等于推理或全双工验证通过。

### 5.2 独立环境与官方接口

建议使用独立 Python 环境，不与仓库根目录的级联服务或旧版 VoxBase 实验环境混装。下面是基于 Transformers 4.57 系列官方接口的安装起点，**本项目尚未完成 Qwen2.5-Omni 完整推理环境验收**；PyTorch/CUDA 安装方式按目标设备选择，音视频读取另需系统 `ffmpeg`。

```bash
python -m pip install torch 'transformers==4.57.6' accelerate qwen-omni-utils soundfile
python -c "from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; print('Qwen2.5-Omni imports OK')"
```

完整文本和语音输出应使用 `Qwen2_5OmniForConditionalGeneration` 与配套 `Qwen2_5OmniProcessor`，按官方示例处理输入并调用 `generate()`。仅加载 Thinker 的文本输出示例不能作为语音生成基线。初次验证语音输出时保留官方要求的 system prompt；显存预算随输入长度和生成设置变化，不在此承诺固定硬件占用。

具体推理用法见 [Transformers 官方文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_omni) 与 [7B 模型卡](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)。这些是上游推理入口，不是本项目的全双工启动命令。

## 6. 现有文件与历史方案

| 路径 | 当前定位 |
| --- | --- |
| `script/download_models.py` | 当前 Qwen2.5-Omni 模型下载入口 |
| `model/` | 本地模型资产，不提交 Git |
| `src/voxbase_omni/` | 旧版 Qwen3 + Mimi + 可选 AuT 输入侧实验，保留但不作为新路线入口 |
| `pyproject.toml` | 旧实验包及其依赖，不代表已封装 Qwen2.5-Omni |
| `tests/test_modeling.py`、`tests/test_encoders.py` | 旧输入侧架构测试，不是 Qwen2.5-Omni 或全双工验收 |
| `tests/test_download_models.py` | 下载脚本的离线参数与错误路径测试 |
| [旧版 VoxBase 说明](docs/voxbase-omni-legacy.md) | 保留此前自建 Omni 的结构、依赖与调用契约，仅供历史参考 |
| [旧版 Route A](docs/route-a-implementation.md) | 历史 Qwen3-Moshi 训练路线，不再执行 |
| [旧版架构图提示词](docs/qwen3-moshi-architecture-prompt.md) | 仅描述旧 Qwen3-Moshi 方案 |
| `docs/figures/`、`docs/moshi-architecture-overview.png`、`assert/` | 原版 Moshi 参考图与历史 Mimi 样例，不表示当前模型结构或能力 |

旧实验代码与接口暂不删除，也不通过把模型名称换成 Qwen2.5-Omni 来假装兼容。其使用说明单独保留在历史文档中。

下载入口的离线测试无需安装模型依赖或下载权重：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_download_models.py' -v
```

## 7. 后续验收重点

- 官方权重与处理器能够完成音频理解及文本/语音生成，基础能力没有因适配明显退化。
- 生成与播放期间到达的新输入能够被处理并影响后续行为，而不只是排队到回复结束。
- 打断后停止旧生成与旧播放，历史、缓存和下一次响应保持一致。
- 长会话超过预算时可压缩且不破坏近期上下文、关键事实与必要音频状态。
- 明确记录首音频延迟、打断响应延迟、长会话资源增长及失败恢复情况。

## 上游依据

- [Qwen2.5-Omni 官方仓库](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) / [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [Transformers Qwen2.5-Omni 文档（4.57.1）](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_omni)

原生模型能力以上游文档为依据；本仓库已实现能力以上述实现状态和实际验证为准。
