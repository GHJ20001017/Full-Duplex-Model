<p align="center">
  <img src="./arvis-brand-assistant-blue.png" alt="ARVIS — A Real-time Voice Intelligence System" width="800" />
</p>

## 项目介绍

· **ARVIS（A Real-time Voice Intelligence System）** 是一个面向自然对话与任务执行的开源实时语音智能系统。项目旨在让语音助手不仅能够“听懂再回答”，还能够在持续倾听中及时回应，处理对话中的插话、附和与打断，让人与 AI 的交流更接近日常交谈。

· **全双工语音交互**：让助手在说话的同时持续倾听，用户无需等待播报结束，就可以插话、补充或打断。当前采用级联式流程：麦克风采集音频，经 AEC3 回声抑制与本地唤醒门控后，进入语音活动检测（VAD）与流式语音识别（ASR），再由语言模型（LLM）生成回复，通过语音合成（TTS）流式播放；输入监听与输出播放并行运行，配合话轮控制协调双方发言。后续将推进端到端全双工模型训练，以 Qwen3 为语言骨干、Mimi 为音频编解码器，参考 Moshi 架构，探索直接对用户与助手音频流进行联合建模。

· **语义路由与话轮控制**：采用由 jev 训练的 **Qwen3-0.6B 意图识别模型**，结合助手已说出的内容与用户的实时转写，输出 `wait`（等待）、`continue`（继续说）、`yield`（让出话轮）等标签，为对话中的发言决策提供依据。相比仅凭音量或静音时长触发打断，语义路由关注用户“说了什么”，区分简短附和与真正的插话意图，让助手知道何时继续、何时等待、何时把话语权交还给用户。

· **语音驱动的 Agent 执行**：[Qwen Audio Agent](qwen-audio-agent/README.md) 连接实时对话与后台任务执行，可通过 **MCP** 接入 Computer Use 等工具，也可通过 **ACP 适配器**连接 Codex、Claude Code（CC）等 **Agent Harness**。用户可以用语音发起任务，在后台 Agent 执行期间继续交流、补充需求或查询进度，任务结果再返回当前对话，让语音助手从“回答问题”走向“执行任务”。

## 安装与启动

下面以 **Apple Silicon macOS** 为例，先启动当前的级联式语音服务端，再选择 **S2S 修改版客户端**或 **Qwen Audio Agent 客户端**连接。中文识别使用 **Paraformer**，回复通过 **OpenAI 兼容的 Chat Completions API** 生成，语音合成使用 **Qwen3-TTS**。服务端需要 Python 3.10+（以下使用 3.11）、Homebrew，以及可用的 LLM API 地址、模型名称和密钥；Qwen Audio Agent 的 Node.js 环境在第 3.2 节单独配置。

### 1. 安装源码与依赖

首次下载项目并创建环境：

```bash
git clone https://github.com/GHJ20001017/Full-Duplex-Model.git
cd Full-Duplex-Model/cascaded-full-duplex

brew install uv portaudio ffmpeg meson ninja pkg-config
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[paraformer,wake-word]"
```

已有仓库时无需再次克隆，直接进入其中的 `cascaded-full-duplex` 目录执行安装步骤。这里安装的是本仓库源码，而不是 PyPI 上的上游版本。

**仅 S2S 修改版客户端需要此步骤。** 它默认启用 AEC3 回声消除，首次使用前需在同一目录构建原生库并准备唤醒词模型；仅使用 Qwen Audio Agent 时可跳过：

```bash
# 首次使用 macOS 开发工具时执行；已安装则跳过
xcode-select --install

# 等待开发工具安装完成后执行
./native/aec3/build_macos.sh
export S2S_AEC3_LIBRARY="$PWD/native/aec3/build/libs2s_aec3.dylib"

# 预下载默认 hey jarvis 唤醒词模型
python -c 'from openwakeword.utils import download_models; download_models(model_names=["hey_jarvis"])'
```

构建需要访问 GitHub；语音模型首次加载也会下载权重，请预留网络、磁盘和内存。其他平台的 AEC3 构建与库路径配置见 [原生 AEC3 说明](cascaded-full-duplex/native/aec3/README.md)，不要在 Linux／Windows 上直接运行上述 macOS 构建脚本。

### 2. 启动服务端（终端一）

进入 `cascaded-full-duplex` 目录并激活环境。先将下面的占位值替换为自己的 OpenAI 兼容服务配置；`LLM_BASE_URL` 是 API 基地址，不要填写完整的 `/chat/completions` 路径。密钥仅保存在本地环境变量中，不要写入仓库。

```bash
source .venv/bin/activate

export OPENAI_API_KEY="替换为你的 API 密钥"
export LLM_BASE_URL="https://你的服务域名/v1"
export LLM_MODEL="替换为该服务实际提供的模型名称"

speech-to-speech serve \
  --host 127.0.0.1 \
  --port 7869 \
  --stt paraformer \
  --paraformer_stt_model_name paraformer-zh-streaming \
  --paraformer_stt_device cpu \
  --enable_live_transcription true \
  --llm_backend chat-completions \
  --responses_api_base_url "$LLM_BASE_URL" \
  --model_name "$LLM_MODEL" \
  --tts qwen3 \
  --qwen3_tts_device mps \
  --qwen3_tts_language Chinese \
  --init_chat_prompt "你是 ARVIS，一个实时语音助手。请用简洁、自然的中文回答。"
```

- `OPENAI_API_KEY` 由 LLM 后端读取；`responses_api_base_url` 同样适用于 `chat-completions` 后端。
- 此配置在 macOS 上用 CPU 运行 Paraformer，Qwen3-TTS 自动使用 MLX。Linux NVIDIA GPU 服务端可将 `--paraformer_stt_device cpu` 改为 `cuda`，将 `--qwen3_tts_device mps` 改为 `cuda`，并准备匹配的 CUDA／PyTorch 环境。
- 服务端首次启动会加载 VAD、ASR、TTS 和默认启用的 Smart Turn 模型，等待模型加载和监听完成后再启动客户端。
- 本机 WebSocket 地址为 `ws://127.0.0.1:7869/v1/realtime`。默认只监听本机；跨机器访问需要显式设置 `--host 0.0.0.0`，客户端使用服务端实际地址。接口自身不提供访问鉴权，不要直接暴露到公网，应通过受控网络或带鉴权的 TLS 网关访问。

### 3. 选择并启动客户端

项目提供两种客户端，连接第 2 步启动的同一个语音服务端，可按使用场景选择：

| 客户端 | 定位 | 主要能力 |
| --- | --- | --- |
| **S2S 修改版客户端** | 在 speech-to-speech 自带客户端基础上修改，适合直接语音对话与双工调试 | 本地麦克风／扬声器、AEC3 回声消除、`hey jarvis` 唤醒、对话窗口，以及关键词／模型语义路由切换 |
| **Qwen Audio Agent 客户端** | 带 Gateway 的语音 Agent 客户端与运行时，适合边对话边执行任务 | WebUI、TUI、桌面悬浮球，后台任务编排，以及 MCP 工具和 ACP Agent 接入 |

服务端默认只有一个流水线实例（`--num_pipelines 1`），先选一种客户端连接；切换时断开前一个客户端的语音会话。下面的 AEC3 构建、`hey jarvis` 和 `talk` 参数属于 **S2S 修改版**，不直接套用于 Qwen Audio Agent。

#### 3.1 S2S 修改版客户端

##### Demo 展示

https://github.com/user-attachments/assets/39a33a28-4189-48f3-a419-51ceb4859e0b

##### 启动客户端

新开终端二，进入**同一个** `cascaded-full-duplex` 目录，再执行：

```bash
source .venv/bin/activate
export S2S_AEC3_LIBRARY="$PWD/native/aec3/build/libs2s_aec3.dylib"

speech-to-speech talk \
  --url ws://127.0.0.1:7869/v1/realtime \
  --interruption-route keyword \
  --wake-word hey_jarvis \
  --wake-word-timeout 300 \
  --wake-ack "嗯哼，您说"
```

允许终端访问麦克风，先说 **“hey jarvis”**，听到“嗯哼，您说”后开始对话。客户端默认打开本地对话窗口；加 `--no-open-browser` 可只输出窗口地址而不自动打开浏览器，加 `--no-ui` 可只使用终端。服务端和客户端分别按 `Ctrl+C` 停止。

如果默认麦克风或扬声器不正确，先列出设备，再用实际设备编号替换示例中的 `1` 和 `2`：

```bash
python -m sounddevice

speech-to-speech talk \
  --url ws://127.0.0.1:7869/v1/realtime \
  --input-device 1 \
  --output-device 2
```

##### 可选：启用模型语义路由

上面的基础命令显式选择 `keyword` 路线，不依赖外部意图识别服务。使用 `wait`／`continue`／`yield` 模型决策时，需要先准备兼容的意图识别 HTTP 服务；这里配置的是其**完整推理接口 URL**，不是 LLM 的 API 基地址。当前仓库提供调用客户端，不会通过 `serve` 自动启动该意图模型。

在**终端一**设置以下变量，然后重新运行第 2 步的完整服务端命令：

```bash
export S2S_SEMANTIC_TURN_URL="https://你的意图服务域名/完整推理路径"
export S2S_SEMANTIC_TURN_TIMEOUT_S="0.12"
```

该接口接收包含 `state.assistant_said`、`state.user_said` 和 `questions.action` 的 JSON POST，请求结果需在 `answers.action.choice` 返回 `wait`、`continue` 或 `yield`。默认超时为 0.12 秒，应按实际服务延迟调整。未配置接口时，请勿选择 `semantic` 路线。

在**终端二**停止旧客户端，再切换为语义路由：

```bash
speech-to-speech talk \
  --url ws://127.0.0.1:7869/v1/realtime \
  --interruption-route semantic \
  --wake-word hey_jarvis \
  --wake-word-timeout 300 \
  --wake-ack "嗯哼，您说"
```

#### 3.2 Qwen Audio Agent 客户端

Qwen Audio Agent 通过自己的 Gateway 连接 S2S Realtime 接口，再向 WebUI、TUI 或桌面端提供对话与任务能力。**保留终端一的语音服务端运行**，不需要同时运行 `speech-to-speech talk`。

**安装与构建。** 需要 Node.js `^22.22.2`、`^24.15.0` 或 `>=26.0.0`，以及 npm 10+。新开终端二，从本仓库根目录进入已包含的源码快照，不必另行克隆上游仓库：

```bash
cd qwen-audio-agent
npm ci
npm run build
npm run cli -- config
```

最后一条命令会显示配置文件路径，并在缺失时创建模板；默认路径为 `~/.config/qwaudio/config.env`。打开该文件，将以下配置项设置为对应值（已有同名项时修改原值，不要重复追加）：

```dotenv
QWEN_AUDIO_REALTIME_PROVIDER=speech-to-speech
SPEECH_TO_SPEECH_REALTIME_URL=ws://127.0.0.1:7869/v1/realtime
AGENT_PROTOCOL=none
```

这里使用本项目的 **7869** 端口，不是上游文档中的 8765。选择 `speech-to-speech` 后，语音模型和 LLM API 仍由终端一的服务端配置，无需为了连接本地 S2S 再填写 DashScope 语音 API Key。`AGENT_PROTOCOL=none` 用于先验证纯语音对话；启用后台任务时，再按 [后端 Agent 配置](qwen-audio-agent/docs/backends/overview.md) 选择并完成相应 Agent 的安装、认证和工具配置。Codex、Claude Code 通过外部 ACP 适配器接入。

**启动 Gateway。** 在终端二的 `qwen-audio-agent` 目录运行：

```bash
npm run gateway
```

**打开 WebUI。** 新开终端三，从本仓库根目录进入 `qwen-audio-agent`，运行：

```bash
cd qwen-audio-agent
npm run cli -- webui
```

在浏览器中允许麦克风访问并连接语音会话。若偏好终端界面，可将最后一条命令替换为 `npm run cli -- tui`；其音频依赖和平台限制见 [TUI 使用说明](qwen-audio-agent/docs/getting-started/tui.md)。

**桌面端可选。** 如需悬浮球界面，在完成上述源码安装和构建后，于 `qwen-audio-agent` 目录执行：

```bash
npm run desktop
```

桌面端内置 Gateway，是独立的启动方式，不必额外执行 `npm run gateway` 或打开 WebUI；切换前停止前述独立 Gateway、断开旧语音会话，在桌面设置中确认选择 Speech-to-Speech 及相同的 Realtime 地址。桌面端详情见 [桌面客户端说明](qwen-audio-agent/docs/desktop/overview.md)。

更多语音后端与服务端参数见 [级联服务文档](cascaded-full-duplex/README.md)；Qwen Audio Agent 的客户端、后台任务及工具配置见 [Qwen Audio Agent 文档](qwen-audio-agent/README.md)。

## 引用

级联路线基于 Hugging Face 的 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 项目，使用或分发时请同时引用上游原项目；组件模型引用见 `cascaded-full-duplex/README.md`。

## License

Apache License 2.0。
