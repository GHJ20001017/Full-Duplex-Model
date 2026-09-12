<div align="center">
  <p>&nbsp;</p>

# Cascaded Full-Duplex

低延迟、完全模块化的级联语音智能体流水线：**VAD -> STT -> LLM -> TTS**，通过 **OpenAI Realtime 核心事件集**（WebSocket 与 WebRTC）对外暴露。

本仓库是 Hugging Face [speech-to-speech](https://github.com/huggingface/speech-to-speech) 的增强版分支。上游项目的每个组件都是可替换的，LLM 槽位兼容 OpenAI 协议，因此可以指向托管服务商、[HF Inference Providers](https://huggingface.co/inference-providers) 或自己硬件上的 vLLM / llama.cpp 服务器，构成完全本地、完全开源的对话栈。在此基础上，本分支针对**真实双工对话的痛点**做了工程化增强，具体见 [Enhancements](#enhancements)。

<hr/>
</div>

## 目录

* [特性总览](#特性总览)
* [核心增强 Enhancements](#enhancements)
* [How it works 工作原理](#how-it-works-工作原理)
* [安装 Installation](#安装-installation)
* [快速开始 Quickstart](#快速开始-quickstart)
* [支持的组件 Supported Components](#支持的组件-supported-components)
* [命令 Commands](#命令-commands)
* [对话界面 Conversation Window](#对话界面-conversation-window)
* [Realtime API](#realtime-api)
* [多语言支持 Multi-Language Support](#多语言支持-multi-language-support)
* [CLI 参考 CLI Reference](#cli-参考-cli-reference)
* [Citations 引用](#citations-引用)
* [License 许可](#license-许可)

## 特性总览

这是一个面向真实语音对话场景的级联双工流水线。除了上游项目的模块化级联与 OpenAI Realtime 兼容层，本分支主要解决了以下问题：

| 场景痛点 | 本分支方案 |
|---|---|
| 本地对讲时扬声器声音进入麦克风，导致 ASR 反复把助手自己的话认成用户输入 | **AEC3 回声抑制**：原生 WebRTC `EchoCanceller3`，在送入 VAD / 唤醒词 / 上传前对麦克风音频做声学回声消除 |
| 长期挂机时麦克风一直向服务端上传无效音频，浪费算力、易误触发 | **本地服务端唤醒词**：客户端 openWakeWord 门控，识别到唤醒词后只在语音窗口内上传音频，并触发服务端固定应答 |
| 用户说话期间要尽快看到转写，但传统整段转写有延迟 | **流式 ASR**：Parakeet 智能渐进式转写（每 500 ms 出部分结果）+ 流式 Paraformer（中文） |
| 助手说话时用户插话，简单 VAD 就暴力打断，容易误伤“嗯”“对”等附和音 | **语义化 barge-in**：`TurnController` 等待稳定转写再决定是否打断，区分附和音（backchannel）与真正的打断意图 |

## Enhancements 核心增强

### 1. AEC3 回声抑制（Echo Cancellation）

本地双工场景下，扬声器放出的助手声音会被麦克风重新采集。若不处理，VAD 会把助手自己的声音当成用户输入，STT 也会把回声当成指令，造成自打断和对话失控。

本分支在 [native/aec3](native/aec3/README.md) 中提供了一个极小的 C ABI 适配层，封装现代 [`webrtc-audio-processing`](https://webrtc.github.io/webrtc-org/) 2.x 的 `AudioProcessing` 模块，启用真实的 WebRTC `EchoCanceller3` 路径：

```cpp
config.echo_canceller.enabled = true;
config.echo_canceller.mobile_mode = false;
```

客户端内部使用 10 ms 单声道 PCM 帧，将播放给扬声器的音频（render）喂给 `ProcessReverseStream`，将麦克风采集的音频（capture）在**送入唤醒词门控和 WebSocket 上传之前**喂给 `ProcessStream`：

```python
def callback_recv(outdata, ...):
    playback.write(outdata)
    aec3.process_render(bytes(outdata))   # 把正在播放的音频作为参考信号

def callback_send(indata, ...):
    mic_queue.put_nowait(aec3.process_capture(bytes(indata)))  # 回声消除后再上传
```

在 Apple Silicon macOS 上构建：

```bash
./native/aec3/build_macos.sh
```

产物为 `native/aec3/build/libs2s_aec3.dylib`（默认被 Git 忽略）。其余平台可用 `S2S_AEC3_LIBRARY` 环境变量指定库路径加载对应的 `libs2s_aec3.so` / `s2s_aec3.dll`。AEC3 处理是默认开启的线路——旧的 `--block-mic-during-playback` 选项已标记为废弃兼容项，麦克风采集始终经过 AEC3 处理。

> 说明：AEC3 属于原生音频处理，在 `talk` / `local` 的本地麦克风-扬声器客户端中生效。浏览器端的 WebSocket / WebRTC 客户端（`demo/`）与原生客户端共享 Realtime 协议，但不走这套声音设备管线。

### 2. 本地服务端唤醒词（Wake Word）

语音助手在无人说话时仍持续采集麦克风并上传音频，白白消耗带宽、算力和模型推理，还容易误触发。本分支加入**客户端本地唤醒词门控**：

- 客户端用 [openWakeWord](https://github.com/dscripka/openWakeWord)（本地 ONNX 推理，不上传音频）持续监听唤醒词，默认 `hey jarvis`。
- 未唤醒时不向服务端发送任何音频；唤醒后进入“语音窗口”，允许跟随的语音通过，并带 400 ms preroll 保留前面一小段，避免吞掉第一个音节。
- 唤醒窗口默认 300 秒（`--wake-word-timeout`），期间用户说话才持续上传；窗口内无语音或超时则自动回到未唤醒状态。
- 唤醒成功后，客户端向服务端发送一条私有的 `wake_word.detected` 事件（不属于公开 OpenAI Realtime schema），服务端直接把一段固定的应答文本（默认中文“嗯哼，您说”，可用 `--wake-ack` 修改）送入 TTS 队列，实现“听到唤醒 → 音箱应答 → 用户开口”的本地服务端唤醒闭环。

```bash
# 安装唤醒词依赖
pip install "speech-to-speech[wake-word]"

# talk 命令默认即启用 hey jarvis 唤醒
speech-to-speech talk --wake-word "hey_computer" --wake-word-timeout 120 \
    --url ws://127.0.0.1:7869/v1/realtime
```

相关参数汇总见 [Cli 参考](#cli-参考-cli-reference)，实现见 `src/speech_to_speech/api/openai_realtime/wake_word.py`。

### 3. 流式 ASR（Streaming Speech-to-Text）

本分支提供**两种**流式转写能力，按 STT 后端选择：

#### Parakeet：智能渐进式转写（Smart Progressive Streaming）

为 `parakeet-tdt` 后端新增了 [SmartProgressiveStreamingHandler](src/speech_to_speech/STT/smart_progressive_streaming.py)，实现在用户说话**过程中**持续产出部分转写，而不是等整段说完：

- 每 **500 ms** 输出一次部分转写（`transcribe_incremental` / `transcribe_progressive`）。
- 使用**增长窗口**（最长 15 s）保证较短的语句也能获得足够上下文、提升准确率。
- 音频超过 15 s 时，按**句子边界**滑动窗口：已完成的句子固化为 `fixed_text`，只对“活动中的”部分重新转写，在准确率与窗口长度之间取得平衡。

```python
result = handler.transcribe_incremental(audio)   # 反复调用，携带增长的音频缓冲
print(result.fixed_text, result.active_text, result.is_final)
```

#### Paraformer：流式中文 ASR

`paraformer` 后端默认加载流式中文模型 `paraformer-zh-streaming`，通过 FunASR 的增量 `cache` 机制实现流式解码。流水线（VAD）会把**只包含新增采样的音频块**（`streaming_audio_chunks`）送入 STT，而不是整段重发，配合逐轮增量缓存降低计算量：

```bash
speech-to-speech serve --stt paraformer \
    --paraformer_stt_model_name paraformer-zh-streaming
```

> 启用 Paraformer 需要安装可选依赖 `pip install "speech-to-speech[paraformer]"`，详见 [支持的组件](#支持的组件-supported-components)。

### 4. 语义化 Barge-in（打断控制）

基于 VAD 的原始打断有一个常见问题：**任何**近似语音的音频（包括“嗯”“对”“好的”等附和音、以及助手自己的回声残留）都可能触发打断，把好端端的回答切掉。

本分支的 [TurnController](src/speech_to_speech/api/openai_realtime/turn_controller.py) 是**一个偏保守的语义门控**：VAD 只负责说“存在类语音音频”，它负责在把候选升级为真正的硬打断之前，等待一个足够稳定的转写：

- **附和音（backchannel）**（如 `嗯`、`啊`、`哦`、`好的`、`知道了`、`ok`、`okay` 等）→ **不打断**，助手继续说话。
- **明确的打断短语**（如 `等一下`、`停一下`、`暂停`、`先别说`、`不是这个意思`、`我换个问题` 等）→ **立即取消** 当前回答。
- **最终转写是权威的**：即便部分转写过短，一旦拿到最终转写且非附和音，就视为真实用户轮次并打断。
- 候选处于“待定打断”状态时按时间与内容综合评判，见 `service.begin_barge_in_candidate` / `_evaluate_barge_in` / `_confirm_barge_in`。

打断仍然遵循会话配置：`turn_detection.interrupt_response`（默认开启）控制是否允许打断，见 `RuntimeConfig.interrupt_response_enabled`；关闭时用户说话会被转写但回答继续播放。

## How it works 工作原理

级联流水线由四个组件构成，各自运行在线程中、通过队列连接：

1. **Voice Activity Detection (VAD)**：[Silero VAD v5](https://github.com/snakers4/silero-vad) 检测语音边界与轮次转换，并把“只含新增采样”的音频块传递给流式 STT。
2. **Speech to Text (STT)**：转写用户轮次，可选流式部分转写（见上文流式 ASR）。
3. **Language Model (LLM)**：生成回答，支持流式文本与工具调用。
4. **Text to Speech (TTS)**：合成音频并流式回传给客户端；播放的音频同时作为参考信号送入 AEC3，用于回声抑制。

每个阶段都有多个可互换后端，通过 CLI 标志选择。代码便于修改，聚焦于 Transformers 与 Hugging Face Hub 上可用的模型。

## 安装 Installation

需要 Python 3.10+。

```bash
pip install speech-to-speech
```

默认安装覆盖标准的 Realtime 路径：

- Parakeet TDT 用于 STT
- OpenAI 兼容 API 用于语言模型
- Qwen3-TTS 用于语音输出（非 macOS 默认 GGML 后端，Apple Silicon 用 `mlx-audio`）
- 本地音频与 Realtime 服务器模式

### 可选组件

```bash
pip install "speech-to-speech[wake-word]"       # openWakeWord 唤醒词
pip install "speech-to-speech[kokoro]"          # Kokoro-82M TTS（非 macOS）
pip install "speech-to-speech[pocket]"          # Pocket TTS
pip install "speech-to-speech[chattts]"         # ChatTTS
pip install "speech-to-speech[omnivoice]"       # OmniVoice TTS
pip install "speech-to-speech[faster-whisper]"  # Faster Whisper STT
pip install "speech-to-speech[whisper-mlx]"     # Lightning Whisper MLX STT（macOS）
pip install "speech-to-speech[paraformer]"      # Paraformer 流式 STT（通过 FunASR）
pip install "speech-to-speech[mlx-lm]"          # mlx-vlm 视觉模型（macOS）
pip install "speech-to-speech[supertonic]"      # Supertonic TTS
pip install "speech-to-speech[webrtc]"          # WebRTC 支持
```

已废弃的实现（含 MeloTTS 等）存放在 [`archive/`](./archive)，不再接入 CLI。

> **DeepFilterNet 注意**：DeepFilterNet 用于 VAD 的可选音频增强，要求 `numpy<2`，与 Pocket TTS（要求 `numpy>=2`）冲突，仅在不用 Pocket TTS 的环境里手动安装。

### 从源码构建

```bash
git clone https://github.com/GHJ20001017/Full-Duplex-Model.git
cd "cascaded-full-duplex"
uv sync
```

（macOS 上构建 AEC3 原生库：`./native/aec3/build_macos.sh`。）

## 快速开始 Quickstart

### 服务端 + 独立客户端

```bash
# 终端 1：启动服务器
export OPENAI_API_KEY=...
speech-to-speech serve

# 终端 2：本地麦克风/扬声器客户端（默认启用 hey jarvis 唤醒词 + AEC3）
speech-to-speech talk --url ws://127.0.0.1:7869/v1/realtime
```

服务器监听 `ws://localhost:7869/v1/realtime`。先对麦克风说唤醒词（默认 `hey jarvis`），再开始对话；助手播放回答时会经过 AEC3 回声消除。

### 一条命令本地运行

```bash
speech-to-speech local
```

### 完全本地 LLM

用 llama.cpp 在本机服务 Gemma 等模型，再把 OpenAI 兼容的 LLM 后端指向它（详见 [LLM backends](#llm-backends)）：

```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 65536 -fa on --swa-full

speech-to-speech serve \
    --model_name "ggml-org/gemma-4-E4B-it-GGUF" \
    --responses_api_base_url "http://127.0.0.1:8080/v1" \
    --responses_api_api_key ""
```

## 支持的组件 Supported Components

| 组件 | 后端 | 平台 | 安装方式 |
|---|---|---|---|
| VAD | [Silero VAD v5](https://github.com/snakers4/silero-vad) | 全部 | 内置 |
| VAD 增强 | WebRTC AEC3 回声抑制 | 原生（macOS 构建脚本） | `native/aec3/build_macos.sh` |
| STT | [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)（默认，支持智能渐进式流式转写） | CUDA / CPU（nano-parakeet）、Apple Silicon（MLX） | 内置 |
| STT | [Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper)（Transformers） | CUDA / CPU | 内置 |
| STT | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) | CUDA / CPU | `faster-whisper` |
| STT | [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx) | Apple Silicon | `whisper-mlx` |
| STT | [MLX Audio Whisper](https://github.com/huggingface/mlx-audio) | Apple Silicon | macOS 内置 |
| STT | [Paraformer](https://github.com/modelscope/FunASR)（默认**流式中文**） | CUDA / CPU | `paraformer` |
| STT | OpenAI 兼容 `/v1/audio/transcriptions` | 本地或远程 HTTP | 内置 |
| LLM | OpenAI 兼容 API（`responses-api` / `chat-completions`） | 托管或自托管 | 内置 |
| LLM | [Transformers](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) | CUDA / CPU | 内置 |
| LLM | [mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple Silicon | macOS 内置 |
| TTS | [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)（默认） | GGML / CUDA（Linux）、mlx-audio（macOS） | 内置 |
| TTS | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | CUDA / CPU、Apple Silicon | 非 macOS `kokoro`；macOS 内置 |
| TTS | [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) | CPU / CUDA | `pocket` |
| TTS | [ChatTTS](https://github.com/2noise/ChatTTS) | CUDA / CPU | `chattts` |
| TTS | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | CUDA / Intel XPU / Apple Silicon | `omnivoice` |
| TTS | [MMS TTS](https://huggingface.co/docs/transformers/model_doc/mms) | CUDA / CPU | 内置 |
| TTS | OpenAI 兼容 `/v1/audio/speech` | 本地或远程 HTTP | 内置 |
| 唤醒词 | [openWakeWord](https://github.com/dscripka/openWakeWord)（本地 ONNX） | 全部 | `wake-word` |

用 `--stt`、`--llm_backend`、`--tts` 选择具体实现。CLI 只构造所选后端的配置；未激活后端的已知选项仍被接受但会忽略并告警。

> 流式 ASR 说明：`parakeet-tdt` 默认启用智能渐进式转写；`paraformer` 默认用流式中文模型且流水线只发送新增音频块。二者都要求启用实时转写开关（`--enable_live_transcription`，见 [Realtime API](#realtime-api)）。

## 命令 Commands

| 命令 | 行为 | 使用场景 |
|---|---|---|
| `serve` | 通过 OpenAI Realtime WebSocket / WebRTC 运行流水线服务器 | 在开发基于 API 的应用或设备 |
| `talk --url <完整-realtime-url>` | 运行打包的麦克风/扬声器客户端 | 想与现有 Realtime 服务器对话 |
| `local` | 在进程内通过 loopback 组合 `serve` 与 `talk` | 想用一条命令同时运行服务器并对话 |

`serve` 默认绑定 `127.0.0.1`；需要对外暴露时显式传 `--host 0.0.0.0`。`local` 始终绑定 loopback，并把打包客户端连接到 `ws://127.0.0.1:<port>/v1/realtime`。

> **端口变更**：本分支的默认 Realtime 端口为 **7869**（上游为 8765）。若沿用旧配置请显式传 `--port`。

## 对话界面 Conversation Window

`talk` / `local` 启动时会额外打开一个本地浏览器窗口，实时显示**用户转写**与**助手回复**，看起来像普通的聊天界面。它默认开启，终端输出保持不变。

- 客户端在 loopback 上以随机端口起一个极小的 HTTP 服务（Python 标准库 `http.server`，无需新增依赖），并自动在默认浏览器打开页面；启动日志会打印实际地址，如 `Conversation window available at http://localhost:53210/`。
- 页面通过 Server-Sent Events（`/events`）接收事件：用户与助手的流式转写在同一个气泡内逐字追加，轮次结束时定稿。刷新页面会重放最近的定稿消息，不会清空对话。
- 页面**只展示用户与服务端返回的对话文本**，不显示工具调用、原始 JSON 或任何协议事件。
- 界面为深色、单色画布，只在「角色回显」处借用状态色（用户＝青色、助手＝紫色），与服务端状态灯一致。

关闭或调整：

```bash
# 完全关闭窗口，只用终端
speech-to-speech talk --url ws://127.0.0.1:7869/v1/realtime --no-ui

# 只起服务、不自动打开浏览器（自行访问打印出的地址）
speech-to-speech talk --no-open-browser

# local 命令：默认开启，用 --no-local-audio-ui 关闭
speech-to-speech local --no-local-audio-ui
```

> 浏览器不可用或端口绑定失败时，客户端会打印一条告警并**自动退化为纯终端模式**，对话本身不受影响。

## Realtime API

Realtime 模式在 WebSocket 与 WebRTC 之上支持 OpenAI Realtime 协议，含实时转写与低延迟轮次切换。WebSocket 客户端连接 `/v1/realtime`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7869/v1",
    websocket_base_url="ws://localhost:7869/v1",
    api_key="not-needed",
)

with client.realtime.connect(model="local") as conn:
    conn.send(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a helpful assistant.",
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "interrupt_response": True,
                        }
                    }
                },
            },
        }
    )

    for event in conn:
        print(event.type)
```

服务器实现核心 Realtime 事件集：入站 `input_audio_buffer.append`、`session.update`、`conversation.item.create`、`conversation.item.truncate`、`response.create`、`response.cancel`；出站语音开始/结束、流式转写、音频增量、工具调用与 `response.done`。CI 通过 SDK 自带的 WebSocket 和 WebRTC 传输连接固定的 `@openai/agents` `RealtimeSession` 实例。这是一个经过测试的核心子集，而非完整 OpenAI Realtime API 等价性的声明。扩展事件与架构细节见 [Realtime Engine README](./src/speech_to_speech/api/openai_realtime/README.md)。

### 本分支新增事件

- `wake_word.detected`（客户端 → 服务端，**私有事件**）：通知服务端唤醒词已被本地识别，服务端直接把固定应答文本送入 TTS 队列作为回应。见 [Enhancements 第 2 节](#2-本地服务端唤醒词wake-word)。
- 实时转写结合流式 ASR，会持续输出 `input_audio_transcription.delta`（部分结果）与 `completion`（最终结果）。

### 实时转写（Live Transcription）

```bash
speech-to-speech serve --enable_live_transcription
```

配合流式 ASR，客户端会通过 `conversation.item.input_audio_transcription.delta` 逐个字符地看到正在进行的用户转写，详见 `_FriendlyEventRenderer` 的实时覆盖显示。

## LLM Backends

LLM 是流水线中最耗算力、延迟最高的组件。支持：

- **本地推理**：CUDA / CPU 上的 `transformers`，Apple Silicon 上的 `mlx-lm`。
- **自托管服务器**：`responses-api` 和 `chat-completions` 可指向本机 [vLLM](https://github.com/vllm-project/vllm) 或 [llama.cpp](https://github.com/ggerganov/llama.cpp) 服务器。
- **服务商 API**：同一批后端也可对接 OpenAI、[HF Inference Providers](https://huggingface.co/inference-providers)、[OpenRouter](https://openrouter.ai) 等 OpenAI 兼容服务商。

两个 API 后端共享同一组 `--responses_api_*` 连接参数：

- `--llm_backend responses-api`（默认）访问 `/v1/responses`。
- `--llm_backend chat-completions` 访问 `/v1/chat/completions`。

### 直接音频输入（无需 STT）

用 `--stt none --llm_backend chat-completions` 把每个完整的 VAD 音频段直接发送给支持音频输入的模型。该模式不支持 `responses-api` 后端，且 `--model_name` 必须显式指定为支持音频的模型（默认 `gpt-5.6-terra` 只接受文本与图像）。同时支持嵌入 WAV base64（`--responses_api_audio_content_type input_audio`，默认）或 base64 data URL（`audio_url`）。

```bash
speech-to-speech serve \
    --stt none \
    --llm_backend chat-completions \
    --model_name "YOUR_AUDIO_CAPABLE_MODEL" \
    --responses_api_base_url "https://provider.example/v1" \
    --responses_api_api_key "$PROVIDER_API_KEY"
```

### Responses API 后端

`--responses_api_base_url` 指向任意实现了 OpenAI Responses API 的服务商或服务器：

| 服务商 / 服务器 | `--responses_api_base_url` | `--responses_api_api_key` |
|---|---|---|
| OpenAI | 省略则用 OpenAI 默认 | `$OPENAI_API_KEY` |
| HF Inference Providers | `https://router.huggingface.co/v1` | `$HF_TOKEN` |
| OpenRouter | `https://openrouter.ai/api/v1` | `$OPENROUTER_API_KEY` |
| vLLM | `http://localhost:8000/v1` | 省略或任意串 |
| llama.cpp | `http://127.0.0.1:8080/v1` | 空串 |

```bash
speech-to-speech local \
    --stt parakeet-tdt \
    --llm_backend responses-api \
    --tts qwen3 \
    --model_name "gpt-4o-mini" \
    --responses_api_api_key "$OPENAI_API_KEY" \
    --responses_api_stream \
    --enable_live_transcription
```

### Chat Completions 后端

配置与 `responses-api` 相同，但访问 `/v1/chat/completions`。适合：服务商在 Responses 路径下忽略 `chat_template_kwargs.enable_thinking`、或服务器在 Responses 流式工具调用路径上不可靠（部分 vLLM 构建，见上游 [#312](https://github.com/huggingface/speech-to-speech/issues/312)）的情况。可用 `--responses_api_reasoning_effort none` 关闭推理以降低语音延迟。

## 多语言支持 Multi-Language Support

语言覆盖取决于所选 STT / TTS 后端：

| 组件 | 后端 | 语言 |
|---|---|---|
| STT | Parakeet TDT（默认） | 25 种欧洲语言 |
| STT | Whisper / Whisper MLX / Faster Whisper | 取决于所选 checkpoint 的广泛多语言覆盖 |
| STT | Paraformer | 取决于所选 FunASR checkpoint；默认偏中文（`zh`） |
| TTS | Qwen3-TTS（默认） | 多语言，`--qwen3_tts_language auto` 默认 |
| TTS | Kokoro | 多种语言/音色映射 |
| TTS | ChatTTS | 英文与中文 |
| TTS | MMS TTS | 通过 MMS checkpoint 的广泛多语言覆盖 |

务必确保所选 STT、LLM、TTS 都覆盖你的目标语言。两种用法：

- **单一语言**：`--language` 指定目标语言代码，默认 `en`。
- **语言切换**：`--language auto`，STT 检测每句语音的语言并转发给 LLM；可选 `--enable_lang_prompt` 追加“请用……回复我”指令。

## CLI 参考 CLI Reference

流水线 CLI 参数详见 [arguments classes](./src/speech_to_speech/arguments_classes) 与 `speech-to-speech serve -h`；客户端参数见 `speech-to-speech talk -h`。

### 模块级参数

见 [ModuleArguments](./src/speech_to_speech/arguments_classes/module_arguments.py)。可设置公共 `--device`、macOS 默认（`--mac-optimal-settings`）、STT / LLM / TTS 实现、日志级别、实时转写与流水线池大小（`--num_pipelines`）。

日志默认不包含内容：含转写的记录只报告字符数。调试时用 `--log_transcripts` 记录完整用户/助手转写（启动时会告警，因为它会把对话内容写入日志收集处）。

### VAD 参数

见 [VADHandlerArguments](./src/speech_to_speech/arguments_classes/vad_arguments.py)。`--thresh` 触发阈值、`--min_speech_ms` 最小语音时长、`--min_speech_continuation_ms` 续接语音的迟滞阈值（推荐与 `--min_speech_ms 384` 配 `192`）、`--min_silence_ms` 静音分段长度、`--short_segment_merge_ms` 相邻短段合并窗口、`--speculative_reopen_ms` / `--unanswered_reopen_ms` 软结束回合的补开窗。

### 唤醒词参数（本分支新增）

见 [LocalAudioArguments](./src/speech_to_speech/arguments_classes/local_audio_arguments.py)。服务端侧参数以 `--local_audio_...` 为前缀，客户端侧 `talk` 用别名：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--wake-word`（即 `--local_audio_wake_word`） | `hey_jarvis` | 启用本地 openWakeWord 门控并指定唤醒词 |
| `--wake-word-timeout`（即 `--local_audio_wake_word_timeout_s`） | `300` | 唤醒后允许语音的秒数（5 分钟） |
| `--wake-ack` | `嗯哼，您说` | 唤醒识别后固定播报的应答文本 |

> 关闭唤醒门控：显式传空串，如 `--wake-word ""`（客户端仅在 `wake_word` 非空时创建唤醒门控）。

### STT / LLM / TTS 参数

每个 STT / TTS 实现暴露 `model_name`、`torch_dtype`、`device` 等，使用 handler 前缀，如 `--stt_model_name`、`--qwen3_tts_device`。LLM 模型选择与对话设置跨后端共享无前缀标志（`--model_name`、`--chat_size`）；后端特定标志用 `responses_api_` 前缀。其他生成参数可用 handler 前缀加 `_gen_` 追加，例如 `--stt_gen_max_new_tokens 128`、`--llm_gen_temperature 0.7`。

## Citations 引用

### 上游项目

本仓库基于 Hugging Face 的 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 项目。使用或分发本分支时，请同时引用上游原项目：

```bibtex
@misc{speech-to-speech,
  title        = {{Speech To Speech: Build voice agents with open-source models}},
  author       = {{Hugging Face}},
  year         = {2025},
  howpublished = {\url{https://github.com/huggingface/speech-to-speech}},
  note         = {Apache 2.0 licensed}
}
```

### 组件模型

如果你使用本流水线，也请引用你所运行的组件模型。默认配置为：

#### Silero VAD

```bibtex
@misc{SileroVAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  email = {hello@silero.ai}
}
```

#### Parakeet TDT

```bibtex
@misc{parakeet-tdt,
  author = {NVIDIA},
  title = {Parakeet TDT 0.6B v3},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3}}
}
```

#### Qwen3-TTS

```bibtex
@misc{qwen3-tts,
  author = {Qwen Team},
  title = {Qwen3-TTS},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}}
}
```

以下可选后端的引用见对应组件 README（`src/speech_to_speech/STT` / `TTS`）：Kokoro、Pocket TTS、ChatTTS、Whisper 系列、Paraformer、MMS、Supertonic、OmniVoice，以及唤醒词 [openWakeWord](https://github.com/dscripka/openWakeWord) 与回声抑制 [WebRTC Audio Processing](https://github.com/descriptinc/webrtc-audio-processing)。

## License 许可

Apache License 2.0。AEC3 原生适配层（`native/aec3`）内置的 `webrtc-audio-processing` 采用其各自的许可证（MPL/BSD 许可组件），请在使用本分支时遵循对应上游的许可要求。