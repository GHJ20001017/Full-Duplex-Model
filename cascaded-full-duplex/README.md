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
* [Linux 与 Windows 部署](#linux-与-windows-部署)
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

**这是可选客户端功能，不随 Python 3.12 主 requirements 安装。** Linux 的 `tflite-runtime` 没有 CPython 3.12 wheel，即使使用 ONNX 推理，安装 openWakeWord 时也会遇到其依赖解析限制。主环境运行 `talk` / `local` 时显式传 `--wake-word ""`，这不改变服务端 semantic 话轮模式。

需要唤醒词时，在独立 Python 3.11 环境安装项目的 `wake-word` extra，而不是在主清单中取消该条目注释。以下 Bash 命令从 `cascaded-full-duplex` 执行；服务端继续使用主环境，客户端仍需按平台构建并配置 AEC3：

```bash
conda create -n speech_to_speech_wake python=3.11 pip -y
conda activate speech_to_speech_wake
(cd .. && python -m pip install -e './cascaded-full-duplex[wake-word]')
python -m pip check
python -c 'from openwakeword.utils import download_models; download_models(model_names=["hey_jarvis"])'

python -m speech_to_speech.cli talk --wake-word hey_jarvis --wake-word-timeout 120 \
    --url ws://127.0.0.1:7869/v1/realtime
```

这是独立客户端环境，不读取 Python 3.12 主清单；仍须按实际平台验证依赖解析与音频设备。换用 `hey_computer` 等模型时需相应下载模型并更改 `--wake-word`。

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

> 根目录 requirements 已包含 Paraformer 依赖，详见 [支持的组件](#支持的组件-supported-components)。

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

使用 Conda（Miniconda 或 Miniforge）创建 Python 3.12 环境。以下在 `cascaded-full-duplex` 目录执行，安装子 shell 先切换到仓库根目录；若已在根目录，直接运行 `python -m pip install -r requirements.txt`。清单中的 editable 项目路径相对当前工作目录解析，不能仅修改 `-r` 文件路径。

```bash
conda create -n speech_to_speech_system python=3.12 pip -y
conda activate speech_to_speech_system
(cd .. && python -m pip install -r requirements.txt)
python -m pip check
```

根目录 [`requirements.txt`](../requirements.txt) 是统一的 Python 安装入口，以 editable 方式安装本项目及 `paraformer,fun-asr-nano,webrtc`，并包含 `uvicorn[standard]`、`huggingface-hub[oauth]`、ModelScope。基础后端仍由项目元数据按平台选择：Parakeet STT、OpenAI 兼容 API、Qwen3-TTS、本地音频及 Realtime 服务器。

### 可选组件

Kokoro（非 macOS）、Pocket、ChatTTS、OmniVoice、Faster Whisper、Lightning Whisper MLX、MLX 视觉模型及 Supertonic 等依赖集中在根 requirements 的注释分组。先取消所需包条目的注释，再在 `cascaded-full-duplex` 执行 `(cd .. && python -m pip install -r requirements.txt)`；仅重复安装未修改的清单不会安装注释项，不要把所有可选后端一起安装。CLI 名称 `pocket`、`chattts`、`whisper-mlx` 分别对应清单中的 `pocket-tts`、`ChatTTS`、`lightning-whisper-mlx`。Pocket 要求 NumPy >=2，需先按清单说明移除 Linux `numpy==1.26.4` pin，且不与 DeepFilterNet 共用环境。VoiceMem、LightRAG 和 benchmark 目录仅为独立环境的依赖索引，不属于主环境。唤醒词使用上文独立 Python 3.11 extra 安装步骤，不在 Python 3.12 主环境启用。

已废弃的实现（含 MeloTTS 等）存放在 [`archive/`](./archive)，不再接入 CLI。

> **DeepFilterNet 注意**：DeepFilterNet 用于 VAD 的可选音频增强，要求 `numpy<2`，与 Pocket TTS（要求 `numpy>=2`）冲突，仅在不用 Pocket TTS 的环境里手动安装。

### 从源码构建

```bash
git clone https://github.com/GHJ20001017/Full-Duplex-Model.git
cd Full-Duplex-Model/cascaded-full-duplex
conda create -n speech_to_speech_system python=3.12 pip -y
conda activate speech_to_speech_system
(cd .. && python -m pip install -r requirements.txt)
```

（macOS 上构建 AEC3 原生库：`./native/aec3/build_macos.sh`。）

## Linux 与 Windows 部署

本节从本仓库源码安装 Paraformer + Qwen3-TTS，并连接 OpenAI 兼容的 Chat Completions API。需要 Conda 和 Python 3.12及可用的 LLM API 地址、模型名称和密钥。已有仓库时跳过克隆，直接进入 `cascaded-full-duplex` 目录。macOS 快速上手见[项目首页](../README.md#安装与启动)。

以下步骤依据仓库依赖和后端实现整理，尚未在 Linux／Windows 完成安装及语音联调；请同时阅读对应平台限制。

### Linux + NVIDIA GPU（Ubuntu 24.04 示例）

先安装 NVIDIA 驱动，确认 `nvidia-smi` 能显示显卡。以下使用 Bash；其他发行版需要替换系统包安装命令。

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential python3-dev \
  portaudio19-dev ffmpeg meson ninja-build pkg-config

git clone https://github.com/GHJ20001017/Full-Duplex-Model.git
cd Full-Duplex-Model/cascaded-full-duplex
conda create -n speech_to_speech_system python=3.12 pip -y
conda activate speech_to_speech_system
(cd .. && python -m pip install -r requirements.txt)

# 按根 requirements 中的平台说明选择匹配 wheel 后检查 CUDA
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动与 PyTorch wheel"; print(torch.cuda.get_device_name(0))'
```

PyTorch wheel 自带所需的 CUDA 运行时，但不包含显卡驱动；常规 wheel 推理通常无需另装完整 CUDA Toolkit。主清单将 `torch` 与 `torchaudio` 同时固定为 `2.11.0`。若需不同 CUDA／XPU 构建，请先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 核对驱动、索引及配对版本，在独立平台环境中明确覆盖主清单的两项 pin 后安装，不要单独升级其中一个包。平台覆盖不是默认安装步骤；再次安装未修改的主清单会恢复其配对版本。安装后运行 `python -m pip check` 并重新检查 CUDA。

**Linux GGML 是可选项：** 主依赖安装 `faster-qwen3-tts`，不再强制安装 `[ggml]` 或 `qwentts-cpp-python`。本文 NVIDIA 启动命令显式使用 `--qwen3_tts_backend torch`，无需 GGML wheel；底层 CLI 默认仍为 GGML，直接启动时也应显式选 Torch。只有另选 GGML 后端时，才需要按 [TTS 依赖说明](src/speech_to_speech/TTS/README.md)安装与 CUDA／glibc 匹配的可选 wheel；上游 CUDA 12.8 wheel 的 `manylinux_2_39` 限制不再是主环境的强制前提。

**仅使用 S2S 修改版客户端时：构建 Linux AEC3。** 仓库目前只提供 macOS 自动构建脚本，下面是依据同一原生适配器整理的 Linux 手工构建步骤（尚未实机验证），不要运行 `build_macos.sh`：

```bash
mkdir -p .native-build/aec3 native/aec3/build
# 已有该源码目录时跳过 clone
git clone --depth 1 https://github.com/okarlsen/webrtc-audio-processing.git \
  .native-build/aec3/webrtc-audio-processing
meson setup .native-build/aec3/meson-build \
  .native-build/aec3/webrtc-audio-processing \
  --prefix "$PWD/.native-build/aec3/install" --libdir lib --buildtype release
ninja -C .native-build/aec3/meson-build
ninja -C .native-build/aec3/meson-build install
export PKG_CONFIG_PATH="$PWD/.native-build/aec3/install/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
c++ -std=c++17 -O3 -fPIC -shared native/aec3/aec3_wrapper.cc \
  $(pkg-config --cflags --libs webrtc-audio-processing-2) \
  -Wl,-rpath,"$PWD/.native-build/aec3/install/lib" \
  -o native/aec3/build/libs2s_aec3.so
export S2S_AEC3_LIBRARY="$PWD/native/aec3/build/libs2s_aec3.so"
```

重复配置已有 Meson 构建目录时，在 `meson setup` 后加 `--reconfigure`。上述动态库记录了依赖的绝对路径，移动仓库后需重新构建。无桌面／无音频设备的 GPU 服务器仅运行服务端，将客户端放在有麦克风和扬声器的电脑上；仅运行服务端或 Qwen Audio Agent 不需要构建此 AEC3 库。

### Windows + NVIDIA GPU（PowerShell）

先安装 NVIDIA 驱动并检查 `nvidia-smi`。以下使用原生 PowerShell，不是 Bash；安装工具后需重新打开终端，使 PATH 生效。

```powershell
winget install --id Git.Git -e
winget install --id Gyan.FFmpeg -e
```

另行安装 Miniconda 或 Miniforge，并在已初始化 Conda 的 PowerShell 新终端执行：

```powershell
git clone https://github.com/GHJ20001017/Full-Duplex-Model.git
cd Full-Duplex-Model/cascaded-full-duplex
conda create -n speech_to_speech_system python=3.12 pip -y
conda activate speech_to_speech_system
Push-Location ..
try {
  python -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }
} finally { Pop-Location }
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
```

若 `conda activate` 尚不可用，先按 Conda 安装说明初始化 PowerShell 并重开终端。CUDA wheel 的驱动要求与 Linux 相同；按根 requirements 的平台说明及 PyTorch 官方选择器选择适配版本，安装后运行 `python -m pip check`。

**Windows 边界：** 项目为 Windows 声明了不带 GGML 的 `faster-qwen3-tts` 依赖，而运行时默认仍为 GGML，因此下文显式指定 `--qwen3_tts_backend torch`。这是一条待实机验证的服务端安装路线，不代表全部依赖和音频功能已验证兼容。原生 S2S 客户端还需要符合本仓库 C ABI 的 `s2s_aec3.dll`，仓库没有 Windows 构建脚本或预编译 DLL，不能把 `.dylib`／`.so` 改名使用，也没有可跳过 AEC3 的现成启动选项。**Windows 优先使用[首页第 3.2 节 Qwen Audio Agent 的 WebUI](../README.md#32-qwen-audio-agent-客户端) 连接服务端**；不需要为它构建本地 S2S AEC3 或下载唤醒词模型。

也可以在 **WSL2 + Ubuntu 24.04** 中按 Linux 路线部署服务端；需先确认 WSL 内 `nvidia-smi` 和 PyTorch CUDA 检查成功。麦克风／扬声器客户端放在 Windows 侧，不将 WSL 音频透传作为默认前提。Windows 到 WSL 的本机连接先尝试 `ws://localhost:7869/v1/realtime`；网络模式需要非回环监听时，按下文服务端启动说明的受控网络要求配置，而不是直接对公网开放。

### 启动 NVIDIA 服务端

在 `cascaded-full-duplex` 目录中新开终端，替换下列占位值。`LLM_BASE_URL` 是 API 基地址，不是完整的 `/chat/completions` 路径；密钥只放本地环境变量，不要写入仓库。

**Linux + NVIDIA（Bash）：** 使用 CUDA 运行 ASR 和 TTS，显式选择 Torch TTS 后端。

```bash
conda activate speech_to_speech_system
export OPENAI_API_KEY="替换为你的 API 密钥"
export LLM_BASE_URL="https://你的服务域名/v1"
export LLM_MODEL="替换为该服务实际提供的模型名称"

speech-to-speech serve \
  --host 127.0.0.1 --port 7869 \
  --stt paraformer --paraformer_stt_model_name paraformer-zh-streaming \
  --paraformer_stt_device cuda --enable_live_transcription true \
  --llm_backend chat-completions --responses_api_base_url "$LLM_BASE_URL" \
  --model_name "$LLM_MODEL" \
  --tts qwen3 --qwen3_tts_backend torch --qwen3_tts_device cuda \
  --qwen3_tts_language Chinese \
  --init_chat_prompt "你是 ARVIS，一个实时语音助手。请用简洁、自然的中文回答。"
```

**Windows + NVIDIA（PowerShell）：** 用 `$env:` 设置环境变量，用反引号换行（反引号后不能有空格），不能直接复制 Bash 的 `export` 和反斜杠续行。

```powershell
conda activate speech_to_speech_system
$env:OPENAI_API_KEY = "替换为你的 API 密钥"
$env:LLM_BASE_URL = "https://你的服务域名/v1"
$env:LLM_MODEL = "替换为该服务实际提供的模型名称"

speech-to-speech serve `
  --host 127.0.0.1 --port 7869 `
  --stt paraformer --paraformer_stt_model_name paraformer-zh-streaming `
  --paraformer_stt_device cuda --enable_live_transcription true `
  --llm_backend chat-completions --responses_api_base_url "$env:LLM_BASE_URL" `
  --model_name "$env:LLM_MODEL" `
  --tts qwen3 --qwen3_tts_backend torch --qwen3_tts_device cuda `
  --qwen3_tts_language Chinese `
  --init_chat_prompt "你是 ARVIS，一个实时语音助手。请用简洁、自然的中文回答。"
```

首次启动需要下载并加载模型，请等待监听完成后连接。默认仅监听本机 `ws://127.0.0.1:7869/v1/realtime`；跨机器访问需显式设置 `--host 0.0.0.0`，并将客户端 URL 改为服务端实际地址。接口自身无访问鉴权，只能通过受控网络或带鉴权的 TLS 网关访问，不要直接暴露到公网。

### 连接客户端

保留服务端运行。Linux 使用原生 S2S 客户端时，先完成上述 AEC3 构建，再在有音频设备的电脑上打开另一终端。主 Python 3.12 环境关闭唤醒门控；需要唤醒词时另按[独立环境说明](#2-本地服务端唤醒词wake-word)安装：

```bash
conda activate speech_to_speech_system
export S2S_AEC3_LIBRARY="$PWD/native/aec3/build/libs2s_aec3.so"
speech-to-speech talk \
  --url ws://127.0.0.1:7869/v1/realtime \
  --wake-word ""
```

Windows 未准备兼容 AEC3 DLL 时，使用 [Qwen Audio Agent WebUI](../README.md#32-qwen-audio-agent-客户端)。两种客户端连接同一个服务端，切换时断开旧语音会话；默认只提供一个流水线实例。

## 快速开始 Quickstart

### 服务端 + 独立客户端

```bash
# 终端 1：启动服务器（主环境不安装 GGML wheel）
export OPENAI_API_KEY=...
speech-to-speech serve --qwen3_tts_backend torch

# 终端 2：本地麦克风/扬声器客户端（需已构建 AEC3，关闭可选唤醒词）
speech-to-speech talk --wake-word "" --url ws://127.0.0.1:7869/v1/realtime
```

服务器监听 `ws://localhost:7869/v1/realtime`。直接对麦克风开始对话；助手播放回答时会经过 AEC3 回声消除。这些底层 CLI 示例不替代首页的 semantic 服务启动流程。

### 一条命令本地运行

```bash
speech-to-speech local --wake-word "" --qwen3_tts_backend torch
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
| STT | [Fun-ASR-Nano 800M](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)（累计音频渐进转写 + 热词） | PyTorch（默认 CUDA，可配置 CPU） | `fun-asr-nano` |
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
| 唤醒词 | [openWakeWord](https://github.com/dscripka/openWakeWord)（本地 ONNX） | 依赖与音频设备需按平台验证 | 独立 Python 3.11 环境安装 `wake-word` extra；主清单不包含 |

用 `--stt`、`--llm_backend`、`--tts` 选择具体实现。CLI 只构造所选后端的配置；未激活后端的已知选项仍被接受但会忽略并告警。下列省略 TTS 选项的示例在非 macOS 主环境运行时，应追加 `--qwen3_tts_backend torch`；CLI 的 GGML 默认值不代表主清单安装了其可选 wheel。

> 流式 ASR 说明：`parakeet-tdt` 默认启用智能渐进式转写；`paraformer` 默认用流式中文模型且流水线只发送新增音频块。二者都要求启用实时转写开关（`--enable_live_transcription`，见 [Realtime API](#realtime-api)）。

### Fun-ASR-Nano 与热词文件

Paraformer 保持原有配置；通过 `--stt fun-asr-nano` 切换到独立的 Nano 后端。两者都使用 FunASR 的 `AutoModel` 加载和 `generate` 推理，Nano 不依赖 vLLM。

根 requirements 已包含 Nano 依赖。尚未安装时，在本仓库 `cascaded-full-duplex` 目录执行：

```bash
(cd .. && python -m pip install -r requirements.txt)
```

准备自己的 UTF-8 热词文件，例如 `/absolute/path/hotwords.txt`：

```text
# 每行一个词或短语，不使用 :权重 格式
全双工
语义打断
DeepSeek
Fun ASR Nano
```

在原有启动命令中替换 STT 参数，保留原有 LLM、TTS 和服务器配置：

```bash
speech-to-speech serve \
  --stt fun-asr-nano \
  --fun_asr_nano_stt_model_name FunAudioLLM/Fun-ASR-Nano-2512 \
  --fun_asr_nano_stt_device cuda \
  --fun_asr_nano_stt_hotwords_file /absolute/path/hotwords.txt \
  --enable_live_transcription
```

- 仅 Nano 读取该文件，启动时加载一次：忽略空行和以 `#` 开头的注释行，去掉首尾空白并去重，保留短语内部空格。支持 UTF-8 BOM。未提供文件、文件不可读或没有有效词条时启动失败，不静默忽略；修改词表后需重启服务。
- 每次推理都通过复数参数 `hotwords` 传入词表。这是模型上下文引导，不是带权重的解码热词偏置，也不是识别后替换。
- 默认 hub 为 `hf`，可用 `--fun_asr_nano_stt_hub ms` 选择 ModelScope；模型参数也接受完整本地目录。加载遵循官方用法设置 `trust_remote_code=True`，只使用可信模型来源。
- `--fun_asr_nano_stt_language` 支持 `中文`（默认）、`英文`、`日文`；`--fun_asr_nano_stt_itn false` 可关闭文本规整。
- 开启实时转写时，Nano 对 VAD 提供的累计音频快照独立重识别，输出可修订的完整 partial；句末对完整音频生成 final。不复用旧文字前缀，不把完整结果追加成重复文本，也不启用 Paraformer 的增量块缓存模式。未开启实时转写时只做句末识别。
- 这是渐进式转写而非缓存式原生流式编码。长句会重复计算；首字延迟、热词准确率和与 TTS 并跑的资源占用需在部署机器上实测。

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