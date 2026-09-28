<p align="center">
  <img src="./arvis-brand-assistant-blue.png" alt="ARVIS — A Real-time Voice Intelligence System" width="800" />
</p>

## 项目介绍

· **ARVIS（A Real-time Voice Intelligence System）** 是一个面向自然对话与任务执行的开源实时语音智能系统。项目旨在让语音助手不仅能够“听懂再回答”，还能够在持续倾听中及时回应，处理对话中的插话、附和与打断，让人与 AI 的交流更接近日常交谈。

· **全双工语音交互**：让助手在说话的同时持续倾听，用户无需等待播报结束，就可以插话、补充或打断。当前采用级联式流程：麦克风采集音频，经 AEC3 回声抑制与本地唤醒门控后，进入语音活动检测（VAD）与流式语音识别（ASR），再由语言模型（LLM）生成回复，通过语音合成（TTS）流式播放；输入监听与输出播放并行运行，配合话轮控制协调双方发言。后续将推进端到端全双工模型训练，以 Qwen3 为语言骨干、Mimi 为音频编解码器，参考 Moshi 架构，探索直接对用户与助手音频流进行联合建模。

· **语义路由与话轮控制**：采用由 jev 训练的 **Qwen3-0.6B 意图识别模型**，结合助手已说出的内容与用户的实时转写，输出 `wait`（等待）、`continue`（继续说）、`yield`（让出话轮）等标签，为对话中的发言决策提供依据。相比仅凭音量或静音时长触发打断，语义路由关注用户“说了什么”，区分简短附和与真正的插话意图，让助手知道何时继续、何时等待、何时把话语权交还给用户。

· **语音驱动的 Agent 执行**：[Qwen Audio Agent](qwen-audio-agent/README.md) 连接实时对话与后台任务执行，可通过 **MCP** 接入 Computer Use 等工具，也可通过 **ACP 适配器**连接 Codex、Claude Code（CC）等 **Agent Harness**。用户可以用语音发起任务，在后台 Agent 执行期间继续交流、补充需求或查询进度，任务结果再返回当前对话，让语音助手从“回答问题”走向“执行任务”。

## 项目结构

- `cascaded-full-duplex/`：级联式全双工语音智能体（VAD → STT → LLM → TTS），基于 Hugging Face [speech-to-speech](https://github.com/huggingface/speech-to-speech) 增强，通过 OpenAI Realtime 协议（WebSocket / WebRTC）对外服务。
- `end-to-end-full-duplex/`：端到端全双工模型训练工程，探索从交叠音频流直接建模输出音频流。
- `qwen-audio-agent/`：语音智能体运行时，提供持续对话、后台任务编排与工具集成能力。

级联与端到端两条路线相互独立，分别维护依赖、配置、实验日志和模型检查点。

## Cascaded 关键点

针对真实双工对话的四个核心增强：

- **AEC3 回声抑制**：原生 WebRTC `EchoCanceller3`，麦克风音频在送入 VAD / 唤醒词 / 上传前做声学回声消除，避免助手声音被识别成用户输入。
- **本地服务端唤醒词**：客户端 openWakeWord 门控（默认 `hey jarvis`），未唤醒不上传音频；唤醒后服务端播报固定应答。
- **流式 ASR**：Parakeet 智能渐进式转写（每 500 ms 出部分结果、句子边界滑动窗口）+ 流式 Paraformer（中文）。
- **语义化 barge-in**：`TurnController` 区分附和音（`嗯`/`对`/`ok`，不打断）与明确打断短语（`停一下`/`等一下`，立即取消）。


https://github.com/user-attachments/assets/39a33a28-4189-48f3-a419-51ceb4859e0b


默认 Realtime 端口为 **7869**（上游为 8765）。详见 [cascaded-full-duplex/README.md](cascaded-full-duplex/README.md)。

## End-to-End 关键点

目标是从用户与系统的交叠音频流直接预测系统响应音频流，支持低延迟流式推理。评估覆盖 WER、响应延迟、打断延迟、音频质量和端点连续性。详见 [end-to-end-full-duplex/README.md](end-to-end-full-duplex/README.md)。

## 引用

级联路线基于 Hugging Face 的 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 项目，使用或分发时请同时引用上游原项目；组件模型引用见 `cascaded-full-duplex/README.md`。

## License

Apache License 2.0。
