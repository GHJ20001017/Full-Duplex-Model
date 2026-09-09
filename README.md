# Full-Duplex Model

全双工语音模型工程，包含两条独立路线：

- `cascaded-full-duplex/`：级联式全双工语音智能体（VAD → STT → LLM → TTS），基于 Hugging Face [speech-to-speech](https://github.com/huggingface/speech-to-speech) 增强，通过 OpenAI Realtime 协议（WebSocket / WebRTC）对外服务。
- `end-to-end-full-duplex/`：端到端全双工模型训练工程，从交叠音频流直接建模输出音频流。

两条路线相互独立，分别维护依赖、配置、实验日志和模型检查点。

## Cascaded 关键点

针对真实双工对话的四个核心增强：

- **AEC3 回声抑制**：原生 WebRTC `EchoCanceller3`，麦克风音频在送入 VAD / 唤醒词 / 上传前做声学回声消除，避免助手声音被识别成用户输入。
- **本地服务端唤醒词**：客户端 openWakeWord 门控（默认 `hey jarvis`），未唤醒不上传音频；唤醒后服务端播报固定应答。
- **流式 ASR**：Parakeet 智能渐进式转写（每 500 ms 出部分结果、句子边界滑动窗口）+ 流式 Paraformer（中文）。
- **语义化 barge-in**：`TurnController` 区分附和音（`嗯`/`对`/`ok`，不打断）与明确打断短语（`停一下`/`等一下`，立即取消）。

默认 Realtime 端口为 **7869**（上游为 8765）。详见 [cascaded-full-duplex/README.md](cascaded-full-duplex/README.md)。

## End-to-End 关键点

目标是从用户与系统的交叠音频流直接预测系统响应音频流，支持低延迟流式推理。评估覆盖 WER、响应延迟、打断延迟、音频质量和端点连续性。详见 [end-to-end-full-duplex/README.md](end-to-end-full-duplex/README.md)。

## 引用

级联路线基于 Hugging Face 的 [speech-to-speech](https://github.com/huggingface/speech-to-speech) 项目，使用或分发时请同时引用上游原项目；组件模型引用见 `cascaded-full-duplex/README.md`。

## License

Apache License 2.0。
