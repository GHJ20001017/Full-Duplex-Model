# Full-Duplex Model

Cascaded Full-Duplex Model and End-to-End Full-Duplex Model.

本项目包含两条全双工语音模型路线：

- `cascaded-full-duplex/`：级联式全双工模型，迁移自 `TTS/speech-to-speech`，保留其 STT → LLM → TTS 的实时流水线。
- `end-to-end-full-duplex/`：端到端全双工模型训练工程，输入音频流，直接建模输出音频流，训练代码与数据处理流程在此目录中逐步实现。

两条路线相互独立，分别维护依赖、配置、实验日志和模型检查点。
