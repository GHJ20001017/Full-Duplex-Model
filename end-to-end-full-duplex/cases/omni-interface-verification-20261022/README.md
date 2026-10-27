# Qwen2.5-Omni 三个关键接口验证

本报告对应 `run-01/report.json` 的实测结果。没有训练、修改模型权重、解包全量数据或修改现有 Python 环境。

## 环境和输入

- 95 上已下载的 `Qwen2.5-Omni-3B`，不是 7B。
- 单卡 NVIDIA A800-SXM4-80GB，物理 GPU 5；其他进程未被停止。
- PyTorch 2.12.0、Transformers 5.6.0、Accelerate 1.6.0、NumPy 2.5.3。
- BF16 权重，eager attention，完全离线加载。
- 模型加载约 4.50 秒；本次测试的峰值 PyTorch allocated CUDA memory 约 11.34 GiB，不是训练显存预算。
- 输入来自 `behavior-sd/test/0000.tar::0000000005.flac` 的首句话、channel 0。
- 输入只用于验证，不加入训练；从 22,050 Hz 重采样为处理器要求的 16,000 Hz。
- 记录了配置、权重索引和安装实现的 SHA-256，见原始 JSON。

## 1. 基础推理：接口通过，质量仅单样本检查

原始标注：

> Hi, Brion. It’s surprising we’ve never met before.

模型转写：

> Hi! It's surprising we've never met before.

漏掉了人名，因此不能把此次结果表述成完全准确的 ASR。正常输出非空文本，主要句意一致；文本生成耗时约 0.99 秒。这不是标准 WER/CER 基准测试。

同一音频作为输入，另行请求模型回复并输出语音，返回：

> Yeah, it is. But it's also kind of cool that we met. So, what's your name?

保存为 `run-01/response.wav`，24 kHz、5.1 秒；波形有限且 RMS 约 0.117。整个文本加音频调用约 7.28 秒，不是首音频延迟，也不是在线流式播放延迟。

模型没有严格遵守“一句短回复”的指令，输出了多句。接口通过不代表指令遵循、声音质量、语音文本一致性或全双工能力已通过。

## 2. 生成期间增量音频：未通过未修改原生接口

实际运行了两次 Thinker 缓存生成：控制组和生成途中注入新音频特征的组。两组初始输入完全相同，每组生成 8 个 token。

- 初始步骤均正常传入音频特征。
- 从第二个生成步骤起，注入组在调用原生 `prepare_inputs_for_generation` 前提供新的 100 ms 音频特征。
- 该接口在有缓存的后续步骤将 `input_features` 改为 `None`，新音频没有进入后续 forward。
- 两组音频编码器均只调用 1 次，两组输出 token ID 完全相同。
- 原生方法源码已保存为 `run-01/prepare_inputs_for_generation.txt`。

结论只适用于本次检查的已安装实现及未改造的缓存生成接口。该测试没有实现或验证自定义双流调度，也没有证明模型结构无法改造成全双工。

### 音频编码因果性附加检查

对相同长度的 Mel 特征做干预：保留前 10 帧，置零后面的第 10 到第 199 帧，再分别执行音频编码。

最前面两个音频 embedding 的最大绝对差约为 4.666。说明该输入块内早期表示依赖后续特征，不能把完整块离线编码后切片当作严格因果的 100 ms 输入。

这是特征级干预测试，不是完整波形级流式等价测试，也没有据此给出精确的最小流式延迟。

## 3. FLAC 到原生 Talker token：本地接口阻塞

检查了本地权重索引、加载后的 Thinker/Talker/Token2Wav 对象和已安装实现：

- 有 Thinker 连续音频感知编码器。
- 有 Talker 的 codec embedding/head 和 Token2Wav 生成链路。
- 这些对象没有可调用的 `encode(waveform)` 接口。
- 本地权重索引未找到匹配 speech/audio tokenizer、quantizer 或 codec encoder 的权重名。
- 没有找到可验证为与当前 Talker 码本兼容的真实音频编码入口，因此没有生成真实 FLAC 对应的训练标签。

基础推理已实际跑通“模型生成 token 到波形”，但这不是“真实音频到 token”的逆向验证。这里的阻塞结论限定于本地检查的资产，不声称外部不存在兼容编码器。

`run-01/generated_talker_codes.npy` 是在生成时捕获的原始 Talker 返回序列，包含前缀/特殊 token，首部有 mask token 8296；它不是从输入或目标 FLAC 编码得到的真值，也不是可直接作为训练标签的纯 codec 序列。

## 文件和复现

- `run-01/report.json`：机器可读报告和运行时轨迹。
- `run-01/input.wav`：实际用于验证的短音频。
- `run-01/response.wav`：实际生成的回复语音。
- `run-01/generated_talker_codes.npy`：原始生成序列，仅用于检查。
- `run-01/prepare_inputs_for_generation.txt`：本次使用的增量准备方法源码。
- 项目脚本：`script/verify_omni_interfaces.py`。

在具有模型依赖的服务器环境运行，选择有足够空闲显存的单卡，并指定一个尚无报告的新输出目录：

```bash
CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
python script/verify_omni_interfaces.py \
  --model-dir /path/to/Qwen2.5-Omni-3B \
  --dataset-dir /path/to/behavior-sd \
  --output-dir /path/to/new-validation-run
```

复现脚本不安装依赖；默认已有完整官方资产，且恰好暴露一张 CUDA 卡。它会拒绝覆盖已有 `report.json`。诊断流程 exit 0 仅表示流程执行完毕，各接口是否通过以报告中各自的 `status` 为准。

本次 base 环境在导入部分可选依赖时出现了 NumPy 1.x/2.x ABI 告警，但模型加载和上述推理实际完成。未全局升级或降级依赖；后续正式开发应使用单独、固定版本且消除告警的环境。

## 后续阻塞

1. 增量音频输入与因果编码需要定制，不能直接使用当前标准 `generate()` 实现双流训练/推理。
2. 原生 Talker 的真实语音标签路径需要先找到兼容编码器并完成验证；在此之前不能声称数据集可直接用于原生音频 token CE 训练。
3. 本次没有训练行为 token，也没有验证双流融合、100 ms 实时性、打断停播或自由生成质量。
