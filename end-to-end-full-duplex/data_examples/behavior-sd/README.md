# behavior-sd: Real Case Inspection

这些文件来自 95 上已下载并校验的数据，没有全量解包，也没有修改原数据。

远端根目录：`/gpu3/guhj/Full-Duplex_Model/end-to-end-full-duplex/data/behavior-sd`。

## 文件

- `0000000001.raw.json`：`train/0000.tar`，附和与打断，完整原始 JSON。
- `0000000003.raw.json`：`train/0000.tar`，打断，完整原始 JSON。
- `0000000005.raw.json`：`test/0000.tar`，普通轮换，完整原始 JSON。
- `cases.inspect.json`：以上原始 JSON、音频信息和声道能量抽查结果。
- `views.preview.json`：自定义的数据构建预览，不是数据集原始格式，也不是最终模型输入。

三个样本实测均为 22,050 Hz、双声道 FLAC。单说话人区间的能量抽查显示：channel 0 对应 `speaker_idx=0`，channel 1 对应 `speaker_idx=1`。附和区间也核对了对侧声道活动。该抽查不能替代整个数据集的自动质量检查。

## Case 1: 普通轮换

来源：`test/0000.tar::0000000005.json`，44.203 秒。

指定 Carlyn / channel 0 为用户，Brion / channel 1 为助手。

| 时间 / 秒 | 用户轨 | 助手轨 |
|---|---|---|
| 0.000–4.609 | “Hi, Brion. It’s surprising we’ve never met before.” | 无该说话人的话语标注 |
| 4.609–4.947 | 轮换间隔 | 轮换间隔 |
| 4.947–12.421 | 无该说话人的话语标注 | “Hi, Carlyn. It is surprising, especially since we live in the same town. I feel like I’ve heard your name a few times.” |

用途：观察正常听说轮换和轮换间隔。此例来自 test，只用于检查，不要加入训练集。

## Case 2: 助手附和，但不抢话

来源：`train/0000.tar::0000000001.json`，92.188 秒。

指定 Rylea / channel 0 为用户，Shavon / channel 1 为助手。

- 用户话语：14.692562–20.546803 秒，“I'm sorry to hear that. If you want, I'm more than happy to listen and help if I can.”
- 助手附和：15.796190–16.455782 秒，“Mhm.”。
- 附和嵌在 `utterances[3].backchannels[0]`，不是独立的顶层 utterance。
- 该嵌套记录的宿主 speaker 是 0，但附和说话人是对侧 speaker 1；这个对应关系已用声道能量抽查。
- 助手附和结束后，用户继续说到 20.546803 秒。不要把这个重叠误标为抢话或让用户停止。

原始片段：

```json
{
  "speaker_idx": 0,
  "start_time": 14.692562358276644,
  "end_time": 20.546802721088437,
  "backchannels": [{
    "bc_idx": 0,
    "tts_text": "Mhm.",
    "start_time": 15.796190476190477,
    "end_time": 16.45578231292517
  }]
}
```

## Case 3: 用户打断助手

来源：`train/0000.tar::0000000003.json`，96.614 秒。

固定 Mattea / channel 1 为助手，Danna / channel 0 为用户。当前只保留这个角色方向，不构建助手主动打断用户的反向视图。

- 助手：0.000000–7.871565 秒，“I'm thinking about ways we can improve our team collaboration to boost productivity. We need to focus on”。
- 用户：7.419048–13.044218 秒，“Communication strategies that ensure everyone is on the same page and understands their roles clearly.”。
- 用户这句的 `uttr_type` 为 `interrupt`。
- 双方重叠：7.419048–7.871565 秒，约 453 ms。

- 7.419048 秒用户插话时，助手还在说。
- 7.871565 秒助手轨的这句结束，此后用户继续说。
- 这是从已有时间线得到的“被打断后结束话语”示例，不是额外标注出的动作原因或最优停止时间。
- 不能把它改造成“用户一开口，助手同一帧立刻静音”，否则改变了原始时序。
- 该样本的两次显式打断都由 speaker 0 发起，因此必须用 channel 0 作为用户轨；不能只检查当前短窗口而忽略后面的打断。

## 当前角色筛选规则

目标是普通轮换、助手附和、用户打断助手；不包含助手主动打断用户。以后需要时再单独构建助手纠错式打断数据。

| 整段对话的显式打断发起方 | 用户轨 | 助手轨 | 当前处理 |
|---|---|---|---|
| 仅 speaker 0 | channel 0 | channel 1 | 保留这个方向 |
| 仅 speaker 1 | channel 1 | channel 0 | 保留这个方向 |
| 两方都没有 | 可选任一方 | 另一方 | 先按助手内容适配性选择方向 |
| 两方都有 | 无法固定满足约束 | 无法固定满足约束 | 暂时排除整段 |

初筛依据是整段 `utterances[*].uttr_type == "interrupt"` 对应的 `speaker_idx`，不是顶层 `behaviors.interruptions` 的数值。通过初筛不等于标注完整：实际做训练集时还要审核“助手主话语在用户主话语尚未结束时开始”的未标注重叠，疑似主动抢话先隔离，不自动当作正常回答。

助手短暂附和仍然允许；用户给助手附和也不等于用户要求抢话。不能用“用户声道有声音就让助手停止”的统一标签。原始重叠和停止边界先保留，不凭空添加立即停说监督，也不把被打断后停止当作整段会话结束。

## 构建方式

建议以整段同步双轨和一份轻量 manifest 为基本单位，训练时按时间块读，不必把每个 100 ms 存成一个音频小文件。

```json
{
  "shard": "train/0000.tar",
  "audio_member": "0000000001.flac",
  "json_member": "0000000001.json",
  "sample_rate": 22050,
  "user_channel": 0,
  "assistant_channel": 1,
  "chunk_ms": 100
}
```

在这个采样率下，100 ms 是每个声道 2,205 个采样点。按同一个时间区间取用户块和助手块，保留真实重叠与静音，不要分别按句裁剪后重新拼接。

`views.preview.json` 展示了普通轮换、助手附和、助手被用户打断这三种视图的 100 ms 网格。角色筛选检查了整段对话的显式打断标注，而不只是展示窗口。`*_annotated_span_fraction` 是原始话语/附和区间覆盖当前块的比例，不是测得的 VAD，也不是音频 token。没有逐 100 ms 手工动作标签，仍然可以从连续助手轨得到每块的音频监督。

如果采用不带显式动作 token 的双流建模，普通回复、附和和被用户打断后停止可以先由助手轨本身表达，行为事件单独保留做采样和评估。当前不为助手设计或监督主动 `INTERRUPT_USER` 动作。若另外增加 `BACKCHANNEL`、`YIELD` 等 token，应标明这是自定义监督，而非原始数据自带的逐帧 token；并明确插入时刻和损失权重。

训练中必须另外明确因果延迟：读取完整用户 100 ms 块再预测同一时间段助手音频，会引入最多一个块的观察延迟。不要把这种离线同步训练直接解释为零延迟在线生成。音频 codec 的帧率也应与所选 codec 对齐，不能仅由这个时间网格决定。

## 文本与标签注意事项

- `tts_text` 只有句段级时间戳。没有词级对齐时，不要把一整句未来文本提前放进当前块的输入。
- 历史文本只能包含当前时刻已知的内容。预览文件保守地只纳入窗口开始前已结束的 utterance 和附和；这还不是流式 ASR 的实现。
- `narrative` 是整段叙事，不应直接作为在线对话历史输入，否则可能泄漏后续内容。
- `behaviors` 不是逐帧事件列表，也不要直接按字段数值当作事件次数。例如 case 1 的 `filler_words=0` 与 `statistics.num_filler_words=3` 同时存在。
- `num_turns`、`len(utterances)`、`statistics.num_utterances` 不是同一个计数口径。Case 1 分别为 10、17 和 `[5, 5]`。
- 如果要监督文本生成，可先用句段文本做离线辅助监督；要做时间同步的文本 token 流，需要补做词级/子词级对齐。
- 原始人名角色对话不天然等同于合规、会纠错的助手对话。纠错、拒绝不当打断、真实用户噪声和回声等能力仍需要额外验证或补数据。

本次只导出了少量 JSON 和检查结果，没有解包整套数据，也没有复制整段音频到本地。
