# 长期记忆

长期记忆让助手在新对话中仍能了解你。默认实现使用 Markdown 文件，无需单独部署数据库。

## 记住、查看与删除

可以直接说：

- “记住，我现在住在杭州。”
- “你记住了我哪些信息？”
- “把我的住址改成苏州。”
- “忘掉那条住址记录。”

用户偏好写入 `<data-dir>/USER.md`，事实和决定写入 `<data-dir>/MEMORY.md`。新建对话不会清空这些文件。对话修改立即生效；直接编辑文件后，在下一次语音会话生效。

清单和资料文档使用各自功能，不应写进长期记忆：见[清单与提醒](../guides/notes-reminders.zh.md)、[资料库](../guides/knowledge.zh.md)。

## 自动整理

默认 Markdown Provider 可在会话结束后用文本模型补记长期信息：明确的交互偏好写入 `USER.md`，稳定事实与决定写入 `MEMORY.md`。不会修改 `ASSISTANT.md`。

默认使用 DashScope `qwen-flash` 并复用 `DASHSCOPE_API_KEY`。没有可用 Key 时自动整理关闭，明确要求的记忆工具不受影响。可关闭自动整理，或配置另一个 OpenAI 兼容文本服务：

```dotenv
QWEN_AUDIO_MEMORY_AUTO=off
```

| 配置项 | 用途 |
| --- | --- |
| `QWEN_AUDIO_MEMORY_MODEL` | 自动整理使用的文本模型 |
| `QWEN_AUDIO_MEMORY_BASE_URL` | OpenAI 兼容服务地址 |
| `QWEN_AUDIO_MEMORY_API_KEY` | 该服务的凭据 |

自动整理会额外产生模型用量。它只学习新对话，不重复学习重连恢复的历史；通过工具或客户端成功编辑记忆后，过期学习结果不会把旧内容写回来。自动整理仍可能记错，请定期查看并纠正。

每个至少有一条新用户发言的会话结束后都会尝试增量提取，不再要求四条发言或等待 30 分钟。提取时读取完整档案；新的明确纠正使用精确编辑替换旧事实，而不是追加相互矛盾的条目。同一用户的提取按顺序处理，版本冲突最多重读重试一次。工具和客户端的明确修改仍可即时执行；后台提取不是每轮实时分析。

### 容量整理

默认开启 `QWEN_AUDIO_MEMORY_COMPACTION=on`。会话结束并完成提取、偏好晋升后检查容量；通过工具或客户端成功修改记忆后也会检查：

| 文档 | 整理阈值 | 模型注入预算 |
| --- | --- | --- |
| `USER.md` | 4500 字符 | 6000 字符 |
| `MEMORY.md` | 6000 字符 | 8000 字符 |

可用 `QWEN_AUDIO_USER_COMPACTION_THRESHOLD_CHARS` 和 `QWEN_AUDIO_MEMORY_COMPACTION_THRESHOLD_CHARS` 调整阈值。超过阈值后尝试合并重复记录、精简事实表述，目标为阈值的 75%；只有结果缩短且回到阈值内才写入。标题和分区保留，所有原记录必须被结果覆盖；用户要求和推断标签保留原文，不允许整理模型自行裁定事实冲突。

整理读取磁盘完整内容，不从已截断的注入快照反向覆盖文档。提交仍检查版本，用户并发修改优先。输出无效、遗漏来源或无法安全缩短时保留原文并记录诊断，不强制删除尾部。因此，过长且不可改写的要求仍可能超过注入预算；这种情况需要用户精简。关闭自动学习或缺少文本模型凭据时，后台整理不运行。不会整理 `ASSISTANT.md` 或其他项目 Markdown。

来源覆盖校验不能证明模型的概括在语义上完全无损。因此每次压缩提交前，原文保存为同目录的 `USER.md.pre-compaction` 或 `MEMORY.md.pre-compaction`，仅保留上一版、权限为 `0600`，不会自动恢复或参与聊天注入。备份失败则不压缩。后续通过 Provider 成功修改该文档时（包括工具、API 和增量提取），删除对应旧备份；直接手改文件不会触发这项清理，需要一并处理备份。

## 会话回溯

默认随 Gateway 开启。会话结束时，新的用户发言用于增量生成话题、一句要点及最多 400 字符的具体经历摘要，独立保存在 `<state-dir>/session-digests.json`。每个用户最多保留 60 场，保存 90 天。关闭方式：

```dotenv
QWEN_AUDIO_SESSION_DIGEST=off
```

可以问“前几天我们聊过的那个项目是什么？”摘要按需检索，不是完整录音或逐字聊天记录。

`recall` 对话题、经历摘要和关联工作做关键词与中文双字词片段相关性排序，默认最多返回 5 场，不做向量语义检索。所有经历不会一起塞进系统提示词；没有匹配时不猜测。重连恢复的旧消息不重复摘要，同一场有新的发言时更新原摘要，不重复追加。

摘要可记录曾经派发的工作，但不保存一份永久不变的工作状态。可用状态来自当前任务台账；记录已过期时，只能回顾聊过或做过什么，不能据此确认最新进展。

## VoiceMem 与自定义 Provider

可选 [VoiceMem](../scenarios/voicemem.zh.md) 能接管记忆、召回与会话学习。它需要单独安装，不包含在核心 npm 包中：

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=voicemem
VOICEMEM_PYTHON=/absolute/path/to/python
VOICEMEM_SIDECAR=/absolute/path/to/voicemem-sidecar.py
VOICEMEM_INPUT_MODE=text
```

`text` 使用已有转写；`audio` 把按轮次截取的用户音频交给 VoiceMem。切换 Provider 不会自动迁移或删除另一套记忆，请先备份。

开发接口见 [Memory Provider](memory-provider.zh.md)。

## 隐私与诊断

相关记忆会进入语音模型上下文；自动整理还会把对话交给配置的文本服务。不要保存密码、API Key、验证码或令牌。

`<state-dir>/memory-audit.jsonl` 保存学习处理的诊断信息。分享日志前仍需检查内容；常规日志目录和轮转见[本地日志](../configuration/advanced.zh.md#本地日志)。
