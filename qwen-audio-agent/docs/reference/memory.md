# Long-Term Memory

Long-term memory helps the assistant remember you across conversations. The default implementation uses Markdown files without a separate database.

## Remember, Inspect, and Delete

Try:

- “Remember that I now live in Hangzhou.”
- “What have you remembered about me?”
- “Change my home city to Suzhou.”
- “Forget that address.”

Preferences are stored in `<data-dir>/USER.md`; facts and decisions in `<data-dir>/MEMORY.md`. A new conversation does not clear these files. Conversation edits apply immediately; direct file edits apply in the next voice session.

Lists and reference documents have separate features: [Lists and Reminders](../guides/notes-reminders.md), [Knowledge Library](../guides/knowledge.md).

## Automatic Extraction

After a session ends, the default Markdown provider can use a text model to save missing long-term information: explicit interaction preferences in `USER.md` and stable facts or decisions in `MEMORY.md`. It cannot change `ASSISTANT.md`.

The default is DashScope `qwen-flash` using `DASHSCOPE_API_KEY`. Without an available key, automatic extraction is disabled; explicit memory-tool requests remain available. Disable extraction or configure another OpenAI-compatible text service:

```dotenv
QWEN_AUDIO_MEMORY_AUTO=off
```

| Setting | Purpose |
| --- | --- |
| `QWEN_AUDIO_MEMORY_MODEL` | Text model for extraction |
| `QWEN_AUDIO_MEMORY_BASE_URL` | OpenAI-compatible service endpoint |
| `QWEN_AUDIO_MEMORY_API_KEY` | Credentials for that service |

Extraction incurs additional model usage. It learns only from new conversation, not replayed history. After a successful memory edit through a tool or client, stale learning results cannot restore the old content. Extraction can still make mistakes; review and correct memories periodically.

Every session with at least one fresh user message attempts incremental extraction, without the former four-message minimum or 30-minute debounce. Full documents are read for reconciliation. Explicit corrections replace old facts with exact edits. Extraction is serialized per owner; a revision conflict gets one fresh-read retry. Explicit tools and client edits remain immediate; background extraction is not per-turn analysis.

### Capacity Maintenance

`QWEN_AUDIO_MEMORY_COMPACTION=on` is the default. After extraction and preference promotion, and after successful explicit memory edits, maintenance checks `USER.md` at 4500 characters and `MEMORY.md` at 6000 characters. Configure these using `QWEN_AUDIO_USER_COMPACTION_THRESHOLD_CHARS` and `QWEN_AUDIO_MEMORY_COMPACTION_THRESHOLD_CHARS`; injection budgets remain 6000 and 8000 characters respectively.

The model targets 75% of the threshold by merging duplicates and shortening facts. Every source record must be covered; headings, sections, explicit requirements and inferred labels are preserved. Maintenance cannot decide factual conflicts without new user evidence. Only shorter results within the threshold are committed, with revision checks. Full disk content is read, never a truncated injection snapshot.

Invalid, incomplete or unsafe results leave the original untouched and record diagnostics. Requirements too long to safely condense can still exceed the injection budget and need manual editing. Disabling automatic memory or lacking text-model credentials disables background maintenance. `ASSISTANT.md` and other project Markdown files are not compacted.

Source coverage does not prove semantic equivalence. Before committing compaction, the previous full document is saved beside it as `USER.md.pre-compaction` or `MEMORY.md.pre-compaction`, with mode `0600`. Only one version is retained; it is never injected or automatically restored. A backup failure prevents compaction. Any subsequent successful provider edit, including tools, the API and incremental extraction, removes that document's old backup. Direct file edits do not trigger backup cleanup.

## Recall Earlier Conversations

Enabled by default with the Gateway. Fresh user messages produce topics, a short gist and a bounded experience summary of up to 400 characters in `<state-dir>/session-digests.json`. Each owner retains up to 60 sessions for 90 days. To disable:

```dotenv
QWEN_AUDIO_SESSION_DIGEST=off
```

Ask “What was the project we discussed a few days ago?” Summaries are retrieved on demand; they are not recordings or verbatim transcripts.

`recall` ranks keyword and Chinese-bigram overlap across topics, experience summaries and related work, returning up to five sessions by default. It does not use vector semantic search or inject the entire archive into system instructions. Restored history is not summarized again; fresh messages in an existing session update that session's summary rather than append a duplicate.

A summary may record earlier work, but does not freeze its status. Available status comes from the current task ledger. Once a record expires, recall can describe what was discussed or requested, not verify current progress.

## VoiceMem and Custom Providers

Optional [VoiceMem](../scenarios/voicemem.md) takes over memory, retrieval, and session learning. Install it separately; the core npm package does not include it:

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=voicemem
VOICEMEM_PYTHON=/absolute/path/to/python
VOICEMEM_SIDECAR=/absolute/path/to/voicemem-sidecar.py
VOICEMEM_INPUT_MODE=text
```

`text` reuses transcription; `audio` sends turn-scoped user audio to VoiceMem. Switching providers does not migrate or delete the other provider's data. Back up first.

See [Memory Provider](memory-provider.md) for developer interfaces.

## Privacy and Diagnostics

Relevant memories become voice-model context. Automatic extraction also sends conversation to the configured text service. Do not store passwords, API keys, verification codes, or tokens.

`<state-dir>/memory-audit.jsonl` contains learning diagnostics. Review logs before sharing. See [Local Logs](../configuration/advanced.md#local-logs) for standard log locations and rotation.
