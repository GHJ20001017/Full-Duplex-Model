# Assistant Profile and Preferences

Set a default assistant style and save your own preferences through conversation. They are stored separately, and upgrades do not overwrite your local profile.

## Set Preferences in Conversation

- “Call me Captain from now on.”
- “Keep your replies concise from now on.”
- “When we talk, your name is River.”

Explicit long-term requests become user preferences. “Keep this reply short” is temporary and should not become a lasting setting.

## Edit the Default Profile

On first start, a template creates `<config-dir>/ASSISTANT.md`. Edit it to change the default name, personality, relationship, and communication style. Changes apply to the next voice session.

To use another file:

```dotenv
QWEN_AUDIO_AGENT_ASSISTANT_PROFILE_PATH=/absolute/path/to/ASSISTANT.md
```

The profile does not select the speaking voice; configure that under [voice frontend settings](../configuration/frontend.md). It cannot override tool permissions, task routing, or core runtime rules.

## Where Settings Live

| Content | Default file | Example |
| --- | --- | --- |
| Default assistant profile | `<config-dir>/ASSISTANT.md` | Default name and style |
| Explicit long-term preferences | `<data-dir>/USER.md` | “Call me Captain” |
| Long-term facts and decisions | `<data-dir>/MEMORY.md` | “I live in Hangzhou” |

The current request takes precedence over saved preferences; preferences can override the default profile. Facts support understanding, not permission grants or execution instructions.

Changes made through conversation apply immediately. Direct file edits apply in the next voice session. Do not store passwords, keys, or verification codes here. See [data directories](../configuration.md#configuration-and-data-directories).

## Optional Preference Learning

Base profile learning is off by default. Set `QWEN_AUDIO_PREFERENCE_LEARNING=on` to let the default memory provider observe occupation, skills, reply length, and reply style after sessions and write cross-session confirmations into the inferred section of `USER.md`.

This is off by default and makes additional text-model calls. Explicit preferences take precedence over inferences. You can delete unwanted inferences from the file. See [Preference Learning](preference-learning.md) for thresholds and implementation details.

Companionship preference learning starts automatically with the Gateway by default,
adapting support style, humor boundaries, and contextual check-ins to a particular
user. No environment setting is needed. To disable it, set this and restart the Gateway:

```dotenv
QWEN_AUDIO_COMPANIONSHIP_LEARNING=off
```

This needs the automatic-memory text-model credentials.
Repeated concrete feedback must span different sessions and at least 24 hours
before becoming an inference. Temporary requests are not standing preferences.
Explicit long-term requests still use ordinary memory, and the current request
always takes precedence. Humor is contextual, not appropriate during distress
just because the user usually enjoys jokes. Check-in preferences affect replies
in an ongoing conversation, not timed interruptions or out-of-session outreach.

Turning learning off does not erase saved preferences. Use the existing
[memory controls](memory.md) to inspect, correct, or delete them. See
[Companionship Preferences](preference-learning.md#companionship-preferences)
for values, evidence guards, and cascaded-speech limitations.

## Privacy and Other Providers

Files stay on the Gateway host, but relevant profile, preference, and memory content is sent as context when using cloud models. Local storage does not mean fully offline processing.

See [Memory](memory.md) for automatic extraction, inspection, and deletion. Use [VoiceMem](../scenarios/voicemem.md) or implement a [Memory Provider](memory-provider.md) to replace storage and learning.
