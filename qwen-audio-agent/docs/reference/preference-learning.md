# Preference Learning

With `QWEN_AUDIO_PREFERENCE_LEARNING=on`, the Gateway observes user traits from a
finished session and writes them to `USER.md` only after enough cross-session
confirmation. It is off by default because it adds one model call per session.

The base switch observes four fields with deliberately narrow value spaces. It
does not implicitly enable companionship learning:

| Field | Meaning |
| --- | --- |
| `occupation` | Occupation |
| `special_skills` | Technologies or domains the user is strong in, capped at 6 |
| `response_length` | Reply length; only `brief`, `normal`, or `detailed` |
| `response_style` | Reply style |

Writes land in the `## 观察推断` (observed) section of `USER.md`, kept
**physically separate** from `## 用户明确要求` (explicitly stated). An explicit
statement always wins on conflict. The split prevents inferred content from
polluting what the user wrote: the user can see which lines came from their own
words and which the system guessed, and can edit or delete the latter.

### Promotion gate

An observation reaches the document only when `confirm ≥ 2` **and** the
confirmations come from **≥ 2 distinct sessions**. `confirm` resets after 90 days
with no new confirmation.

### Companionship Preferences

`QWEN_AUDIO_COMPANIONSHIP_LEARNING` defaults to `on`, so the Gateway automatically
enables these three contextual response dimensions at startup. No extra setting
or base preference switch is required. Set it to `off` to disable learning. When both are
enabled they share one observer call. This requires `QWEN_AUDIO_MEMORY_AUTO=on`
(the default) and a working `QWEN_AUDIO_MEMORY_API_KEY` or `DASHSCOPE_API_KEY`.
`QWEN_AUDIO_MEMORY_MODEL` and `QWEN_AUDIO_MEMORY_BASE_URL` configure the text model.
Providers owning session observation continue to own learning; the built-in
observer is not layered on top of them.

| Field | Values | Scope |
| --- | --- | --- |
| `support_style` | `listen_first`, `advice_first`, `ask_first` | Listening, actionable advice, or asking which support the user wants when they share a concern |
| `humor_style` | `no_jokes`, `gentle`, `light_teasing` | Humor boundaries; positive humor preferences apply only to light conversation |
| `checkin_style` | `no_unsolicited`, `occasional`, `contextual` | No unsolicited follow-up, occasional care, or moderate natural follow-up on a relevant topic during an ongoing conversation |

These are response preferences, not personality, health, or attachment scores.
Priority is **current explicit request > explicit standing preference > inference
> default assistant style**. Support preferences do not change ordinary task
answers. A humor preference never calls for teasing during serious distress.

Promotion requires valid feedback in **at least two distinct sessions**, with
**at least 24 hours between the first and last supporting observations**. The same
value counts once per session. Waiting alone is not evidence. A changed enum value
restarts confirmation, as does a 90-day evidence gap. Processing happens at session
close; promoted preferences enter the next session's instructions.

The first implementation conservatively validates **Chinese feedback**. For
example, repeated “你先听我说，别急着给我建议” can support `listen_first`.
“今天你别拿我开玩笑” is temporary and is rejected even if the model quotes only
the part without “今天”: validation checks the containing user turn.
“以后我倾诉时先听我说” instead takes the existing explicit-memory path without
waiting for inferred promotion.

Evidence must come from one user turn and support the specific enum value.
Temporary requests, quoted dialogue, third-party statements, obvious questions,
and sarcasm are excluded. Ambiguous approval such as “that was better” is not
resolved against generated assistant text: generated, interrupted, or unplayed
output is not evidence of an experienced and approved reply. Silence, conversation
length, and continuing to talk are not positive feedback. These guards are not a
complete semantic parser; complex phrasing may be missed. Explicit standing
instructions remain available.
Each session with at least one fresh user message can trigger observation.
Short sessions still require the same cross-session evidence, 24-hour interval
and verbatim-feedback checks before companionship preferences are promoted.

Check-in preferences affect **reply content only**. There is no new proactive
speech scheduler, timed or turn-count check-in, reminder creation, or out-of-session
contact. The feature lives in Gateway memory and text decisions and works with a
cascaded ASR → LLM → TTS stack. It neither changes TTS nor recognizes vocal emotion.

Disabling the switch stops observation and pending promotion but does not delete
saved preferences. Inspect and edit them through the existing `GET /api/memory`
and revision-checked `PATCH /api/memory`, or edit `USER.md`. Successful API/tool
edits discard pending candidates and invalidate in-flight learning. Direct file
edits do not reset the candidate pool; disable learning first if needed. Evidence
is capped at three 50-character quotes with a short rationale. Owners are isolated
by `ownerId`; the default personal identity is not speaker recognition, so a
multi-user deployment must supply separate authenticated owner identities.

### Four structural guards

A model sometimes supplies a genuine quote while the inference from that quote
does not hold. Repeated sampling cannot filter this class out — the user says the
same sentence every session, the model makes the same wrong inference, and the
counter climbs to the threshold anyway. The guards therefore apply at admission
time:

| Guard | What it blocks |
| --- | --- |
| `quote_not_from_user` | The quote must appear verbatim in a user turn: blocks fabricated evidence, assistant speech treated as user preference, and self-reinforcement |
| `value_not_anchored` | The literal parts of the conclusion must be findable in the quote |
| `value_parrots_quote` | Value equals the quote — that is repetition, not feature extraction |
| `quote_not_about_interaction` | For interaction-preference fields the quote must address the assistant: blocks "how long the content should be" being read as "how long the reply should be" |

Diagnostics land in `memory-audit.jsonl`, so a rejected observation can be
explained after the fact.

See [Personalization](personalization.md) for user settings and [Memory Provider](memory-provider.md) for integration boundaries.
