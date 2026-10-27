import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { ConversationSync } from '../src/conversation/conversation-sync.mjs'
import { buildMemoryContext } from '../src/memory/context.mjs'
import { createMemoryModule } from '../src/memory/module.mjs'
import { COMPANIONSHIP_FIELDS } from '../src/memory/learning/companionship-preferences.mjs'
import { MemoryExtractor } from '../src/memory/learning/extractor.mjs'
import { PreferenceCandidatePool, renderLabel, isSameFieldLabel } from '../src/memory/learning/preference-candidates.mjs'
import { PreferenceCandidateStore } from '../src/memory/learning/preference-candidate-store.mjs'
import { PreferencePromoter, splitObservedSection } from '../src/memory/learning/preference-promoter.mjs'
import { ProfileObserver } from '../src/memory/learning/profile-observer.mjs'
import { MarkdownContextStore } from '../src/memory/providers/markdown/context-store.mjs'
import { MarkdownMemoryProvider } from '../src/memory/providers/markdown/provider.mjs'

const DAY = 24 * 60 * 60_000
const START = Date.parse('2026-10-01T09:00:00Z')
const logger = { warn() {}, debug() {} }
const examples = [
  ['support_style', 'listen_first', '你先听我说，别急着给我建议'],
  ['support_style', 'advice_first', '我倾诉时你直接给我建议，我觉得更有帮助'],
  ['support_style', 'ask_first', '你先问我想要倾听还是建议，我觉得更舒服'],
  ['humor_style', 'no_jokes', '你别拿我开玩笑，我不喜欢'],
  ['humor_style', 'gentle', '我喜欢你开温和的玩笑'],
  ['humor_style', 'light_teasing', '我能接受你轻轻调侃我'],
  ['checkin_style', 'no_unsolicited', '你不用主动问我近况，我想聊时自己说'],
  ['checkin_style', 'occasional', '你偶尔问我后来怎么样，我觉得很贴心'],
  ['checkin_style', 'contextual', '聊天时你可以多问问我最近的进展'],
]

function fixture(t, { enabled = true, profile = true, directory,
  initial = '# USER', reply } = {}) {
  let clock = START
  const root = directory || mkdtempSync(join(tmpdir(), 'companion-learning-'))
  if (!directory) t.after(() => rmSync(root, { recursive: true, force: true }))
  const store = new PreferenceCandidateStore({ filePath: join(root, 'candidates.json') })
  const pool = new PreferenceCandidatePool({ store, now: () => clock,
    companionshipEnabled: enabled, profileLearningEnabled: profile })
  const provider = new MarkdownMemoryProvider({
    userStore: new MarkdownContextStore({ filePath: join(root, 'USER.md'), scope: 'user',
      personalOwnerId: 'u1', template: initial, maxChars: 6000 }),
    memoryStore: new MarkdownContextStore({ filePath: join(root, 'MEMORY.md'), scope: 'memory',
      personalOwnerId: 'u1', template: '# MEMORY', maxChars: 6000 }),
  })
  if (!provider.stores.user.readRaw('u1')) provider.stores.user.persist('u1', initial)
  const audit = []
  const calls = []
  const observer = new ProfileObserver({ candidatePool: pool,
    llmCall: async input => { calls.push(input); return reply ?? '{"answers":[]}' },
    audit: { record: entry => audit.push(entry) }, logger })
  const promoter = new PreferencePromoter({ memoryService: provider, candidatePool: pool, logger,
    audit: { record: entry => audit.push(entry) } })
  return { root, store, pool, provider, observer, promoter, audit, calls,
    advance: ms => { clock += ms },
    document: (ownerId = 'u1') => provider.list(ownerId, { scope: 'user' })[0] }
}

function answer(field, value, quote) {
  return { field, value, quote, basis: '用户对助手的直接反馈', vs_known: 'first_time' }
}

async function observe(f, rows, userTurns, assistantTurns = [], sessionId = 's1', ownerId = 'u1') {
  f.observer.llmCall = async input => {
    f.calls.push(input)
    return JSON.stringify({ answers: rows })
  }
  return f.observer.run({ ownerId, sessionId, messages: [
    ...userTurns.map(content => ({ role: 'user', content })),
    ...assistantTurns.map(content => ({ role: 'assistant', content })),
  ] })
}

for (const [field, value, quote] of examples) {
  test(`accepts supported ${field}=${value} with a contextual fixed label`, async t => {
    const f = fixture(t)
    const accepted = await observe(f, [answer(field, value, quote)], [quote])
    assert.equal(accepted.length, 1)
    const [slot] = f.pool.list('u1')
    assert.equal(slot.value, value)
    assert.equal(slot.state, 'tentative')
    assert.equal(slot.evidence[0].quote, quote)
    assert.equal(slot.evidence[0].basis, '用户对助手的直接反馈')
    assert.equal(isSameFieldLabel(field, slot.label), true)
    assert.deepEqual(slot.missing, ['confirm', 'sessions', 'time_span'])
    assert.deepEqual(await f.promoter.run({ ownerId: 'u1' }), [])
  })
}

const rejected = [
  ['temporary full turn, shortened quote', 'humor_style', 'no_jokes', '你别拿我开玩笑', '今天你别拿我开玩笑'],
  ['temporary support request', 'support_style', 'listen_first', '你先听我说', '现在你先听我说'],
  ['standing instruction uses explicit path', 'support_style', 'listen_first', '先听我说', '以后我倾诉时先听我说'],
  ['quoted task content', 'humor_style', 'gentle', '我喜欢你开温和的玩笑', '帮我写一段台词：我喜欢你开温和的玩笑'],
  ['third party', 'humor_style', 'gentle', '我喜欢你开温和的玩笑', '朋友说：我喜欢你开温和的玩笑'],
  ['ambiguous feedback', 'support_style', 'listen_first', '这样好多了', '这样好多了'],
  ['silence is not approval', 'humor_style', 'light_teasing', '嗯', '嗯'],
  ['duration is not approval', 'checkin_style', 'contextual', '我们聊了两个小时', '我们聊了两个小时'],
  ['neutral unrelated user request', 'support_style', 'advice_first', '你帮我看一下代码', '你帮我看一下代码'],
  ['opposite enum', 'support_style', 'advice_first', '你先听我说', '你先听我说'],
  ['invalid enum', 'humor_style', 'unlimited', '我喜欢你开温和的玩笑', '我喜欢你开温和的玩笑'],
  ['negated preference, shortened quote', 'support_style', 'listen_first', '先听我说', '我不喜欢你先听我说'],
  ['question, shortened quote', 'humor_style', 'gentle', '我喜欢你开温和的玩笑', '我喜欢你开温和的玩笑吗？'],
  ['sarcasm', 'humor_style', 'gentle', '我喜欢你开温和的玩笑', '我喜欢你开温和的玩笑，这是反话'],
  ['no-jokes question', 'humor_style', 'no_jokes', '你别拿我开玩笑', '你别拿我开玩笑？'],
  ['negated boundary, shortened quote', 'humor_style', 'no_jokes', '你别拿我开玩笑', '我不希望你别拿我开玩笑'],
  ['task advice is not emotional support', 'support_style', 'advice_first', '你直接给我建议', '这段代码你直接给我建议'],
  ['medical context is not durable evidence', 'support_style', 'listen_first', '你先听我说我的病历', '你先听我说我的病历'],
]
for (const [name, field, value, quote, user] of rejected) {
  test(`does not learn ${name}`, async t => {
    const f = fixture(t)
    assert.deepEqual(await observe(f, [answer(field, value, quote)], [user]), [])
    assert.deepEqual(f.pool.list('u1'), [])
    assert.equal(f.audit.some(entry => entry.op === 'skip'), true)
  })
}

test('assistant-only and invented feedback never count, even if output was interrupted', async t => {
  const f = fixture(t)
  const quote = '我喜欢你开温和的玩笑'
  assert.deepEqual(await observe(f, [answer('humor_style', 'gentle', quote)],
    ['继续', '好的'], [quote]), [])
  assert.doesNotMatch(f.calls[0].user, new RegExp(quote))
  assert.deepEqual(f.pool.list('u1'), [])
})

test('a quote cannot be stitched across user turns', async t => {
  const f = fixture(t)
  assert.deepEqual(await observe(f, [answer('humor_style', 'gentle', '我喜欢你开温和的玩笑')],
    ['我喜欢你', '开温和的玩笑']), [])
  assert.equal(f.audit.at(-1).reason, 'quote_not_from_single_user_turn')
})

test('companionship can be independently disabled, including model output under the old style field', async t => {
  const f = fixture(t, { enabled: false })
  const quote = '我喜欢你开温和的玩笑'
  assert.deepEqual(await observe(f, [answer('humor_style', 'gentle', quote),
    answer('response_style', '喜欢温和的玩笑', quote)], [quote]), [])
  assert.equal(f.pool.observe({ ownerId: 'u1', field: 'humor_style', value: 'gentle' }), null)
  assert.doesNotMatch(f.observer.knownProfile('u1'), /humor_style|support_style|checkin_style/)
  assert.doesNotMatch(f.calls[0].system, /- humor_style：/)
})

test('all seven field answers survive the old five-answer cap; duplicates count only once', async t => {
  const f = fixture(t)
  const unknown = ['occupation', 'special_skills', 'response_style', 'response_length']
    .map(field => answer(field, 'unknown', ''))
  const rows = [examples[0], examples[4], examples[7]].map(parts => answer(...parts))
  const accepted = await observe(f, [...unknown, ...rows, rows[0]], rows.map(row => row.quote))
  assert.equal(accepted.length, 3)
  assert.equal(f.pool.list('u1').every(slot => slot.confirm === 1), true)
  assert.equal(f.audit.some(entry => entry.reason === 'duplicate_companionship_field'), true)
  assert.match(f.calls[0].system, /- checkin_style：/)
  assert.doesNotMatch(f.calls[0].system, /四个问题|四个字段都要出现/)
})

test('requires feedback across sessions and a full day, not merely time passing', t => {
  const f = fixture(t)
  const observation = { ownerId: 'u1', field: 'support_style', value: 'listen_first' }
  f.pool.observe({ ...observation, sessionId: 's1' })
  f.pool.observe({ ...observation, sessionId: 's1' })
  assert.equal(f.pool.list('u1')[0].confirm, 1)
  f.advance(60_000)
  f.pool.observe({ ...observation, sessionId: 's2' })
  assert.deepEqual(f.pool.list('u1')[0].missing, ['time_span'])
  f.advance(DAY)
  assert.deepEqual(f.pool.promotable('u1'), [], 'time alone cannot invent later feedback')
  f.pool.observe({ ...observation, sessionId: 's3' })
  assert.equal(f.pool.promotable('u1').length, 1)
  assert.equal(f.pool.diagnose('u1').fields.support_style[0].minSpanMs, DAY)
})

test('contradicting enums restart evidence even when the model calls them a refinement', t => {
  const f = fixture(t)
  const observation = { ownerId: 'u1', field: 'support_style', value: 'listen_first' }
  f.pool.observe({ ...observation, sessionId: 's1' })
  f.advance(DAY)
  f.pool.observe({ ...observation, sessionId: 's2' })
  assert.equal(f.pool.promotable('u1').length, 1)
  f.pool.observe({ ...observation, value: 'advice_first', relation: 'refine', sessionId: 's3' })
  const [slot] = f.pool.list('u1')
  assert.equal(slot.confirm, 1)
  assert.equal(slot.firstAt, slot.lastAt)
  assert.deepEqual(slot.sessions, ['s3'])
  assert.deepEqual(f.pool.promotable('u1'), [])
})

test('stale companionship evidence restarts its time window', t => {
  const f = fixture(t)
  const observation = { ownerId: 'u1', field: 'humor_style', value: 'gentle' }
  f.pool.observe({ ...observation, sessionId: 's1' })
  f.advance(91 * DAY)
  f.pool.observe({ ...observation, sessionId: 's2' })
  f.advance(1000)
  f.pool.observe({ ...observation, sessionId: 's3' })
  assert.deepEqual(f.pool.list('u1')[0].missing, ['time_span'])
})

test('persists, promotes per owner, replaces conflicting inferred labels, and honors rejection', async t => {
  const f = fixture(t)
  for (const [field, value, quote] of [examples[0], examples[4], examples[7]]) {
    f.pool.observe({ ownerId: 'u1', sessionId: 's1', field, value, quote })
  }
  f.advance(DAY)
  for (const [field, value, quote] of [examples[0], examples[4], examples[7]]) {
    f.pool.observe({ ownerId: 'u1', sessionId: 's2', field, value, quote })
  }
  const restarted = fixture(t, { directory: f.root })
  assert.equal(restarted.pool.promotable('u1').length, 3)
  assert.equal((await restarted.promoter.run({ ownerId: 'u1' })).length, 3)
  assert.equal(splitObservedSection(restarted.document().content).items.length, 3)
  assert.doesNotMatch(restarted.document('u2')?.content || '', /倾诉支持|玩笑边界|会话内关心/)
  assert.deepEqual(await restarted.promoter.run({ ownerId: 'u1' }), [])

  const replacement = { ownerId: 'u1', field: 'humor_style', value: 'no_jokes' }
  restarted.advance(2 * DAY)
  restarted.pool.observe({ ...replacement, sessionId: 's3' })
  restarted.advance(DAY)
  restarted.pool.observe({ ...replacement, sessionId: 's4' })
  await restarted.promoter.run({ ownerId: 'u1' })
  assert.match(restarted.document().content, /不主动对用户开玩笑/)
  assert.doesNotMatch(restarted.document().content, /轻松聊天时可用温和玩笑/)
  await restarted.promoter.reject(replacement)
  assert.doesNotMatch(restarted.document().content, /玩笑边界/)
  assert.equal(restarted.pool.observe({ ...replacement, sessionId: 's5' }), null)
  assert.equal(fixture(t, { directory: f.root }).pool.blocked('u1', 'humor_style', 'gentle'), true)
})

test('switching learning off prevents saved tentative candidates from promoting', async t => {
  const f = fixture(t)
  f.pool.observe({ ownerId: 'u1', sessionId: 's1', field: 'humor_style', value: 'gentle' })
  f.advance(DAY)
  f.pool.observe({ ownerId: 'u1', sessionId: 's2', field: 'humor_style', value: 'gentle' })
  const disabled = fixture(t, { directory: f.root, enabled: false })
  assert.equal(disabled.pool.list('u1').length, 1, 'disabling does not erase user data')
  assert.deepEqual(await disabled.promoter.run({ ownerId: 'u1' }), [])
})

test('explicit wording owns its dimension and is not overwritten by a contrary inference', async t => {
  const initial = '# USER\n## 用户明确要求\n- 以后我倾诉时先听我说\n'
  const f = fixture(t, { initial })
  f.pool.observe({ ownerId: 'u1', sessionId: 's1', field: 'support_style', value: 'advice_first' })
  f.advance(DAY)
  f.pool.observe({ ownerId: 'u1', sessionId: 's2', field: 'support_style', value: 'advice_first' })
  assert.deepEqual(await f.promoter.run({ ownerId: 'u1' }), [])
  assert.match(f.document().content, /以后我倾诉时先听我说/)
  assert.doesNotMatch(f.document().content, /偏好先给可执行建议/)
})

for (const [profile, companionship] of [[false, false], [true, false], [false, true], [true, true]]) {
  test(`module switches remain independent: profile=${profile}, companionship=${companionship}`, async t => {
    const f = fixture(t)
    const module = createMemoryModule({ config: { preferenceLearningEnabled: profile,
      companionshipLearningEnabled: companionship }, memoryProvider: f.provider,
    logger, conversationSync: new ConversationSync(), textModelCall: async () => '{"changes":[]}' })
    t.after(() => module.close())
    const pool = module.services.preferenceCandidates
    if (!profile && !companionship) return assert.equal(pool, null)
    assert.equal(pool.allows('occupation'), profile)
    assert.equal(pool.allows('support_style'), companionship)
  })
}

test('injected guidance retains current-request priority and never authorizes unsolicited speech', t => {
  const f = fixture(t, { initial: `# USER\n## 观察推断\n- ${renderLabel('humor_style', 'gentle')}` })
  const context = buildMemoryContext({ memories: [f.document()] })
  assert.match(context, /用户当前明确请求 > 用户明确的长期要求 > 观察推断/)
  assert.match(context, /不因玩笑偏好而打趣/)
  assert.match(context, /不凭偏好创建提醒、定时问候、会话外联系/)
  assert.match(context, /不抢话或打断任务/)
})

test('explicit extraction supports companionship and keeps it out of the observed section', async t => {
  const f = fixture(t, { initial: '# USER\n## 观察推断\n- 回答简短，直接说要点' })
  const extractor = new MemoryExtractor({ memoryService: f.provider, logger,
    llmCall: async () => JSON.stringify({ changes: [{ document: 'user', edits: [],
      append: '- 用户倾诉时先倾听，不急于给建议' }] }) })
  await extractor.run({ ownerId: 'u1', messages: [{ role: 'user', content: '以后我倾诉时先听我说' }] })
  const parts = splitObservedSection(f.document().content)
  assert.match(parts.before, /## 用户明确要求\n- 用户倾诉时先倾听/)
  assert.deepEqual(parts.items, ['回答简短，直接说要点'])
})

for (const user of ['今天你别拿我开玩笑', '你别拿我开玩笑，我不喜欢', '朋友说以后你别拿我开玩笑']) {
  test(`explicit extraction cannot bypass learning gates: ${user}`, async t => {
    const f = fixture(t)
    const audit = []
    const extractor = new MemoryExtractor({ memoryService: f.provider, logger,
      audit: { record: entry => audit.push(entry) },
      llmCall: async () => JSON.stringify({ changes: [{ document: 'user', edits: [],
        append: '- 不要主动拿我开玩笑' }] }) })
    await extractor.run({ ownerId: 'u1', messages: [{ role: 'user', content: user }] })
    assert.doesNotMatch(f.document().content, /不要主动拿我开玩笑/)
    assert.equal(audit.at(-1).reason, 'companionship_directive_not_explicit')
  })
}

test('manual deletion invalidates in-flight companionship feedback for that user', async t => {
  const f = fixture(t, { initial: '# USER\n- 以后回答简短一些' })
  const sync = new ConversationSync()
  const deferred = Promise.withResolvers()
  const module = createMemoryModule({ config: { companionshipLearningEnabled: true },
    logger, conversationSync: sync, memoryProvider: f.provider, textModelCall: () => deferred.promise })
  t.after(() => module.close())
  const scope = { ownerId: 'u1', sessionId: 's1' }
  for (let index = 0; index < 4; index++) sync.record({ ...scope, id: String(index),
    role: 'user', source: 'voice-user', content: examples[0][2] })
  const pending = module.services.profileObserver.maybeRun(scope)
  const memory = module.services.frontendMemory
  const document = memory.list('u1', { scope: 'user' })[0]
  await memory.apply('u1', [{ document: 'user', expectedRevision: document.revision,
    edits: [{ old_text: '- 以后回答简短一些', new_text: '' }] }], { source: 'gateway-memory-api' })
  deferred.resolve(JSON.stringify({ answers: [answer(...examples[0])] }))
  assert.deepEqual(await pending, [])
  assert.deepEqual(module.services.preferenceCandidates.list('u1'), [])
})

test('session-close pipeline learns across days and injects only the right owner\'s preference', async t => {
  const f = fixture(t)
  const sync = new ConversationSync()
  let clock = START
  const module = createMemoryModule({ config: { companionshipLearningEnabled: true },
    logger, conversationSync: sync, memoryProvider: f.provider, textModelCall: async ({ system }) => (
      system.includes('你维护语音助手') ? '{"changes":[]}'
        : JSON.stringify({ answers: [answer(...examples[0])] })
    ) })
  t.after(() => module.close())
  const pool = module.services.preferenceCandidates
  pool.now = () => clock
  const close = async (ownerId, sessionId, content = examples[0][2]) => {
    for (let index = 0; index < 4; index++) sync.record({ ownerId, sessionId,
      id: `${sessionId}-${index}`, role: 'user', source: 'voice-user', content })
    await module.sessionObservers[0].onSessionClosed({ ownerId, sessionId, logger })
  }
  await close('u1', 'day-one')
  assert.equal(pool.list('u1')[0].confirm, 1)
  assert.doesNotMatch(f.document().content, /倾诉支持/)
  await close('u2', 'other-owner', '帮我看代码')
  assert.deepEqual(pool.list('u2'), [])
  clock += DAY
  await close('u1', 'day-two')
  assert.match(f.document().content, /## 观察推断[\s\S]*倾诉支持/)
  const context = buildMemoryContext({ memories: module.services.frontendMemory.list('u1') })
  assert.match(context, /用户倾诉时先倾听和回应感受/)
  assert.match(context, /用户当前明确请求 > 用户明确的长期要求/)
  assert.doesNotMatch(buildMemoryContext({ memories: module.services.frontendMemory.list('u2') }), /倾诉支持/)
  await module.sessionObservers[0].onSessionClosed({ ownerId: 'u1', sessionId: 'day-two', logger })
  assert.equal(pool.list('u1')[0].confirm, 2, 'duplicate close is not new evidence')
})

test('sensitive content in a model rationale cannot be persisted as evidence', async t => {
  const f = fixture(t)
  const row = { ...answer(...examples[0]), basis: '密码是 123456' }
  assert.deepEqual(await observe(f, [row], [examples[0][2]]), [])
  assert.deepEqual(f.pool.list('u1'), [])
})

test('a provider owning observation is not given a second companionship learner', t => {
  const f = fixture(t)
  const description = f.provider.describe()
  f.provider.describe = () => ({ ...description, key: 'custom',
    capabilities: { ...description.capabilities, sessionObservation: true } })
  f.provider.observe = async () => ({ observed: true })
  const module = createMemoryModule({ config: { companionshipLearningEnabled: true },
    logger, memoryProvider: f.provider, textModelCall: async () => '{"answers":[]}' })
  t.after(() => module.close())
  assert.equal(module.services.profileObserver, null)
  assert.equal(module.services.preferencePromoter, null)
})

test('companionship field definitions are closed and limited to three dimensions', () => {
  assert.deepEqual(Object.keys(COMPANIONSHIP_FIELDS), ['support_style', 'humor_style', 'checkin_style'])
  for (const definition of Object.values(COMPANIONSHIP_FIELDS)) {
    assert.equal(definition.values.length, 3)
    assert.equal(definition.minSpanMs, DAY)
  }
})
