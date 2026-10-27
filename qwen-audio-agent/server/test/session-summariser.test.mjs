import assert from 'node:assert/strict'
import test from 'node:test'
import { SessionDigestPool } from '../src/conversation/session-digest.mjs'
import { SessionSummariser } from '../src/conversation/session-summariser.mjs'
import { ConversationSync } from '../src/conversation/conversation-sync.mjs'

const OWNER = 'user_personal'

function transcript(userLines, assistantLines = []) {
  const messages = []
  userLines.forEach((content, index) => {
    messages.push({ role: 'user', content })
    if (assistantLines[index]) {
      messages.push({ role: 'assistant', content: assistantLines[index] })
    }
  })
  return messages
}

function harness({
  messages = transcript(['一', '二', '三', '四']),
  reply = '{"topics":["话题"],"gist":"聊了些事"}',
  audit = [],
  listSessionWork = null,
} = {}) {
  const digests = new SessionDigestPool({ now: () => Date.parse('2026-08-25T12:00:00Z') })
  const calls = []
  const sync = new ConversationSync()
  messages.forEach((message, index) => sync.record({ ownerId: OWNER, sessionId: 's_a',
    ...message, id: `message-${index}`, source: message.role === 'user' ? 'voice-user' : 'voice-assistant' }))
  const summariser = new SessionSummariser({
    digestPool: digests,
    conversationSync: sync,
    audit: { record: entry => audit.push(entry) },
    llmCall: async payload => {
      calls.push(payload)
      return typeof reply === 'function' ? reply(payload) : reply
    },
    logger: { warn() {}, debug() {} },
    listSessionWork,
  })
  return { digests, summariser, calls, audit, sync }
}

test('summarises a session into topics and a gist', async () => {
  const { summariser, digests } = harness({
    reply: '{"topics":["LOCOMO","压缩评测"],"gist":"对比五个方案，two_tier 最优"}',
  })
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.deepEqual(digest.topics, ['LOCOMO', '压缩评测'])
  assert.equal(digest.gist, '对比五个方案，two_tier 最优')
  assert.equal(digest.turns, 4, 'turns 记的是用户轮数')
  assert.equal(digests.search({ ownerId: OWNER, keyword: 'LOCOMO' }).length, 1)
})

test('sends both roles to the model but counts only user turns', async () => {
  const { summariser, calls } = harness({
    messages: transcript(
      ['帮我看看这个', '再改改', '可以了', '谢谢'],
      ['好的', '改好了', '很好', '不客气'],
    ),
  })
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  // 助手轮要给模型看，否则读不出这场在围绕什么展开
  assert.match(calls[0].user, /助手: 好的/)
  assert.match(calls[0].user, /用户: 帮我看看这个/)
})

test('skips a session without new user messages', async () => {
  const { summariser, calls } = harness({ messages: [] })
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.equal(calls.length, 0)
})

test('skips a session it already recorded', async () => {
  const { summariser, digests, calls } = harness()
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.equal(calls.length, 1)
  // 重连会让同一场触发两次关闭钩子
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.equal(calls.length, 1, '不该为同一场再调一次模型')
  assert.equal(digests.count(OWNER), 1)
})

test('records nothing when the model finds no topic', async () => {
  const { summariser, digests, audit } = harness({ reply: '{"topics":[],"gist":""}' })
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.equal(digests.count(OWNER), 0)
  assert.ok(audit.some(entry => entry.reason === 'no_topic'))
})

test('swallows a malformed model reply', async () => {
  const { summariser, digests, audit } = harness({ reply: '这不是 JSON' })
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.equal(digests.count(OWNER), 0)
  assert.ok(audit.some(entry => entry.op === 'error'))
})

test('tolerates a fenced json reply', async () => {
  const { summariser } = harness({
    reply: '```json\n{"topics":["话题"],"gist":"要点"}\n```',
  })
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.deepEqual(digest.topics, ['话题'])
})

test('records a skip when the pool refuses the digest', async () => {
  const { summariser, audit } = harness({
    reply: '{"topics":["登录"],"gist":"密码是 hunter2"}',
  })
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.ok(audit.some(entry => entry.reason === 'digest_rejected'))
})

test('stays disabled without an llm call or a pool', () => {
  assert.equal(new SessionSummariser({ digestPool: {} }).enabled(), false)
  assert.equal(new SessionSummariser({ llmCall: async () => '' }).enabled(), false)
  const disabled = new SessionSummariser({ digestPool: {}, llmCall: null })
  assert.equal(disabled.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
})

test('requires both an owner and a session id', async () => {
  const { summariser } = harness()
  assert.equal(summariser.maybeRun({ ownerId: '', sessionId: 's_a' }), null)
  assert.equal(summariser.maybeRun({ ownerId: OWNER, sessionId: '' }), null)
})

test('a rejecting model call never escapes the hook', async () => {
  const { summariser, audit } = harness({
    reply: () => { throw new Error('模型不可用') },
  })
  // 关闭路径不能被摘要失败打断
  assert.equal(await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.ok(audit.some(entry => entry.op === 'error'))
})

test('records an audit entry with the accepted topics', async () => {
  const { summariser, audit } = harness({
    reply: '{"topics":["记忆架构"],"gist":"讨论了槽位池"}',
  })
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  const entry = audit.find(item => item.op === 'observe')
  assert.equal(entry.scope, 'session_digest')
  assert.deepEqual(entry.detail.topics, ['记忆架构'])
})

// 任务台账终态只留 30 天，摘要留 90 天，所以派过的活要沉淀进摘要。
test('sinks the work dispatched in the session into the digest', async () => {
  const { summariser, digests } = harness({
    listSessionWork: () => [{ id: 't1', objective: '把压缩评测跑一遍' }],
  })
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.deepEqual(digest.work, [{ id: 't1', objective: '把压缩评测跑一遍' }])
  assert.equal(digests.search({ ownerId: OWNER, keyword: '压缩评测' }).length, 1)
})

test('records a session that only dispatched work and yielded no topic', async () => {
  const { summariser, digests } = harness({
    reply: '{"topics":[],"gist":""}',
    listSessionWork: () => [{ id: 't1', objective: '整理季度报表' }],
  })
  // 模型摘不出话题，但活确实派了 —— 这场仍然值得记
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.ok(digest)
  assert.equal(digests.count(OWNER), 1)
})

test('a throwing work collector never loses the digest', async () => {
  const { summariser, digests } = harness({
    listSessionWork: () => { throw new Error('台账读不到') },
  })
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  // 取不到任务只是少一个字段，不该连带丢掉整条摘要
  assert.ok(digest)
  assert.deepEqual(digest.work, [])
  assert.equal(digests.count(OWNER), 1)
})

test('works without a work collector at all', async () => {
  const { summariser } = harness()
  const digest = await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.deepEqual(digest.work, [])
})

test('a single fresh user message can produce a bounded experience summary', async () => {
  const { summariser, digests } = harness({
    messages: transcript(['我今天参加了项目面试，约好下周二复试']),
    reply: JSON.stringify({ topics: ['项目面试'], gist: '约好复试',
      summary: '用户今天参加项目面试，约好下周二复试。' }),
  })
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.match(digests.search({ ownerId: OWNER, keyword: '复试' })[0].summary, /下周二/)
})

test('restored history is never treated as new summary input', async () => {
  const { summariser, sync, calls } = harness({ messages: [] })
  sync.restore({ ownerId: OWNER, sessionId: 's_a', messages: [
    { id: 'old', role: 'user', content: '过去经历', source: 'voice-user' },
  ] })
  assert.equal(summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  assert.equal(calls.length, 0)
})

test('concurrent duplicate closes summarize each fresh batch once', async () => {
  const deferred = Promise.withResolvers()
  const { summariser, calls } = harness({ reply: () => deferred.promise })
  const first = summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.equal(summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' }), null)
  deferred.resolve('{"topics":["项目面试"],"gist":"约好复试","summary":"用户准备复试。"}')
  await first
  assert.equal(calls.length, 1)
})

test('fresh messages update the existing session summary with its old context', async () => {
  const { summariser, calls, sync, digests } = harness({ messages: transcript(['周二复试']),
    reply: '{"topics":["复试"],"gist":"复试时间","summary":"用户的复试在周二。"}' })
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  sync.record({ ownerId: OWNER, sessionId: 's_a', id: 'correction', role: 'user',
    content: '更正，是周三复试', source: 'voice-user' })
  summariser.llmCall = async payload => {
    calls.push(payload)
    return '{"topics":["复试"],"gist":"复试时间已更正","summary":"用户的复试已改为周三。"}'
  }
  await summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  assert.equal(digests.count(OWNER), 1)
  assert.match(calls[1].user, /本场旧摘要[\s\S]*周二/)
  assert.match(calls[1].user, /新对话转写\n用户: 更正，是周三复试/)
  const saved = digests.get({ ownerId: OWNER, sessionId: 's_a' })
  assert.equal(saved.turns, 2)
  assert.match(saved.summary, /周三/)
})

test('a successful explicit edit invalidates an in-flight experience summary', async () => {
  const deferred = Promise.withResolvers()
  const { summariser, sync, digests } = harness({ reply: () => deferred.promise })
  const pending = summariser.maybeRun({ ownerId: OWNER, sessionId: 's_a' })
  await Promise.resolve()
  sync.discardRecorded(OWNER)
  deferred.resolve('{"topics":["旧经历"],"gist":"旧记录","summary":"不可重新写回的旧经历"}')
  await pending
  assert.equal(digests.count(OWNER), 0)
})
