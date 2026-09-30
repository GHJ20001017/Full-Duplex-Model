import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createDelegationTrace, delegationWirePayload, sanitizeDelegationTrace } from '../src/core/delegation-trace.mjs'
import { RealtimeFrontend, REALTIME_PROVIDERS } from '../src/voice/realtime-provider.mjs'
import { ToolCallHandler } from '../src/frontend/tools/tool-call-handler.mjs'

function sink() {
  const lines = []
  const calls = []
  const io = {
    openSync: (...args) => { calls.push(['open', ...args]); return 7 },
    fstatSync: () => ({ isFile: () => true, nlink: 1, uid: process.getuid() }),
    fchmodSync: (...args) => calls.push(['chmod', ...args]),
    writeFileSync: (_fd, line) => lines.push(JSON.parse(line)),
    closeSync: fd => calls.push(['close', fd]),
  }
  return { io, lines, calls }
}

test('disabled and relative-path traces never evaluate context or touch IO', () => {
  for (const path of ['', 'relative.jsonl']) {
    createDelegationTrace({ path, context: () => { throw Error('unexpected') }, io: null })('test')
  }
})

test('trace redacts credentials, serialized arguments, audio and bounds strings explicitly', () => {
  const value = sanitizeDelegationTrace({
    auth: 'hidden-auth', token: 'hidden-token', apiKey: 'hidden-key',
    arguments: JSON.stringify({ password: 'hidden-pass', text: 'actual user request', nested: { audio: 'hidden-audio' } }),
    text: 'Bearer hidden-bearer https://user:hidden-url@example.com/?token=hidden-query',
    instructions: 'i'.repeat(40_000),
    content: [{ type: 'input_audio', audio: 'hidden-blob' }],
  })
  assert.doesNotMatch(JSON.stringify(value), /hidden-/)
  assert.equal(value.arguments.text, 'actual user request')
  assert.equal(value.instructions.truncated, true)
  assert.equal(value.instructions.originalChars, 40_000)
  assert.equal(value.instructions.text.length, 32_768)
})

test('secure append uses 0600, no-follow and close, while IO failures fail open', () => {
  const { io, lines, calls } = sink()
  const trace = createDelegationTrace({ path: '/trace.jsonl', io, context: () => ({ sessionId: 'session' }) })
  trace('test', { turnId: 'turn' })
  assert.equal(lines[0].sessionId, 'session')
  assert.equal(lines[0].turnId, 'turn')
  assert.equal(calls[0][3], 0o600)
  assert.ok(calls[0][2] & fs.constants.O_NOFOLLOW)
  assert.deepEqual(calls[1], ['chmod', 7, 0o600])
  assert.deepEqual(calls.at(-1), ['close', 7])
  for (const operation of ['openSync', 'fstatSync', 'fchmodSync', 'writeFileSync', 'closeSync']) {
    const failing = { ...io, [operation]: () => { throw Error('secret path') } }
    assert.doesNotThrow(() => createDelegationTrace({ path: '/trace.jsonl', io: failing })('test'))
  }
  const unsafe = { ...io, fstatSync: () => ({ isFile: () => false }) }
  const before = lines.length
  createDelegationTrace({ path: '/trace.jsonl', io: unsafe })('test')
  assert.equal(lines.length, before)
})

test('context/serialization errors fail open and oversized records are explicitly omitted', () => {
  const { io, lines } = sink()
  assert.doesNotThrow(() => createDelegationTrace({ path: '/trace.jsonl', io, context: () => { throw Error('context') } })('test'))
  assert.equal(lines.length, 0)
  createDelegationTrace({ path: '/trace.jsonl', io })('large', { sessionId: 's', payload: Array.from({ length: 20 }, () => 'x'.repeat(32_768)) })
  assert.equal(lines[0].truncated, true)
  assert.ok(lines[0].originalBytes > 262_144)
  assert.equal(lines[0].sessionId, 's')
  assert.equal(lines[0].payload, undefined)
})

test('session payload keeps instructions/tools but excludes configuration and frame traffic', () => {
  const event = { type: 'session.update', session: { instructions: 'delegate', tools: [{ name: 'spawn_thinking' }], tool_choice: 'auto', audio: { secret: 'no' }, headers: { Authorization: 'no' } } }
  const payload = delegationWirePayload(event)
  assert.equal(payload.session.instructions, 'delegate')
  assert.deepEqual(payload.session.tools, event.session.tools)
  assert.equal(payload.session.audio, undefined)
  assert.equal(payload.session.headers, undefined)
  for (const type of ['input_audio_buffer.append', 'response.audio.delta', 'response.text.delta', 'response.function_call_arguments.delta']) {
    assert.equal(delegationWirePayload({ type, delta: 'blob' }), null)
  }
})

test('transport captures encoded outbound payload, final events and stable response correlation', t => {
  const previous = process.env.QWEN_AUDIO_DELEGATION_TRACE_PATH
  process.env.QWEN_AUDIO_DELEGATION_TRACE_PATH = '/trace.jsonl'
  t.after(() => {
    if (previous === undefined) delete process.env.QWEN_AUDIO_DELEGATION_TRACE_PATH
    else process.env.QWEN_AUDIO_DELEGATION_TRACE_PATH = previous
  })
  let turnId = 'turn-1'
  const frontend = new RealtimeFrontend({ provider: REALTIME_PROVIDERS.s2s, getTraceContext: () => ({ sessionId: 's', turnId }) })
  const { io, lines } = sink()
  frontend.delegationTrace = createDelegationTrace({ path: '/trace.jsonl', io, context: () => ({ sessionId: 's', connectionId: frontend.connectionId }) })
  const wire = []
  frontend.ws = { readyState: 1, bufferedAmount: 0, send: body => wire.push(JSON.parse(body)) }
  frontend.send({ type: 'session.update', session: { instructions: 'real prompt', tools: [{ name: 'spawn_thinking' }], audio: {} } })
  const pending = { requestId: 'request-1', context: { turnId: 'turn-1' } }
  frontend.pendingResponses.push(pending)
  frontend.send({ type: 'response.create', response: { instructions: 'real instruction', tool_choice: 'auto' } })
  frontend.send({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'actual outbound text' }] } })
  const outbound = lines.filter(line => line.boundary === 'wire.outbound')
  assert.deepEqual(outbound[1].payload, wire[1])
  assert.equal(outbound[1].requestId, 'request-1')
  assert.equal(outbound[1].sent, true)
  frontend.traceWire('inbound', { type: 'response.created', response: { id: 'r', metadata: wire[1].response.metadata } })
  frontend.pendingResponses.length = 0
  turnId = 'turn-2'
  for (const type of ['response.function_call_arguments.done', 'response.text.done', 'conversation.item.input_audio_transcription.completed', 'response.done']) {
    frontend.traceWire('inbound', { type, response_id: 'r', text: 'final' })
    assert.equal(lines.at(-1).turnId, 'turn-1')
    assert.equal(lines.at(-1).responseId, 'r')
  }
  assert.equal(outbound[2].payload.item.content[0].text, 'actual outbound text')
})

test('dispatch traces missing id, duplicate and stale admission without changing outcomes', async () => {
  const handler = new ToolCallHandler({ getFrontend: () => null, getTurnId: () => 'current', getTurnGeneration: () => 2 })
  const records = []
  handler.delegationTrace = (boundary, details) => records.push({ boundary, ...details })
  await assert.rejects(handler.handle({ name: 'spawn_thinking' }), /call_id/)
  assert.equal(records.at(-1).reason, 'missing_call_id')
  handler.processedCalls.add('duplicate')
  await handler.handle({ call_id: 'duplicate' })
  assert.equal(records.at(-1).reason, 'duplicate_call_id')
  handler.closeStaleCall = async () => {}
  await handler.handle({ call_id: 'old', name: 'spawn_thinking' }, { turnId: 'old-turn', turnGeneration: 1, responseId: 'r' })
  assert.equal(records.at(-1).reason, 'stale_turn')
  assert.equal(records.at(-1).responseId, 'r')
})
