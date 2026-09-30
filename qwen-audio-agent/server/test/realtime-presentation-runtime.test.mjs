import assert from 'node:assert/strict'
import test from 'node:test'
import { RealtimePresentationRuntime } from '../src/voice/realtime-presentation-runtime.mjs'
import { RealtimeTurnState } from '../src/voice/realtime-turn-state.mjs'

function harness({
  nonVoiceClient = false,
  turnCitations = null,
  terminalToolResponses = [],
  resultSummaryResponses = [],
  perResponseInstructions = false,
  tasks = [],
} = {}) {
  const events = []
  const records = []
  const calls = []
  const turns = new RealtimeTurnState({
    createVoiceTurnId: generation => `voice-${generation}`,
  })
  let responseTurnCandidate = null
  const frontend = {
    ready: true,
    provider: { outputSampleRate: 24000 },
    capabilities: { perResponseInstructions },
    ensureResponse: async (...args) => calls.push(['ensureResponse', ...args]),
  }
  const terminalResponses = new Set(terminalToolResponses)
  const runtime = new RealtimePresentationRuntime({
    ownerId: 'owner-1',
    sessionId: 'session-1',
    turns,
    conversationSync: { record: value => records.push(value) },
    announcementWindow: {
      queueAudio: (...args) => calls.push(['queueAudio', ...args]),
      startPlayback: (...args) => calls.push(['startPlayback', ...args]),
      finishPlayback: (...args) => calls.push(['finishPlayback', ...args]),
      responseDone: (...args) => calls.push(['responseDone', ...args]),
    },
    announcements: {
      confirmMany: ids => calls.push(['confirmMany', ids]),
      retryMany: ids => calls.push(['retryMany', ids]),
      flush: () => calls.push(['flush']),
    },
    toolCalls: {
      taskOperations: { get: id => tasks.find(task => task.id === id) },
      requiresToolResultSummary: id => resultSummaryResponses.includes(id),
      consumeTerminalToolResponse: id => {
        calls.push(['consumeTerminalToolResponse', id])
        return terminalResponses.delete(id)
      },
      finishToolResponse: async (...args) => calls.push([
        'finishToolResponse',
        ...args,
      ]),
    },
    send: event => events.push(event),
    getFrontend: () => frontend,
    getOutputEnabled: () => true,
    getNonVoiceClient: () => nonVoiceClient,
    getResponseTurnCandidate: () => responseTurnCandidate,
    clearResponseCandidate: () => {
      responseTurnCandidate = null
      calls.push(['clearResponseCandidate'])
    },
    announcementQuietMs: 60_000,
    responseContextCleanupMs: 60_000,
    turnCitations,
  })
  return {
    runtime,
    turns,
    events,
    records,
    calls,
    setResponseTurnCandidate(value) {
      responseTurnCandidate = value
    },
  }
}

const taskSpeech = '我已经启动后台任务，正在执行了。'
const speechEvents = setup => setup.events.filter(event =>
  event.type === 'audio.delta' || event.type.startsWith('transcript.'))

for (const audioType of ['response.audio.delta', 'response.output_audio.delta']) {
  test(`blocks an uninvoked task claim before ${audioType} or text reaches the client`, () => {
    const setup = harness({ perResponseInstructions: true })
    const turn = setup.turns.beginVoice('input-guard').context
    setup.turns.endSpeech()
    setup.turns.commit(turn)
    deliver(setup.runtime, {
      type: audioType, response_id: 'guarded', delta: 'early-audio',
      __voiceContext: { ...turn, taskId: 'uninvoked-task' },
    })
    deliver(setup.runtime, { type: 'response.text.delta', response_id: 'guarded', delta: taskSpeech })
    setup.runtime.startPlayback('guarded')
    assert.equal(speechEvents(setup).length, 0)
    deliver(setup.runtime, { type: 'response.text.done', response_id: 'guarded', text: taskSpeech })
    deliver(setup.runtime, { type: 'response.audio.delta', response_id: 'guarded', delta: 'late-audio' })
    deliver(setup.runtime, { type: 'response.done', response: { id: 'guarded', status: 'completed' } })
    assert.equal(speechEvents(setup).length, 0)
    assert.equal(setup.records.length, 0)
    assert.equal(setup.calls.filter(([name]) => name === 'ensureResponse').length, 1)
  })
}

for (const [status, executionStartedAt, allowed] of [
  ['running', undefined, false], ['running', null, false], ['running', 0, true],
  ['queued', undefined, false], ['failed', 123, false], ['cancelled', 123, false],
  ['delegated', undefined, false], ['finalizing', undefined, false],
]) {
  test(`uses actual ${status} task evidence (execution marker ${executionStartedAt})`, () => {
    const setup = harness({ tasks: [{
      id: 'real-task', ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1', status, executionStartedAt,
    }] })
    const context = { turnId: 'turn-1', taskIds: ['real-task'], consumesTaskNotification: true }
    deliver(setup.runtime, {
      type: 'response.audio.delta', response_id: 'guarded', delta: 'audio', __voiceContext: context,
    })
    assert.equal(speechEvents(setup).length, 0)
    setup.runtime.markFunctionCall('guarded')
    deliver(setup.runtime, {
      type: 'response.audio_transcript.done', response_id: 'guarded', transcript: taskSpeech,
    })
    setup.runtime.startPlayback('guarded')
    deliver(setup.runtime, { type: 'response.done', response: { id: 'guarded', status: 'completed' } })
    assert.equal(speechEvents(setup).length > 0, allowed)
    if (!allowed) {
      const finish = setup.calls.find(([name]) => name === 'finishToolResponse')
      assert.equal(finish[2].sourceHasSpeech, false)
      assert.equal(finish[2].suppressResponse, false)
    }
  })
}

for (const transcript of [
  '后台任务已经执行了。', '已完成任务。', '已经重启服务。',
  'I restarted the service.', 'I have completed the background task.',
]) {
  for (const [label, task, allowed] of [
    ['no invocation', null, false],
    ['accepted', { status: 'queued' }, false],
    ['executing', { status: 'running', executionStartedAt: 123, result: 'partial progress' }, false],
    ['empty completed', { status: 'completed', executionStartedAt: 123, result: '' }, false],
    ['blank completed', { status: 'completed', result: '  ' }, false],
    ['failed', { status: 'failed', result: 'partial result' }, false],
    ['completed result', { status: 'completed', executionStartedAt: 123, result: '实际执行结果：服务已重启。' }, true],
  ]) {
    test(`completion speech gate: ${label}: ${transcript}`, () => {
      const setup = harness({ tasks: task ? [{
        id: 'work', ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1', ...task,
      }] : [] })
      deliver(setup.runtime, {
        type: 'response.audio.delta', response_id: 'completion', delta: 'early-audio',
        __voiceContext: { turnId: 'turn-1', taskId: 'work' },
      })
      deliver(setup.runtime, {
        type: 'response.audio_transcript.delta', response_id: 'completion', delta: transcript,
      })
      assert.equal(speechEvents(setup).length, 0)
      deliver(setup.runtime, {
        type: 'response.audio_transcript.done', response_id: 'completion', transcript,
      })
      setup.runtime.startPlayback('completion')
      deliver(setup.runtime, { type: 'response.done', response: { id: 'completion', status: 'completed' } })
      assert.equal(speechEvents(setup).length > 0, allowed)
      assert.equal(setup.records.length, allowed ? 1 : 0)
    })
  }
}

for (const [label, other, acceptance, execution, completion] of [
  ['missing', null, false, false, false],
  ['wrong owner', { ownerId: 'someone-else' }, false, false, false],
  ['wrong session', { sessionId: 'another-session' }, false, false, false],
  ['wrong turn', { turnId: 'another-turn' }, false, false, false],
  ['failed', { status: 'failed' }, false, false, false],
  ['queued', { status: 'queued', executionStartedAt: undefined }, true, false, false],
  ['admitted', { status: 'running', executionStartedAt: undefined }, true, false, false],
  ['started', { status: 'running' }, true, true, false],
  ['empty result', { result: '' }, true, true, false],
  ['completed', {}, true, true, true],
]) {
  for (const reverse of [false, true]) {
    for (const [text, allowed] of [
      ['后台任务已经提交。', acceptance],
      ['我已经开始处理后台任务。', execution],
      ['已完成任务。', completion],
    ]) {
      test(`batch evidence requires every task: ${label}, reversed=${reverse}, ${text}`, () => {
        const valid = { ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1',
          status: 'completed', executionStartedAt: 123, result: '实际结果' }
        const setup = harness({ tasks: [
          { ...valid, id: 'first' },
          ...(other ? [{ ...valid, ...other, id: 'second' }] : []),
        ] })
        const taskIds = reverse ? ['second', 'first'] : ['first', 'second']
        deliver(setup.runtime, { type: 'response.audio.delta', response_id: 'batch', delta: 'audio',
          __voiceContext: { turnId: 'turn-1', taskIds } })
        assert.equal(speechEvents(setup).length, 0)
        deliver(setup.runtime, { type: 'response.audio_transcript.done', response_id: 'batch', transcript: text })
        setup.runtime.startPlayback('batch')
        deliver(setup.runtime, { type: 'response.done', response: { id: 'batch', status: 'completed' } })
        assert.equal(speechEvents(setup).length > 0, allowed)
        assert.equal(setup.records.length, allowed ? 1 : 0)
      })
    }
  }
}

test('executing task allows started processing but not a subsequent completion claim', () => {
  const setup = harness({ tasks: [{
    id: 'work', ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1',
    status: 'running', executionStartedAt: 123,
  }] })
  for (const [id, text] of [['started', '我已经开始处理后台任务。'], ['finished', '后台任务已经执行了。']]) {
    deliver(setup.runtime, { type: 'response.text.delta', response_id: id, delta: text,
      __voiceContext: { turnId: 'turn-1', taskId: 'work' } })
    assert.equal(speechEvents(setup).some(event => event.responseId === id), false)
    deliver(setup.runtime, { type: 'response.text.done', response_id: id, text })
    assert.equal(speechEvents(setup).some(event => event.responseId === id), id === 'started')
  }
})

test('a real queued task permits submission but not execution claims', () => {
  const setup = harness({ tasks: [{
    id: 'queued', ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1', status: 'queued',
  }] })
  deliver(setup.runtime, {
    type: 'response.text.done', response_id: 'receipt', text: '后台任务已经提交。',
    __voiceContext: { turnId: 'turn-1', taskId: 'queued' },
  })
  assert.equal(setup.records.length, 1)
})

for (const transcript of [
  '你好，今天想聊什么？',
  '例如：“我已经启动后台任务。”这只是一个例句。',
  '后台任务没有启动。',
  'Background tasks run independently of the foreground UI.',
]) {
  test(`ordinary chat and discussion remains deliverable: ${transcript}`, () => {
    const setup = harness()
    deliver(setup.runtime, { type: 'response.audio.delta', response_id: 'chat', delta: 'audio-1' })
    deliver(setup.runtime, { type: 'response.audio.delta', response_id: 'chat', delta: 'audio-2' })
    deliver(setup.runtime, { type: 'response.text.delta', response_id: 'chat', delta: transcript })
    // Ordinary model replies stream immediately; no transcript-done barrier.
    assert.deepEqual(
      speechEvents(setup).filter(event => event.type === 'audio.delta').map(event => event.audio),
      ['audio-1', 'audio-2'],
    )
    assert.equal(
      speechEvents(setup).some(event => event.type === 'transcript.delta'),
      true,
    )
    deliver(setup.runtime, { type: 'response.text.done', response_id: 'chat', text: transcript })
    setup.runtime.startPlayback('chat')
    assert.deepEqual(speechEvents(setup).filter(e => e.type === 'audio.delta').map(e => e.audio), ['audio-1', 'audio-2'])
    assert.equal(setup.records[0].content, transcript)
  })
}

for (const audioType of ['response.audio.delta', 'response.output_audio.delta']) {
  test(`ordinary ${audioType} streams before done and transcripts follow playback`, () => {
    const setup = harness()
    deliver(setup.runtime, { type: audioType, response_id: 'stream', delta: 'first' })
    assert.equal(speechEvents(setup)[0]?.audio, 'first')
    deliver(setup.runtime, {
      type: 'response.audio_transcript.delta', response_id: 'stream', delta: '你好',
    })
    assert.equal(speechEvents(setup).length, 1)
    setup.runtime.startPlayback('stream')
    assert.equal(speechEvents(setup)[1]?.content, '你好')
    assert.equal(setup.runtime.get('stream').transcriptDone, false)
    assert.equal(setup.records.length, 0)
  })
}

test('uncorrelated task claims stream without a false withheld-speech correction at response.done', () => {
  const setup = harness({ perResponseInstructions: true })
  const turn = setup.turns.beginVoice('input').context
  setup.turns.endSpeech()
  setup.turns.commit(turn)
  deliver(setup.runtime, {
    type: 'response.audio.delta', response_id: 'streamed', delta: 'audio', __voiceContext: turn,
  })
  deliver(setup.runtime, { type: 'response.text.delta', response_id: 'streamed', delta: taskSpeech })
  assert.equal(speechEvents(setup).length, 2)
  deliver(setup.runtime, { type: 'response.text.done', response_id: 'streamed', text: taskSpeech })
  deliver(setup.runtime, { type: 'response.done', response: { id: 'streamed', status: 'completed' } })
  assert.equal(setup.calls.some(([name]) => name === 'ensureResponse'), false)
  assert.equal(setup.records[0].content, taskSpeech)
})

for (const status of ['failed', 'cancelled', 'incomplete', 'completed']) {
  test(`does not release unclassified task-correlated audio on ${status} completion`, () => {
    const setup = harness()
    deliver(setup.runtime, {
      type: 'response.audio.delta', response_id: 'silent', delta: 'unclassified',
      __voiceContext: { taskId: 'unverified-task' },
    })
    deliver(setup.runtime, {
      type: 'response.done', response: { id: 'silent', status },
      __voiceContext: { taskId: 'unverified-task' },
    })
    assert.equal(speechEvents(setup).length, 0)
    assert.equal(setup.runtime.has('silent'), false)
  })
}

test('a proposed function call and bare task ID are not successful submission evidence', () => {
  const setup = harness()
  deliver(setup.runtime, {
    type: 'response.created', response: { id: 'proposal' },
    __voiceContext: { taskId: 'invented', consumesTaskNotification: true },
  })
  setup.runtime.markFunctionCall('proposal')
  deliver(setup.runtime, { type: 'response.text.delta', response_id: 'proposal', delta: taskSpeech })
  deliver(setup.runtime, { type: 'response.done', response: { id: 'proposal', status: 'completed' } })
  assert.equal(speechEvents(setup).length, 0)
})

test('projects turn citations once on the final assistant transcript', () => {
  const stored = [{
    id: 'source_1',
    title: '杭州天气',
    url: 'https://example.com/weather',
  }]
  let consumed = false
  const setup = harness({
    turnCitations: {
      consume(turnId) {
        assert.equal(turnId, 'turn-1')
        if (consumed) return []
        consumed = true
        return stored
      },
    },
  })

  deliver(setup.runtime, {
    type: 'response.text.done',
    response_id: 'response-1',
    text: '今天晴。',
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })

  const final = setup.events.find(event => event.type === 'transcript.final')
  assert.deepEqual(final.citations, stored)
  assert.deepEqual(setup.records[0].citations, stored)
})

test('does not persist a model-generated Gateway protocol envelope', () => {
  const setup = harness()
  const content = [
    '<permission_request>',
    'task_id=fake',
    '</permission_request>',
  ].join(' ')

  deliver(setup.runtime, {
    type: 'response.text.done',
    response_id: 'response-fake-protocol',
    text: content,
    __voiceContext: {
      origin: 'model',
      turnId: 'turn-1',
      turnGeneration: 1,
    },
  })

  assert.equal(setup.records.length, 0)
  assert.equal(
    setup.events.some(event => (
      event.type === 'transcript.final' && event.content === content
    )),
    true,
  )
})

test('allows only one protocol correction per user turn, including repeated invalid corrections', () => {
  const setup = harness({ perResponseInstructions: true })
  const invalidResponse = id => {
    const context = setup.turns.committed()
    deliver(setup.runtime, {
      type: 'response.text.done', response_id: id,
      text: '<permission_request>fake</permission_request>', __voiceContext: context,
    })
    deliver(setup.runtime, { type: 'response.done', response: { id, status: 'completed' } })
  }
  const first = setup.turns.beginVoice('input-1').context
  setup.turns.endSpeech()
  setup.turns.commit(first)
  invalidResponse('response-1')
  invalidResponse('response-2')
  invalidResponse('response-3')
  const corrections = () => setup.calls.filter(([name]) => name === 'ensureResponse')
  assert.equal(corrections().length, 1)
  assert.equal(corrections()[0][2].shouldCreate(), true)
  const second = setup.turns.beginVoice('input-2').context
  setup.turns.endSpeech()
  setup.turns.commit(second)
  assert.equal(corrections()[0][2].shouldCreate(), false)
  invalidResponse('response-4')
  assert.equal(corrections().length, 2)
})

function deliver(runtime, event) {
  runtime.begin(event)
  runtime.handle(event)
}

test('correlates an implicit provider response with the pending voice turn', () => {
  const setup = harness()
  const candidate = setup.turns.beginVoice('item-1').context
  setup.turns.endSpeech()
  setup.setResponseTurnCandidate(candidate)

  deliver(setup.runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
  })

  assert.deepEqual(setup.turns.committed(), candidate)
  assert.equal(setup.events[0].type, 'response.started')
  assert.equal(setup.events.length, 2)
  deliver(setup.runtime, {
    type: 'response.audio_transcript.done', response_id: 'response-1', transcript: '你好。',
  })
  assert.equal(setup.events[1].type, 'audio.delta')
  assert.equal(setup.events[1].sampleRate, 24000)
  assert.equal(
    setup.calls.some(([name]) => name === 'clearResponseCandidate'),
    true,
  )
})

test('holds audio transcripts until playback starts and records them once', () => {
  const { runtime, events, records, calls } = harness({ tasks: [{
    id: 'work-1', ownerId: 'owner-1', sessionId: 'session-1', turnId: 'turn-1',
    status: 'completed', result: '实际任务结果',
  }] })
  const context = {
    turnId: 'turn-1',
    turnGeneration: 1,
    taskIds: ['work-1'],
    consumesTaskNotification: true,
  }

  deliver(runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
    __voiceContext: context,
  })
  deliver(runtime, {
    type: 'response.audio_transcript.delta',
    response_id: 'response-1',
    delta: '后台任务',
  })
  deliver(runtime, {
    type: 'response.audio_transcript.done',
    response_id: 'response-1',
    transcript: '后台任务完成了',
  })
  assert.equal(events.some(event => event.type === 'transcript.final'), false)

  runtime.startPlayback('response-1')

  assert.deepEqual(
    events.filter(event => event.type.startsWith('transcript.')).map(event => ({
      type: event.type,
      content: event.content,
    })),
    [
      { type: 'transcript.delta', content: '后台任务' },
      { type: 'transcript.final', content: '后台任务完成了' },
    ],
  )
  assert.equal(records.length, 1)
  assert.equal(records[0].source, 'realtime-direct')
  assert.equal(
    calls.filter(([name]) => name === 'confirmMany').length,
    1,
  )
})

test('retires an audio response only after response, transcript and playback end', () => {
  const { runtime } = harness()

  deliver(runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })
  runtime.startPlayback('response-1')
  deliver(runtime, {
    type: 'response.audio_transcript.done',
    response_id: 'response-1',
    transcript: '完成',
  })
  deliver(runtime, {
    type: 'response.done',
    response: { id: 'response-1', status: 'completed' },
  })
  assert.equal(runtime.has('response-1'), true)

  runtime.finishPlayback('response-1')

  assert.equal(runtime.has('response-1'), false)
})

test('keeps processing while a foreground tool result is pending', () => {
  const { runtime, events } = harness()

  deliver(runtime, {
    type: 'response.created',
    response: { id: 'response-1' },
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })
  runtime.markFunctionCall('response-1')
  deliver(runtime, {
    type: 'response.done',
    response: { id: 'response-1', status: 'completed' },
  })

  assert.deepEqual(
    events.filter(event => event.type === 'voice.state').map(event => event.state),
    ['processing'],
  )
})

test('returns to idle after a terminal tool response', () => {
  const { runtime, events } = harness({
    terminalToolResponses: ['response-1'],
  })

  deliver(runtime, {
    type: 'response.created',
    response: { id: 'response-1' },
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })
  runtime.markFunctionCall('response-1')
  deliver(runtime, {
    type: 'response.done',
    response: { id: 'response-1', status: 'completed' },
  })

  assert.deepEqual(
    events.filter(event => event.type === 'voice.state').map(event => event.state),
    ['processing', 'idle'],
  )
})

test('releases a spoken function-call turn when its tool follow-up is suppressed', () => {
  const { runtime, events, calls } = harness()

  deliver(runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })
  runtime.markFunctionCall('response-1')
  deliver(runtime, {
    type: 'response.audio_transcript.done', response_id: 'response-1', transcript: '好的。',
  })
  runtime.startPlayback('response-1')
  deliver(runtime, {
    type: 'response.done',
    response: { id: 'response-1', status: 'completed' },
  })
  runtime.finishPlayback('response-1')

  assert.equal(events.at(-1).type, 'voice.state')
  assert.equal(events.at(-1).state, 'idle')
  assert.deepEqual(
    calls.find(([name]) => name === 'finishToolResponse'),
    ['finishToolResponse', 'response-1', {
      suppressResponse: false,
      sourceHasSpeech: true,
    }],
  )
  assert.deepEqual(
    calls.find(([name]) => name === 'responseDone'),
    ['responseDone', {
      turnId: 'turn-1',
      origin: 'model',
      hasAudio: true,
      awaitsToolFollowUp: false,
      suppressed: false,
      failed: false,
    }],
  )
})

test('keeps a spoken inline-tool turn open until its results can be summarized', () => {
  const { runtime, calls } = harness({ resultSummaryResponses: ['response-1'] })
  deliver(runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
    __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
  })
  deliver(runtime, {
    type: 'response.audio_transcript.done', response_id: 'response-1', transcript: '请稍等。',
  })
  runtime.markFunctionCall('response-1')
  deliver(runtime, {
    type: 'response.done',
    response: { id: 'response-1', status: 'completed' },
  })

  assert.deepEqual(calls.find(([name]) => name === 'finishToolResponse'), [
    'finishToolResponse', 'response-1', { suppressResponse: false, sourceHasSpeech: true },
  ])
  assert.equal(calls.find(([name]) => name === 'responseDone')[1].awaitsToolFollowUp, true)
})

for (const status of ['failed', 'cancelled', 'incomplete']) {
  test(`suppresses even an inline result summary when the source response is ${status}`, () => {
    const { runtime, calls } = harness({ resultSummaryResponses: ['response-1'] })
    deliver(runtime, {
      type: 'response.created',
      response: { id: 'response-1' },
      __voiceContext: { turnId: 'turn-1', turnGeneration: 1 },
    })
    runtime.markFunctionCall('response-1')
    deliver(runtime, { type: 'response.done', response: { id: 'response-1', status } })
    assert.deepEqual(calls.find(([name]) => name === 'finishToolResponse'), [
      'finishToolResponse', 'response-1', { suppressResponse: true, sourceHasSpeech: false },
    ])
    assert.equal(calls.find(([name]) => name === 'responseDone')[1].awaitsToolFollowUp, false)
  })
}

test('user interruption confirms an announcement and suppresses late output', () => {
  const { runtime, events, calls } = harness()

  deliver(runtime, {
    type: 'response.audio.delta',
    response_id: 'response-1',
    delta: 'audio',
    __voiceOrigin: 'announcement',
    __voiceContext: {
      turnId: 'turn-1',
      turnGeneration: 1,
      taskIds: ['work-1'],
    },
  })
  runtime.startPlayback('response-1')
  runtime.cancelPlayback('response-1', { reason: 'user_interruption' })
  deliver(runtime, {
    type: 'response.audio_transcript.done',
    response_id: 'response-1',
    transcript: '不应出现',
  })

  assert.equal(
    events.filter(event => event.type === 'transcript.final').length,
    0,
  )
  assert.equal(
    events.filter(event => event.type === 'response.interrupted').length,
    1,
  )
  assert.equal(
    calls.filter(([name]) => name === 'confirmMany').length,
    3,
  )
  assert.equal(calls.some(([name]) => name === 'retryMany'), false)
})

test('a provider failure retries an undelivered announcement', () => {
  const { runtime, calls } = harness()
  runtime.begin({
    type: 'response.created',
    response: { id: 'response-1' },
    __voiceOrigin: 'announcement',
    __voiceContext: {
      turnId: 'turn-1',
      turnGeneration: 1,
      taskIds: ['work-1'],
    },
  })

  runtime.failResponse({ type: 'error', response_id: 'response-1' })

  assert.equal(runtime.has('response-1'), false)
  assert.deepEqual(
    calls.find(([name]) => name === 'retryMany'),
    ['retryMany', ['work-1']],
  )
})
