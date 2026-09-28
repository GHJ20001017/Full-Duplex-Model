import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { resolve } from 'node:path'
import { PassThrough } from 'node:stream'
import test from 'node:test'
import {
  macVoiceIOBinaryPath,
  startMacVoiceIO,
} from '../src/macos-voice-io.mjs'

test('uses the standard macOS cache for the native voice helper', () => {
  assert.equal(
    macVoiceIOBinaryPath({
      homeDirectory: '/Users/example',
    }),
    resolve('/Users/example', 'Library/Caches/qwaudio/tui/macos-voice-io'),
  )
})

test('bridges native playback lifecycle events with response ids', async () => {
  const child = new EventEmitter()
  child.stdin = new PassThrough()
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.killed = false
  child.kill = () => {
    child.killed = true
  }
  const commands = []
  const playbackEvents = []
  child.stdin.on('data', chunk => {
    commands.push(...chunk.toString().trim().split('\n').filter(Boolean))
  })

  const bridgePromise = startMacVoiceIO({
    captureSampleRate: 16000,
    ensureImpl: async () => '/tmp/test-macos-voice-io',
    spawnImpl(command, args) {
      assert.equal(command, '/tmp/test-macos-voice-io')
      assert.deepEqual(args, [
        '--capture-rate',
        '16000',
        '--playback-rate',
        '24000',
      ])
      queueMicrotask(() => child.stdout.write('{"type":"ready"}\n'))
      return child
    },
    onPlaybackStarted: responseId => playbackEvents.push(['started', responseId]),
    onPlaybackEnded: responseId => playbackEvents.push(['ended', responseId]),
  })
  const bridge = await bridgePromise

  child.stdout.write('{"type":"playback.started","responseId":"response-1"}\n')
  child.stdout.write('{"type":"playback.ended","responseId":"response-1"}\n')
  assert.equal(
    bridge.write(Buffer.from([1, 2]), 24000, 'response-1'),
    true,
  )
  assert.equal(bridge.done('response-1'), true)
  bridge.close()

  assert.deepEqual(playbackEvents, [
    ['started', 'response-1'],
    ['ended', 'response-1'],
  ])
  assert.deepEqual(commands.map(line => JSON.parse(line)), [
    {
      type: 'play',
      audio: 'AQI=',
      sampleRate: 24000,
      responseId: 'response-1',
    },
    { type: 'done', responseId: 'response-1' },
    { type: 'close' },
  ])
})
