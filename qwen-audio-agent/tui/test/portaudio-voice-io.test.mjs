import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import test from 'node:test'

import { startPortAudioVoiceIO } from '../src/portaudio-voice-io.mjs'

test('bridges capture, playback, clear and close through the PortAudio helper', async () => {
  const child = new EventEmitter()
  child.stdin = new PassThrough()
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.kill = () => {}
  const commands = []
  const audio = []
  const playbackEvents = []
  child.stdin.on('data', chunk => {
    commands.push(...chunk.toString().trim().split('\n').filter(Boolean))
  })

  const bridgePromise = startPortAudioVoiceIO({
    captureSampleRate: 16000,
    python: 'python-test',
    spawnImpl(command, args) {
      assert.equal(command, 'python-test')
      assert.deepEqual(args.slice(-6), [
        '--capture-rate',
        '16000',
        '--playback-rate',
        '24000',
        '--mode',
        'half',
      ])
      queueMicrotask(() => {
        child.stdout.write('{"type":"ready"}\n')
      })
      return child
    },
    onAudio: chunk => audio.push(chunk),
    onPlaybackStarted: responseId => playbackEvents.push(['started', responseId]),
    onPlaybackEnded: responseId => playbackEvents.push(['ended', responseId]),
  })
  const bridge = await bridgePromise

  child.stdout.write('{"type":"audio","audio":"AQI="}\n')
  child.stdout.write('{"type":"playback.started","responseId":"response-1"}\n')
  child.stdout.write('{"type":"playback.ended","responseId":"response-1"}\n')
  bridge.setCaptureEnabled(true)
  assert.equal(
    bridge.write(Buffer.from([3, 4]), 24000, 'response-1'),
    true,
  )
  assert.equal(bridge.done('response-1'), true)
  bridge.clear()
  bridge.close()

  assert.deepEqual(audio, [Buffer.from([1, 2])])
  assert.deepEqual(playbackEvents, [
    ['started', 'response-1'],
    ['ended', 'response-1'],
  ])
  assert.deepEqual(commands.map(line => JSON.parse(line)), [
    { type: 'capture', enabled: true },
    { type: 'play', audio: 'AwQ=', responseId: 'response-1' },
    { type: 'done', responseId: 'response-1' },
    { type: 'clear' },
    { type: 'close' },
  ])
})
