import assert from 'node:assert/strict'
import test from 'node:test'
import { streamingInputTranscript } from '../src/voice/input-transcript.mjs'

test('combines the confirmed ASR prefix with its mutable streaming suffix', () => {
  assert.equal(streamingInputTranscript({
    type: 'conversation.item.input_audio_transcription.delta',
    text: '帮我查一下',
    stash: '今天的天气',
  }), '帮我查一下今天的天气')
})

test('supports the alternate Qwen ASR streaming event name', () => {
  assert.equal(streamingInputTranscript({
    type: 'conversation.item.input_audio_transcription.text',
    text: '打开项目',
    stash: '',
  }), '打开项目')
})

test('ignores unrelated realtime events', () => {
  assert.equal(streamingInputTranscript({
    type: 'response.audio_transcript.delta',
    text: '不是用户输入',
  }), '')
})
