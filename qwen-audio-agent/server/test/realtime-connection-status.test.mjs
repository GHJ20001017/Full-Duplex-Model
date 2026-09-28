import assert from 'node:assert/strict'
import test from 'node:test'
import { realtimeConnectionStatus } from '../src/voice/realtime-connection-status.mjs'

test('uses one deterministic precedence for realtime lifecycle states', () => {
  const base = { provider: 'dashscope' }
  assert.equal(realtimeConnectionStatus(base).state, 'disconnected')
  assert.equal(realtimeConnectionStatus({ ...base, connecting: true }).state, 'connecting')
  assert.equal(realtimeConnectionStatus({ ...base, ready: true }).state, 'connected')
  assert.equal(realtimeConnectionStatus({
    ...base,
    ready: true,
    waking: true,
  }).state, 'waking')
  assert.equal(realtimeConnectionStatus({
    ...base,
    ready: true,
    waking: true,
    sleeping: true,
  }).state, 'sleeping')
  assert.deepEqual(realtimeConnectionStatus({
    ...base,
    ready: true,
    sleeping: true,
    blockedError: 'credential missing',
  }), {
    provider: 'dashscope',
    state: 'unavailable',
    error: 'credential missing',
  })
})
