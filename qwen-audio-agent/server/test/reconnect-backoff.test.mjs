import assert from 'node:assert/strict'
import test from 'node:test'
import { ReconnectBackoff } from '../src/voice/reconnect-backoff.mjs'

test('backs realtime reconnects off to a bounded delay', () => {
  const backoff = new ReconnectBackoff({
    baseMs: 500,
    maxMs: 2000,
    jitterRatio: 0,
  })

  assert.deepEqual(
    [backoff.next(), backoff.next(), backoff.next(), backoff.next()],
    [500, 1000, 2000, 2000],
  )
})

test('resets realtime reconnect delay after a stable connection', () => {
  const backoff = new ReconnectBackoff({
    baseMs: 250,
    maxMs: 2000,
    jitterRatio: 0,
  })

  backoff.next()
  backoff.next()
  backoff.reset()

  assert.equal(backoff.next(), 250)
})
