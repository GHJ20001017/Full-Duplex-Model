import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_INPUT_SUSPEND_TTL_MS,
  InputArbitration,
  MAX_INPUT_SUSPEND_TTL_MS,
} from '../src/voice/input-arbitration.mjs'

// Deterministic clock and timers: the whole point of this class is timeout
// behaviour, so the tests must not depend on real time.
function controlled() {
  let clock = 1000
  const timers = new Map()
  let nextTimer = 0
  const arbitration = new InputArbitration({
    now: () => clock,
    setTimer: (callback, delay) => {
      const id = ++nextTimer
      timers.set(id, { callback, firesAt: clock + delay })
      return id
    },
    clearTimer: id => timers.delete(id),
  })
  return {
    arbitration,
    advance(ms) {
      clock += ms
      for (const [id, timer] of [...timers]) {
        if (timer.firesAt <= clock) {
          timers.delete(id)
          timer.callback()
        }
      }
    },
    get pendingTimers() {
      return timers.size
    },
  }
}

test('a suspension reports its holder and resumes on request', () => {
  const { arbitration } = controlled()
  const changes = []
  arbitration.subscribe((status, change) => changes.push([status.suspended, change.state]))

  assert.equal(arbitration.suspended, false)
  const suspended = arbitration.suspend({
    owner: 'cosyvoice',
    reason: 'dictation',
    ttlMs: 5000,
  })
  assert.equal(suspended.suspended, true)
  assert.equal(suspended.owner, 'cosyvoice')
  assert.equal(suspended.reason, 'dictation')

  arbitration.resume({ owner: 'cosyvoice' })
  assert.equal(arbitration.suspended, false)
  assert.deepEqual(changes, [[true, 'suspended'], [false, 'resumed']])
})

test('suspend requires an owner so a hold can always be attributed', () => {
  const { arbitration } = controlled()
  assert.throws(
    () => arbitration.suspend({ reason: 'dictation' }),
    error => error.code === 'QWAUDIO_INPUT_OWNER_REQUIRED',
  )
  assert.equal(arbitration.suspended, false)
})

test('repeating suspend for one owner refreshes it instead of stacking', () => {
  const controls = controlled()
  const { arbitration } = controls
  arbitration.suspend({ owner: 'cosyvoice', ttlMs: 1000 })
  const first = arbitration.status().holders[0]
  assert.equal(controls.pendingTimers, 1)

  controls.advance(500)
  arbitration.suspend({ owner: 'cosyvoice', ttlMs: 1000 })
  const status = arbitration.status()
  assert.equal(status.holders.length, 1)
  // The original start time survives; only the deadline moves out.
  assert.equal(status.holders[0].since, first.since)
  assert.ok(status.holders[0].expiresAt > first.expiresAt)

  // A single resume releases it: a host that re-announced on every keypress
  // cannot leave an unreleasable hold behind.
  arbitration.resume({ owner: 'cosyvoice' })
  assert.equal(arbitration.suspended, false)
})

test('holds from several subsystems compose and the last release wins', () => {
  const { arbitration } = controlled()
  const changes = []
  arbitration.subscribe((status, change) => changes.push([status.suspended, change.state]))

  arbitration.suspend({ owner: 'dictation' })
  arbitration.suspend({ owner: 'shortcut-overlay' })
  assert.equal(arbitration.status().holders.length, 2)

  arbitration.resume({ owner: 'dictation' })
  assert.equal(arbitration.suspended, true)
  arbitration.resume({ owner: 'shortcut-overlay' })
  assert.equal(arbitration.suspended, false)
  // Clients are told once on the way in and once on the way out, not per hold.
  assert.deepEqual(changes, [[true, 'suspended'], [false, 'resumed']])
})

test('a hold expires on its own when the host never resumes', () => {
  const { arbitration, advance } = controlled()
  const changes = []
  arbitration.subscribe((status, change) => changes.push([status.suspended, change]))

  arbitration.suspend({ owner: 'cosyvoice', ttlMs: 2000 })
  advance(1999)
  assert.equal(arbitration.suspended, true)
  advance(1)
  assert.equal(arbitration.suspended, false)
  assert.equal(changes.at(-1)[1].expired, true)
})

test('a ttl is clamped to a supported window', () => {
  const { arbitration } = controlled()
  arbitration.suspend({ owner: 'a' })
  assert.equal(arbitration.status().holders[0].ttlMs, DEFAULT_INPUT_SUSPEND_TTL_MS)

  arbitration.suspend({ owner: 'b', ttlMs: 0 })
  assert.equal(
    arbitration.status().holders.find(holder => holder.owner === 'b').ttlMs,
    DEFAULT_INPUT_SUSPEND_TTL_MS,
  )

  arbitration.suspend({ owner: 'c', ttlMs: Number.MAX_SAFE_INTEGER })
  assert.equal(
    arbitration.status().holders.find(holder => holder.owner === 'c').ttlMs,
    MAX_INPUT_SUSPEND_TTL_MS,
  )
})

test('resuming an owner that holds nothing is a no-op', () => {
  const { arbitration } = controlled()
  const changes = []
  arbitration.subscribe((status, change) => changes.push(change))
  assert.equal(arbitration.resume({ owner: 'nobody' }).suspended, false)
  arbitration.suspend({ owner: 'cosyvoice' })
  arbitration.resume({ owner: 'someone-else' })
  assert.equal(arbitration.suspended, true)
  assert.deepEqual(changes.map(change => change.state), ['suspended'])
})

test('releaseAll drops holds but keeps subscribers for the next start', () => {
  const controls = controlled()
  const { arbitration } = controls
  const changes = []
  arbitration.subscribe((status, change) => changes.push(change.state))
  arbitration.suspend({ owner: 'cosyvoice', ttlMs: 1000 })
  arbitration.suspend({ owner: 'overlay', ttlMs: 1000 })

  arbitration.releaseAll()
  assert.equal(arbitration.suspended, false)
  assert.equal(controls.pendingTimers, 0)
  // No expiry callback may fire for a released hold.
  controls.advance(5000)

  arbitration.suspend({ owner: 'cosyvoice' })
  assert.deepEqual(changes, ['suspended', 'resumed', 'suspended'])
})

test('a listener that throws cannot break arbitration for the others', () => {
  const { arbitration } = controlled()
  const seen = []
  arbitration.subscribe(() => {
    throw new Error('client dispatch failed')
  })
  arbitration.subscribe(status => seen.push(status.suspended))
  arbitration.suspend({ owner: 'cosyvoice' })
  assert.deepEqual(seen, [true])
})
