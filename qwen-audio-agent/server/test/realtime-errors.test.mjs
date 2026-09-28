import assert from 'node:assert/strict'
import test from 'node:test'
import { isRecoverableRealtimeInactivityError } from '../src/voice/realtime-errors.mjs'

test('recognizes the provider inactivity closure as recoverable', () => {
  assert.equal(
    isRecoverableRealtimeInactivityError(
      'Your session was closed because no response was generated for 180 seconds.',
    ),
    true,
  )
  assert.equal(
    isRecoverableRealtimeInactivityError(
      'Your session was closed because no response was generated for 300 seconds',
    ),
    true,
  )
})

test('does not hide actionable realtime errors', () => {
  assert.equal(
    isRecoverableRealtimeInactivityError(
      'Cannot create response while user is speaking.',
    ),
    false,
  )
  assert.equal(
    isRecoverableRealtimeInactivityError('Authentication failed'),
    false,
  )
})
