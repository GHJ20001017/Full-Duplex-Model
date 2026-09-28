import assert from 'node:assert/strict'
import test from 'node:test'
import { IdentityManager } from '../src/core/identity.mjs'

function response() {
  return {
    headers: {},
    setHeader(name, value) {
      this.headers[name.toLowerCase()] = value
    },
  }
}

function tokenFrom(res) {
  return res.headers['set-cookie'].match(/^qwen_audio_agent_identity=([^;]+)/)[1]
}

test('issues server-owned identities and rejects forged browser identifiers', () => {
  const manager = new IdentityManager({ secret: 'test-secret-that-is-longer-than-32-characters' })
  const firstResponse = response()
  const first = manager.resolveHttp({ headers: {}, socket: {} }, firstResponse)
  const token = tokenFrom(firstResponse)

  const recognized = manager.resolveUpgrade({
    headers: { cookie: `qwen_audio_agent_identity=${token}` },
  })
  const forged = manager.resolveUpgrade({
    headers: { cookie: 'qwen_audio_agent_identity=attacker-selected-client-id' },
  })

  assert.equal(recognized.ownerId, first.ownerId)
  assert.equal(forged, null)
  assert.match(firstResponse.headers['set-cookie'], /HttpOnly/)
  assert.match(firstResponse.headers['set-cookie'], /SameSite=Strict/)
})

test('keeps separate browsers in separate identities', () => {
  const manager = new IdentityManager({ secret: 'test-secret-that-is-longer-than-32-characters' })
  const first = manager.resolveHttp({ headers: {}, socket: {} }, response())
  const second = manager.resolveHttp({ headers: {}, socket: {} }, response())
  assert.notEqual(first.ownerId, second.ownerId)
})

test('recognizes the same signed browser identity after a server restart', () => {
  const secret = 'test-secret-that-is-longer-than-32-characters'
  const firstManager = new IdentityManager({ secret })
  const issuedResponse = response()
  const issued = firstManager.resolveHttp({ headers: {}, socket: {} }, issuedResponse)
  const token = tokenFrom(issuedResponse)
  const restartedManager = new IdentityManager({ secret })

  const recognized = restartedManager.resolveUpgrade({
    headers: { cookie: `qwen_audio_agent_identity=${token}` },
  })
  assert.equal(recognized.ownerId, issued.ownerId)
})

test('uses one stable owner across browsers and websocket upgrades in personal mode', () => {
  const manager = new IdentityManager({
    secret: 'test-secret-that-is-longer-than-32-characters',
    mode: 'personal',
    personalOwnerId: 'user_my_assistant',
  })

  const first = manager.resolveHttp({ headers: {}, socket: {} }, response())
  const second = manager.resolveHttp({ headers: {}, socket: {} }, response())
  const websocket = manager.resolveUpgrade({ headers: {} })

  assert.deepEqual(first, { ownerId: 'user_my_assistant' })
  assert.deepEqual(second, first)
  assert.deepEqual(websocket, first)
})

test('ignores malformed percent-encoded identity cookies without throwing', () => {
  const manager = new IdentityManager({
    secret: 'test-secret-that-is-longer-than-32-characters',
  })

  assert.equal(manager.resolveUpgrade({
    headers: { cookie: 'qwen_audio_agent_identity=%E0%A4%A' },
  }), null)
})
