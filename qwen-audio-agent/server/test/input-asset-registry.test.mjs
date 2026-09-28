import assert from 'node:assert/strict'
import test from 'node:test'
import { InputAssetRegistry } from '../src/voice/input-asset-registry.mjs'
import { inputPartRef } from '../../shared/input-parts.mjs'

const image = {
  type: 'file',
  mime: 'image/png',
  filename: 'cat.png',
  url: 'data:image/png;base64,aGVsbG8=',
  source: { type: 'clipboard', text: { value: '[Image 1]' } },
}

test('assigns stable session-scoped references and exposes safe metadata', () => {
  const registry = new InputAssetRegistry()
  const [first] = registry.registerParts({
    ownerId: 'owner',
    sessionId: 'voice',
    turnId: 'turn-one',
    parts: [image],
  })
  const [again] = registry.registerParts({
    ownerId: 'owner',
    sessionId: 'voice',
    turnId: 'turn-two',
    parts: [image],
  })

  assert.equal(inputPartRef(first), 'input_1')
  assert.equal(inputPartRef(again), 'input_1')
  assert.deepEqual(registry.metadataForParts([first]), [{
    ref: 'input_1',
    type: 'image',
    label: '[Image 1]',
    filename: 'cat.png',
    mime: 'image/png',
  }])
})

test('resolves assets only inside the owning conversation', () => {
  const registry = new InputAssetRegistry()
  const [registered] = registry.registerParts({
    ownerId: 'owner',
    sessionId: 'voice',
    turnId: 'turn-one',
    parts: [image],
  })

  assert.deepEqual(
    registry.resolve({
      ownerId: 'owner',
      sessionId: 'voice',
      refs: ['input_1'],
    }),
    [registered],
  )
  assert.throws(() => registry.resolve({
    ownerId: 'other-owner',
    sessionId: 'voice',
    refs: ['input_1'],
  }), /已经失效/)
  assert.throws(() => registry.resolve({
    ownerId: 'owner',
    sessionId: 'other-session',
    refs: ['input_1'],
  }), /已经失效/)
})

test('expires conversation assets without retaining stale references', () => {
  const registry = new InputAssetRegistry({ sessionTtlMs: 1 })
  registry.registerParts({
    ownerId: 'owner',
    sessionId: 'voice',
    parts: [image],
  })
  const state = registry.peek('owner', 'voice')
  state.lastAccessedAt = 0

  assert.throws(() => registry.resolve({
    ownerId: 'owner',
    sessionId: 'voice',
    refs: ['input_1'],
  }), /已经失效/)
})
