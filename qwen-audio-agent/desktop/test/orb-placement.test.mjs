import assert from 'node:assert/strict'
import test from 'node:test'
import {
  createOrbPlacement,
  ORB_PLACEMENT_MARGIN,
} from '../src/orb-placement.mjs'

const ORB = { width: 172, height: 204 }
const PRIMARY = {
  id: 1,
  workArea: { x: 0, y: 25, width: 1512, height: 920 },
}
const SECONDARY = {
  id: 2,
  workArea: { x: 1512, y: 0, width: 2560, height: 1415 },
}

function placement({ displays = [PRIMARY], saved = null, onSave } = {}) {
  return createOrbPlacement({
    getDisplays: () => displays,
    orbSize: ORB,
    loadState: () => saved,
    saveState: onSave || (() => {}),
  })
}

test('defaults to the primary work area top-right with a margin', () => {
  const position = placement().initialPosition()
  assert.deepEqual(position, {
    x: PRIMARY.workArea.x + PRIMARY.workArea.width - ORB.width - ORB_PLACEMENT_MARGIN,
    y: PRIMARY.workArea.y + ORB_PLACEMENT_MARGIN,
  })
})

test('restores the saved position where the user left the orb', () => {
  const position = placement({
    saved: { x: 300, y: 400, displayId: 1 },
  }).initialPosition()
  assert.deepEqual(position, { x: 300, y: 400 })
})

test('clamps a saved position back into the hosting display', () => {
  // The orb was left near the bottom edge and the work area then shrank.
  const position = placement({
    saved: { x: 1400, y: 900, displayId: 1 },
  }).initialPosition()
  assert.deepEqual(position, {
    x: PRIMARY.workArea.x + PRIMARY.workArea.width - ORB.width,
    y: PRIMARY.workArea.y + PRIMARY.workArea.height - ORB.height,
  })
})

test('a position on an unplugged display moves to the nearest one', () => {
  // Saved on the secondary display, which is gone now.
  const position = placement({
    displays: [PRIMARY],
    saved: { x: 3000, y: 500, displayId: 2 },
  }).initialPosition()
  const { workArea } = PRIMARY
  assert.ok(position.x >= workArea.x)
  assert.ok(position.x + ORB.width <= workArea.x + workArea.width)
  assert.ok(position.y >= workArea.y)
  assert.ok(position.y + ORB.height <= workArea.y + workArea.height)
})

test('a corrupt state store falls back to the default anchor', () => {
  const position = createOrbPlacement({
    getDisplays: () => [PRIMARY],
    orbSize: ORB,
    loadState: () => {
      throw new Error('corrupt ui-state.json')
    },
  }).initialPosition()
  assert.equal(position.y, PRIMARY.workArea.y + ORB_PLACEMENT_MARGIN)
})

test('records the dragged position with its hosting display', () => {
  const saves = []
  const record = placement({
    displays: [PRIMARY, SECONDARY],
    onSave: state => saves.push(state),
  })
  assert.equal(record.recordPosition({ x: 2000, y: 300 }), true)
  assert.deepEqual(saves, [{ x: 2000, y: 300, displayId: 2 }])
  // Invalid positions are refused rather than persisted.
  assert.equal(record.recordPosition({ x: Number.NaN, y: 10 }), false)
  assert.equal(saves.length, 1)
})

test('a failing state store never breaks the drag itself', () => {
  const record = placement({
    onSave: () => {
      throw new Error('disk full')
    },
  })
  assert.equal(record.recordPosition({ x: 10, y: 30 }), false)
})
