import assert from 'node:assert/strict'
import test from 'node:test'
import { EventEmitter } from 'node:events'
import {
  createOrbWindow,
  ORB_PRELOAD_PATH,
  ORB_WINDOW_SIZE,
  orbWindowOptions,
} from '../src/orb-window.mjs'

class FakeBrowserWindow extends EventEmitter {
  constructor(options) {
    super()
    this.options = options
    this.loaded = []
    this.destroyed = false
    this.flags = []
    this.webContents = Object.assign(new EventEmitter(), {
      setWindowOpenHandler: handler => {
        this.windowOpenHandler = handler
      },
    })
  }

  setAlwaysOnTop(...parameters) {
    this.flags.push(['top', ...parameters])
  }

  setVisibleOnAllWorkspaces(...parameters) {
    this.flags.push(['spaces', ...parameters])
  }

  async loadURL(url) {
    this.loaded.push(url)
  }

  show() {
    this.shown = true
  }

  destroy() {
    this.destroyed = true
    this.emit('closed')
  }
}

function fakeElectron() {
  const created = []
  return {
    created,
    BrowserWindow: class extends FakeBrowserWindow {
      constructor(options) {
        super(options)
        created.push(this)
      }
    },
  }
}

test('the recipe pins the orb form: frameless, sandboxed, floating size', () => {
  const options = orbWindowOptions({ position: { x: 10, y: 20 } })
  assert.equal(options.width, ORB_WINDOW_SIZE.width)
  assert.equal(options.frame, false)
  assert.equal(options.transparent, true)
  assert.equal(options.webPreferences.sandbox, true)
  assert.equal(options.webPreferences.contextIsolation, true)
  assert.equal(options.webPreferences.nodeIntegration, false)
  assert.equal(options.webPreferences.preload, ORB_PRELOAD_PATH)
  assert.deepEqual([options.x, options.y], [10, 20])
  // A host's partition rides into webPreferences for its own permission
  // arbitration.
  const partitioned = orbWindowOptions({ partition: 'persist:host' })
  assert.equal(partitioned.webPreferences.partition, 'persist:host')
})

test('the factory loads the page, applies flags, and tears down synchronously', async () => {
  const electron = fakeElectron()
  const externals = []
  const orb = await createOrbWindow({
    electron,
    pageUrl: () => 'http://127.0.0.1:4321/?desktop=orb&orbSkin=goo',
    placement: { initialPosition: () => ({ x: 5, y: 6 }) },
    onExternalUrl: url => externals.push(url),
  })
  const [window] = electron.created
  assert.deepEqual(window.loaded, ['http://127.0.0.1:4321/?desktop=orb&orbSkin=goo'])
  assert.deepEqual([window.options.x, window.options.y], [5, 6])
  assert.deepEqual(window.flags, [
    ['top', true, 'floating'],
    ['spaces', true, { visibleOnFullScreen: true }],
  ])

  // External links are denied and handed to the host.
  assert.deepEqual(
    window.windowOpenHandler({ url: 'https://example.com' }),
    { action: 'deny' },
  )
  assert.deepEqual(externals, ['https://example.com'])

  // Reload with a new URL — how a host applies a skin change.
  await orb.load('http://127.0.0.1:4321/?desktop=orb&orbSkin=firefly--lingxiaotian')
  assert.equal(window.loaded.length, 2)

  orb.destroy()
  assert.equal(window.destroyed, true)
  assert.equal(orb.disposed(), true)
  assert.equal(orb.window(), null)
  await assert.rejects(() => orb.load(), /destroyed/)
  // destroy is idempotent.
  orb.destroy()
})
