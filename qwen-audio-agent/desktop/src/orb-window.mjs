// The orb window recipe and a host-facing factory.
//
// orbWindowOptions is the single source for the BrowserWindow options the
// orb form requires (frameless, transparent, floating, sandboxed); the desktop
// shell builds its own window from the same recipe, which keeps the two
// surfaces from drifting apart. createOrbWindow assembles the recipe, the
// window policy and placement for an embedding host that does not need the
// desktop shell's extra behaviour.
//
// electron is injected (or imported lazily), so the module stays loadable in
// plain Node for tests.
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { configureOrbWindow } from './orb-shell.mjs'
import {
  DESKTOP_ORB_HEIGHT,
  DESKTOP_ORB_WIDTH,
} from './desktop-surface-layout.mjs'

const here = dirname(fileURLToPath(import.meta.url))

// The renderer bridge the orb page expects (window.qwenAudioAgentDesktop).
export const ORB_PRELOAD_PATH = resolve(here, 'preload.cjs')

export const ORB_WINDOW_SIZE = Object.freeze({
  width: DESKTOP_ORB_WIDTH,
  height: DESKTOP_ORB_HEIGHT,
})

/**
 * The BrowserWindow options of the orb form.
 *
 * @param {object} options
 * @param {{x: number, y: number}} options.position Initial position, usually
 *   from createOrbPlacement().initialPosition().
 * @param {{width: number, height: number}} [options.maxSize] Work-area bounds.
 * @param {string} [options.preload] Preload path; defaults to the shipped one.
 * @param {string} [options.partition] Session partition; hosts pass their own
 *   so microphone permission is granted to the host app's identity.
 */
export function orbWindowOptions({
  position,
  maxSize = null,
  preload = ORB_PRELOAD_PATH,
  partition = '',
} = {}) {
  return {
    width: ORB_WINDOW_SIZE.width,
    height: ORB_WINDOW_SIZE.height,
    minWidth: ORB_WINDOW_SIZE.width,
    minHeight: ORB_WINDOW_SIZE.height,
    ...(maxSize
      ? { maxWidth: maxSize.width, maxHeight: maxSize.height }
      : {}),
    ...(position ? { x: position.x, y: position.y } : {}),
    frame: false,
    transparent: true,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    alwaysOnTop: true,
    hasShadow: false,
    backgroundColor: '#00000000',
    title: 'qwen-audio-agent',
    autoHideMenuBar: true,
    skipTaskbar: true,
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // The floating window is normally unfocused. Keep its renderer timers
      // aligned with Web Audio so playback receipts are not delayed.
      backgroundThrottling: false,
      preload,
      ...(partition ? { partition } : {}),
    },
  }
}

/**
 * Creates the orb window for an embedding host: recipe, floating flags,
 * navigation policy and initial placement. IPC binding stays separate —
 * combine with bindOrbShell, DesktopPresence and createOrbPlacement.
 *
 * @param {object} options
 * @param {string|() => string} options.pageUrl The orb page URL (usually the
 *   Gateway origin plus desktopOrbUrl parameters), re-evaluated on load().
 * @param {object} [options.electron] Electron module; imported lazily when
 *   omitted (the default only works inside an Electron main process).
 * @param {object} [options.placement] createOrbPlacement() result; supplies
 *   the initial position and records drops via the shell's onDragEnd.
 * @param {string} [options.partition] Session partition for the host's own
 *   permission arbitration.
 * @param {(url: string) => void} [options.onExternalUrl] Receives external
 *   links the page tries to open; the window itself always denies them.
 * @returns {Promise<{ window: () => BrowserWindow, load: (url?: string) => Promise<void>, destroy: () => void, disposed: () => boolean }>}
 */
export async function createOrbWindow({
  pageUrl,
  electron = null,
  placement = null,
  partition = '',
  preload = ORB_PRELOAD_PATH,
  windowOptions = {},
  onExternalUrl = null,
  onClosed = null,
} = {}) {
  if (!pageUrl) throw new Error('createOrbWindow: pageUrl is required')
  const { BrowserWindow } = electron || await import('electron')
  const resolveUrl = () => String(
    typeof pageUrl === 'function' ? pageUrl() : pageUrl,
  )

  const window = new BrowserWindow({
    ...orbWindowOptions({
      position: placement?.initialPosition?.() || null,
      preload,
      partition,
    }),
    ...windowOptions,
  })
  configureOrbWindow(window)

  // The orb page never navigates and never opens windows; external links go
  // to the host, which decides how to open them.
  window.webContents.setWindowOpenHandler(({ url }) => {
    onExternalUrl?.(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event, url) => {
    if (url === resolveUrl()) return
    event.preventDefault()
    onExternalUrl?.(url)
  })
  window.once('ready-to-show', () => window.show())

  let disposed = false
  window.on('closed', () => {
    disposed = true
    onClosed?.()
  })

  await window.loadURL(resolveUrl())

  return {
    window: () => (disposed ? null : window),
    async load(url = '') {
      if (disposed) throw new Error('orb window already destroyed')
      await window.loadURL(url || resolveUrl())
    },
    // Synchronous teardown: renderer exit is the only dependable way to
    // release the microphone, so a host must call this when it stops the
    // component rather than merely hiding the window.
    destroy() {
      if (disposed) return
      disposed = true
      window.destroy()
    },
    disposed: () => disposed,
  }
}
