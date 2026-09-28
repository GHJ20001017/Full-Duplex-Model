// The Gateway, CLI, TUI and desktop main process must keep running on the
// oldest Node named in the engines range. Our own machines and most CI jobs
// run newer versions, so a single newer API in a shipped code path would fail
// on a user's baseline install at runtime rather than at install time.
//
// A version range in package.json alone cannot enforce that, so this test is
// the guard: it scans the code that actually ships for APIs that only exist
// above the baseline. CI additionally runs the whole suite on the baseline
// version, which is what proves the baseline rather than asserting it.

import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, relative, resolve, sep } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const selfPath = fileURLToPath(import.meta.url)
const projectRoot = resolve(dirname(selfPath), '..')

// The declared runtime baseline: the lowest version admitted by the engines
// field. Keep in sync with package.json and the Node matrix in
// .github/workflows/ci.yml.
const RUNTIME_BASELINE = '22.22.2'

// Directories whose code runs on a user's Node (or the desktop app's bundled
// Electron), not on our build machine. The build toolchain (vite, eslint,
// electron-builder) and the browser bundle (web/src) are deliberately
// excluded — users never run them on their own Node.
const RUNTIME_ROOTS = [
  'server/src',
  'shared',
  'cli/src',
  'cli/bin',
  'tui/src',
  'desktop/src',
]

// APIs newer than the baseline. Each entry names the Node version that
// introduced it, so a future baseline bump can retire the entry instead of
// guessing why it was listed.
const ABOVE_BASELINE = [
  { pattern: /\bPromise\.try\b/, since: '23.0', name: 'Promise.try' },
  { pattern: /\bRegExp\.escape\b/, since: '23.0', name: 'RegExp.escape' },
  { pattern: /\bError\.isError\b/, since: '24.0', name: 'Error.isError' },
  { pattern: /\bFloat16Array\b/, since: '24.0', name: 'Float16Array' },
  { pattern: /\bUint8Array\.fromBase64\b/, since: '24.0', name: 'Uint8Array.fromBase64' },
  { pattern: /\.toBase64\s*\(/, since: '24.0', name: 'Uint8Array.prototype.toBase64' },
  { pattern: /\bMath\.sumPrecise\b/, since: '24.0', name: 'Math.sumPrecise' },
  { pattern: /\bIterator\.range\b/, since: '25.0', name: 'Iterator.range' },
]

function sourceFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = resolve(directory, entry.name)
    if (entry.isDirectory()) {
      return entry.name === 'node_modules' ? [] : sourceFiles(path)
    }
    return [path]
  }).filter(path => (
    /\.(?:[cm]?js|jsx)$/.test(path)
    // This file names every API it forbids, so scanning it would always fail.
    && path !== selfPath
  ))
}

// Prose must not trip the scan: a comment explaining why an API is avoided
// would otherwise read as a use of it.
function code(content) {
  return content
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
}

test('runtime code stays within the supported Node baseline', () => {
  const violations = []
  for (const root of RUNTIME_ROOTS) {
    for (const file of sourceFiles(resolve(projectRoot, root))) {
      const content = code(readFileSync(file, 'utf8'))
      for (const api of ABOVE_BASELINE) {
        if (!api.pattern.test(content)) continue
        violations.push(
          `${relative(projectRoot, file).split(sep).join('/')}`
          + ` uses ${api.name} (Node ${api.since}+, baseline is ${RUNTIME_BASELINE})`,
        )
      }
    }
  }
  assert.deepEqual(violations, [])
})

test('the declared engines range admits the runtime baseline', () => {
  const manifest = JSON.parse(
    readFileSync(resolve(projectRoot, 'package.json'), 'utf8'),
  )
  // A user cannot follow our development Node version, so the baseline has to
  // be an advertised, supported range rather than an accident that happens to
  // work. The engines minimum and this test's baseline must move together.
  assert.match(String(manifest.engines?.node || ''), /\^?22\.22\.2/)
  const [major, minor, patch] = RUNTIME_BASELINE.split('.')
  assert.ok(Number(major) >= 22, `baseline major ${major} predates engines`)
  assert.ok(Number.isInteger(Number(minor)) && Number.isInteger(Number(patch)))
})
