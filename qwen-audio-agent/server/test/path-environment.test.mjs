import assert from 'node:assert/strict'
import test from 'node:test'
import {
  commandDirectory,
  mergeSearchPath,
} from '../../shared/path-environment.mjs'

test('merges POSIX search paths with colon delimiters', () => {
  assert.equal(
    mergeSearchPath('/usr/bin:/bin', '/opt/homebrew/bin:/usr/bin', {
      platform: 'darwin',
    }),
    '/opt/homebrew/bin:/usr/bin:/bin',
  )
  assert.equal(commandDirectory('/opt/homebrew/bin/npm', 'darwin'), '/opt/homebrew/bin')
})

test('merges Windows search paths case-insensitively', () => {
  assert.equal(
    mergeSearchPath('C:\\Windows;C:\\Node', 'c:\\node;D:\\Tools', {
      platform: 'win32',
    }),
    'D:\\Tools;C:\\Windows;C:\\Node',
  )
  assert.equal(commandDirectory('C:\\Node\\npm.cmd', 'win32'), 'C:\\Node')
})

test('can append a user-selected executable directory on every platform', () => {
  assert.equal(
    mergeSearchPath('/usr/bin:/bin', '/opt/node/bin', {
      platform: 'linux',
      prepend: false,
    }),
    '/usr/bin:/bin:/opt/node/bin',
  )
  assert.equal(
    mergeSearchPath('C:\\Windows;C:\\Node', 'c:\\node', {
      platform: 'win32',
      prepend: false,
    }),
    'C:\\Windows;C:\\Node',
  )
})
