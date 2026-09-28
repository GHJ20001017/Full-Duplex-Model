import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'
import { webDistributionPath } from '../src/core/install-paths.mjs'

test('locates packaged WebUI independently of the process working directory', () => {
  const root = resolve('fixture-installation')
  const moduleUrl = pathToFileURL(
    resolve(root, 'server/src/core/install-paths.mjs'),
  )
  assert.equal(
    webDistributionPath(moduleUrl),
    resolve(root, 'web/dist'),
  )
})
