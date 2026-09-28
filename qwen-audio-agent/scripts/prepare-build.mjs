#!/usr/bin/env node
// Builds web/dist for npm lifecycle "prepare". Global installs from git
// (npm install -g git+...) leak npm_config_global and other npm_config_*
// values into nested npm processes, which breaks --workspace commands and
// skips the local node_modules this build needs. This script therefore runs
// npm with a sanitized environment and installs dependencies itself when the
// preparation environment did not provide them.
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

// The nested npm install below triggers "prepare" again; the guard keeps the
// recursion to a single level.
if (process.env.QWEN_AUDIO_AGENT_PREPARE === '1') process.exit(0)

const environment = { ...process.env, QWEN_AUDIO_AGENT_PREPARE: '1' }
for (const key of Object.keys(environment)) {
  if (key.toLowerCase().startsWith('npm_config_')) delete environment[key]
}

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
function run(args) {
  const result = spawnSync(npm, args, {
    cwd: root,
    env: environment,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  })
  if (result.status !== 0) process.exit(result.status || 1)
}

if (!existsSync(resolve(root, 'node_modules/.bin/vite'))) {
  run(['install', '--no-save', '--no-audit', '--include=dev'])
}
run(['run', 'build'])
