#!/usr/bin/env node

import { spawnSync } from 'node:child_process'
import {
  mkdtempSync,
  readdirSync,
  rmSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const stagingDirectory = mkdtempSync(join(tmpdir(), 'qwen-audio-agent-install-'))

function runNpm(args) {
  const npmExecutable = process.env.npm_execpath
  const command = npmExecutable ? process.execPath : (
    process.platform === 'win32' ? 'npm.cmd' : 'npm'
  )
  const commandArgs = npmExecutable
    ? [npmExecutable, ...args]
    : args
  const result = spawnSync(command, commandArgs, {
    cwd: root,
    env: process.env,
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  if (result.status !== 0) {
    throw new Error(`npm ${args.join(' ')} 执行失败`)
  }
}

function verifyPackage() {
  const result = spawnSync(
    process.execPath,
    [resolve(root, 'scripts/verify-package.mjs')],
    {
      cwd: root,
      env: process.env,
      stdio: 'inherit',
    },
  )
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error('npm 成品自检失败')
}

try {
  runNpm(['pack', '--pack-destination', stagingDirectory])
  verifyPackage()
  const archives = readdirSync(stagingDirectory)
    .filter(name => name.endsWith('.tgz'))
  if (archives.length !== 1) {
    throw new Error('无法确定 qwen-audio-agent 安装包')
  }
  runNpm(['install', '--global', join(stagingDirectory, archives[0])])
  process.stdout.write(
    'qwen-audio-agent 已作为独立成品安装；运行 qwenaudio config 开始配置。\n',
  )
} catch (error) {
  process.stderr.write(`全局安装失败：${error.message}\n`)
  process.exitCode = 1
} finally {
  rmSync(stagingDirectory, { recursive: true, force: true })
}
