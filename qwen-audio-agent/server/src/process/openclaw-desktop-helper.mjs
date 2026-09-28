#!/usr/bin/env node

import {
  syncOpenClawGatewayTokenFile,
  writeIsolatedOpenClawConfig,
} from './backend-drivers/openclaw-auth.mjs'

const [, , command, ...args] = process.argv

if (command === 'prepare') {
  const [sourcePath, targetPath] = args
  if (!writeIsolatedOpenClawConfig({ sourcePath, targetPath })) {
    process.stderr.write('Unable to prepare isolated OpenClaw configuration.\n')
    process.exitCode = 1
  }
} else if (command === 'sync-token') {
  syncOpenClawGatewayTokenFile({ targetPath: args[0] })
} else {
  process.stderr.write(`Unknown OpenClaw desktop helper command: ${command || '(empty)'}\n`)
  process.exitCode = 1
}
