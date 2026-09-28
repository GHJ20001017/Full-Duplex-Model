#!/usr/bin/env node

import { fileURLToPath } from 'node:url'

export function validateDesktopReleaseEnvironment(env = process.env) {
  const signingConfigured = Boolean(env.CSC_LINK || env.CSC_NAME)
  const appleIdNotarization = Boolean(
    env.APPLE_ID
    && env.APPLE_APP_SPECIFIC_PASSWORD
    && env.APPLE_TEAM_ID,
  )
  const apiKeyNotarization = Boolean(
    env.APPLE_API_KEY
    && env.APPLE_API_KEY_ID
    && env.APPLE_API_ISSUER,
  )
  const missing = []
  if (!signingConfigured) {
    missing.push('CSC_LINK（或已安装证书对应的 CSC_NAME）')
  }
  if (!appleIdNotarization && !apiKeyNotarization) {
    missing.push(
      'Apple 公证凭据（APPLE_ID 组合或 APPLE_API_KEY 组合）',
    )
  }
  if (missing.length) {
    throw new Error(
      `正式 macOS 构建缺少：${missing.join('、')}。`
      + ' 本机未签名测试请运行 npm run desktop:build:local。',
    )
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    validateDesktopReleaseEnvironment()
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 1
  }
}
