import assert from 'node:assert/strict'
import test from 'node:test'
import {
  validateDesktopReleaseEnvironment,
} from '../../scripts/check-desktop-release-env.mjs'

test('requires signing and notarization for a formal desktop build', () => {
  assert.throws(
    () => validateDesktopReleaseEnvironment({}),
    /正式 macOS 构建缺少/,
  )
  assert.doesNotThrow(() => validateDesktopReleaseEnvironment({
    CSC_LINK: 'certificate.p12',
    APPLE_ID: 'release@example.com',
    APPLE_APP_SPECIFIC_PASSWORD: 'app-password',
    APPLE_TEAM_ID: 'TEAMID',
  }))
  assert.doesNotThrow(() => validateDesktopReleaseEnvironment({
    CSC_NAME: 'Developer ID Application',
    APPLE_API_KEY: '/secure/AuthKey.p8',
    APPLE_API_KEY_ID: 'KEYID',
    APPLE_API_ISSUER: 'ISSUER',
  }))
})
