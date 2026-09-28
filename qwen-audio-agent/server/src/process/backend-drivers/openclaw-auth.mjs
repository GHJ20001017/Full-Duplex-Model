import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { dirname, resolve } from 'node:path'
import JSON5 from 'json5'

function clean(value) {
  return String(value || '').trim()
}

function configuredToken(value, env) {
  if (typeof value === 'string') {
    const reference = value.match(/^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$/)
    return reference ? clean(env[reference[1]]) : clean(value)
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  const source = clean(value.source || value.provider).toLowerCase()
  const environmentName = clean(
    value.id || value.name || value.key || value.env,
  )
  return source === 'env' && environmentName
    ? clean(env[environmentName])
    : ''
}

export function openClawConfigPath({
  env = process.env,
  homeDirectory = env.HOME,
} = {}) {
  if (clean(env.OPENCLAW_CONFIG_PATH)) {
    return resolve(clean(env.OPENCLAW_CONFIG_PATH))
  }
  const stateDirectory = clean(env.OPENCLAW_STATE_DIR)
    || (homeDirectory ? resolve(homeDirectory, '.openclaw') : '')
  return stateDirectory ? resolve(stateDirectory, 'openclaw.json') : ''
}

export function resolveOpenClawGatewayToken({
  env = process.env,
  homeDirectory = env.HOME,
} = {}) {
  const explicit = clean(env.OPENCLAW_GATEWAY_TOKEN)
  if (explicit) return explicit
  const path = openClawConfigPath({ env, homeDirectory })
  if (!path || !existsSync(path)) return ''
  try {
    const parsed = JSON5.parse(readFileSync(path, 'utf8'))
    return configuredToken(parsed?.gateway?.auth?.token, env)
  } catch {
    return ''
  }
}

export function syncOpenClawGatewayTokenFile({
  env = process.env,
  homeDirectory = env.HOME,
  targetPath,
} = {}) {
  const token = resolveOpenClawGatewayToken({ env, homeDirectory })
  const path = clean(targetPath || env.OPENCLAW_GATEWAY_TOKEN_FILE)
  if (!token || !path) return false
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
  const temporaryPath = `${path}.tmp.${process.pid}`
  writeFileSync(temporaryPath, `${token}\n`, { mode: 0o600 })
  chmodSync(temporaryPath, 0o600)
  renameSync(temporaryPath, path)
  return true
}

export function writeIsolatedOpenClawConfig({
  sourcePath,
  targetPath,
} = {}) {
  const source = clean(sourcePath)
  const target = clean(targetPath)
  if (!source || !target || !existsSync(source)) return false
  const parsed = JSON5.parse(readFileSync(source, 'utf8'))
  // A qwenaudio-owned Gateway only provides the local ACP backend. Reusing
  // channels would connect the user's messaging accounts a second time.
  delete parsed.channels
  mkdirSync(dirname(target), { recursive: true, mode: 0o700 })
  const temporaryPath = `${target}.tmp.${process.pid}`
  writeFileSync(temporaryPath, `${JSON.stringify(parsed, null, 2)}\n`, {
    mode: 0o600,
  })
  chmodSync(temporaryPath, 0o600)
  renameSync(temporaryPath, target)
  return true
}
