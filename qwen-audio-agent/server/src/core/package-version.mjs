import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const manifestPath = fileURLToPath(
  new URL('../../../package.json', import.meta.url),
)

export const PACKAGE_VERSION = JSON.parse(
  readFileSync(manifestPath, 'utf8'),
).version
