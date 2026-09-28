import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export function webDistributionPath(
  moduleUrl = import.meta.url,
) {
  return resolve(dirname(fileURLToPath(moduleUrl)), '../../../web/dist')
}
