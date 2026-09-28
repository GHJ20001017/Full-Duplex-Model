import { dirname, win32 } from 'node:path'

export function pathDelimiter(platform = process.platform) {
  return platform === 'win32' ? ';' : ':'
}

export function commandDirectory(command, platform = process.platform) {
  return platform === 'win32'
    ? win32.dirname(String(command || ''))
    : dirname(String(command || ''))
}

export function mergeSearchPath(currentValue, incomingValue, {
  platform = process.platform,
  prepend = true,
} = {}) {
  const delimiter = pathDelimiter(platform)
  const key = value => platform === 'win32' ? value.toLowerCase() : value
  const current = String(currentValue || '')
    .split(delimiter)
    .map(value => value.trim())
    .filter(Boolean)
  const known = new Set(current.map(key))
  const incoming = String(incomingValue || '')
    .split(delimiter)
    .map(value => value.trim())
    .filter(Boolean)
    .filter(value => {
      const normalized = key(value)
      if (known.has(normalized)) return false
      known.add(normalized)
      return true
    })
  return (prepend ? [...incoming, ...current] : [...current, ...incoming])
    .join(delimiter)
}
