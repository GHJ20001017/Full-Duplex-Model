import fs from 'node:fs'
import { isAbsolute } from 'node:path'

const SECRET = /(?:authorization|authentication|credential|password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|cookie|private[_-]?key)|^(?:token|auth|headers)$/i
const BLOB = /^(?:audio|audio_data|audio_base64|input_audio|image|image_url|image_data|data|bytes|blob|base64)$/i
const MAX_STRING = 32_768
const MAX_NODES = 4096

// Redact before bounding, including JSON encoded tool arguments/results.
export function sanitizeDelegationTrace(value) {
  let nodes = 0
  const visit = (item, depth = 0, key = '') => {
    if (++nodes > MAX_NODES || depth > 16) return '[TRUNCATED: structural limit]'
    if (SECRET.test(key) || BLOB.test(key)) return '[REDACTED]'
    if (typeof item === 'string') {
      if (['arguments', 'output'].includes(key)) {
        try { return visit(JSON.parse(item), depth + 1) } catch { /* Plain text. */ }
      }
      const clean = item
        .replace(/(["'](?:api[_-]?key|token|secret|password|authorization|credential|audio|data)["']\s*:\s*)["'][^"']*["']/gi, '$1"[REDACTED]"')
        .replace(/\b(Bearer|Basic)\s+[^\s"',;]+/gi, '$1 [REDACTED]')
        .replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/gi, '$1[REDACTED]@')
        .replace(/((?:api[_-]?key|token|secret|password|authorization|credential)\s*[=:]\s*)[^\s&,;"']+/gi, '$1[REDACTED]')
        .replace(/data:(?:audio|image)\/[^\s"']+/gi, '[REDACTED media data URI]')
        .replace(/-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----/g, '[REDACTED PRIVATE KEY]')
      return clean.length > MAX_STRING
        ? { text: clean.slice(0, MAX_STRING), truncated: true, originalChars: clean.length, retainedChars: MAX_STRING }
        : clean
    }
    if (item == null || typeof item === 'boolean' || typeof item === 'number') return item
    if (ArrayBuffer.isView(item)) return '[REDACTED binary]'
    if (Array.isArray(item)) {
      const result = item.slice(0, 256).map(entry => visit(entry, depth + 1))
      if (item.length > 256) result.push(`[TRUNCATED: ${item.length - 256} entries]`)
      return result
    }
    if (typeof item !== 'object') return String(item)
    const result = {}
    const entries = Object.entries(item)
    for (const [name, entry] of entries.slice(0, 256)) {
      result[name] = visit(entry, depth + 1, name)
    }
    if (entries.length > 256) result.__truncatedKeys = entries.length - 256
    return result
  }
  return visit(value)
}

export function createDelegationTrace({ path = process.env.QWEN_AUDIO_DELEGATION_TRACE_PATH, context = () => ({}), io = fs } = {}) {
  if (!path || !isAbsolute(path)) return () => {}
  let warned = false
  let announced = false
  return (boundary, details = {}) => {
    let fd
    try {
      const record = sanitizeDelegationTrace({ timestamp: new Date().toISOString(), boundary, ...context(), ...details })
      let line = JSON.stringify(record)
      if (Buffer.byteLength(line) > 262_144) {
        line = JSON.stringify({ timestamp: record.timestamp, boundary, sessionId: record.sessionId, connectionId: record.connectionId, turnId: record.turnId, responseId: record.responseId, truncated: true, originalBytes: Buffer.byteLength(line), reason: 'record exceeds 262144 bytes; payload omitted' })
      }
      fd = io.openSync(path, fs.constants.O_WRONLY | fs.constants.O_APPEND | fs.constants.O_CREAT | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK, 0o600)
      const stat = io.fstatSync(fd)
      if (!stat.isFile() || stat.nlink !== 1 || stat.uid !== process.getuid?.()) throw new Error('unsafe trace target')
      io.fchmodSync(fd, 0o600)
      io.writeFileSync(fd, `${line}\n`)
      if (!announced) {
        announced = true
        try { console.info('[delegation-trace] JSONL active: filtered wire events and dispatch boundaries; private text stays in file') } catch { /* Fail open. */ }
      }
    } catch {
      if (!warned) {
        warned = true
        try { console.warn('[delegation-trace] write failed; diagnostics skipped (runtime unaffected)') } catch { /* Fail open. */ }
      }
    } finally {
      if (fd !== undefined) { try { io.closeSync(fd) } catch { /* Fail open. */ } }
    }
  }
}

export function delegationWirePayload(event) {
  const type = event?.type || ''
  if (type === 'session.update' || type === 'session.created' || type === 'session.updated') {
    const { id, instructions, tools, tool_choice } = event.session || {}
    return { type, event_id: event.event_id, session: { id, instructions, tools, tool_choice } }
  }
  if (['response.create', 'response.created', 'response.done', 'response.cancel', 'error',
    'conversation.item.create', 'conversation.item.created',
    'conversation.item.input_audio_transcription.completed', 'conversation.item.input_audio_transcription.failed',
    'response.text.done', 'response.output_text.done', 'response.audio_transcript.done', 'response.output_audio_transcript.done',
    'response.function_call_arguments.done', 'input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.committed'].includes(type)) return event
  if (['response.output_item.added', 'response.output_item.done'].includes(type) && event.item?.type === 'function_call') return event
  return null
}
