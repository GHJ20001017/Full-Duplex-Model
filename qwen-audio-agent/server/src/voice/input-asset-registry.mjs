import { createHash } from 'node:crypto'
import {
  inputFileParts,
  inputPartLabel,
  inputPartRef,
  parseDataUrl,
  withInputPartRef,
} from '../../../shared/input-parts.mjs'

function sessionKey(ownerId, sessionId) {
  return `${ownerId}\u0000${sessionId}`
}

function clonePart(part) {
  return {
    ...part,
    ...(part?.source
      ? {
          source: {
            ...part.source,
            ...(part.source.text ? { text: { ...part.source.text } } : {}),
          },
        }
      : {}),
    ...(part?._meta ? { _meta: { ...part._meta } } : {}),
  }
}

function fingerprint(part) {
  return createHash('sha256')
    .update(String(part?.mime || ''))
    .update('\u0000')
    .update(String(part?.url || ''))
    .digest('hex')
}

function partBytes(part) {
  return parseDataUrl(part?.url)?.bytes || 0
}

export class InputAssetRegistry {
  constructor({
    maxAssetsPerSession = 32,
    maxBytesPerSession = 64 * 1024 * 1024,
    maxSessions = 500,
    sessionTtlMs = 6 * 60 * 60 * 1000,
  } = {}) {
    this.maxAssetsPerSession = maxAssetsPerSession
    this.maxBytesPerSession = maxBytesPerSession
    this.maxSessions = maxSessions
    this.sessionTtlMs = sessionTtlMs
    this.sessions = new Map()
  }

  state(ownerId, sessionId) {
    this.prune()
    const key = sessionKey(ownerId, sessionId)
    let state = this.sessions.get(key)
    if (!state) {
      while (this.sessions.size >= this.maxSessions) {
        const oldest = [...this.sessions.entries()]
          .sort((left, right) => left[1].lastAccessedAt - right[1].lastAccessedAt)[0]
        if (!oldest) break
        this.sessions.delete(oldest[0])
      }
      state = {
        assets: new Map(),
        byFingerprint: new Map(),
        nextRef: 1,
        totalBytes: 0,
        lastAccessedAt: Date.now(),
      }
      this.sessions.set(key, state)
    }
    state.lastAccessedAt = Date.now()
    return state
  }

  peek(ownerId, sessionId) {
    return this.sessions.get(sessionKey(ownerId, sessionId)) || null
  }

  prune(now = Date.now()) {
    for (const [key, state] of this.sessions) {
      if (now - state.lastAccessedAt >= this.sessionTtlMs) {
        this.sessions.delete(key)
      }
    }
  }

  evict(state) {
    while (
      state.assets.size > this.maxAssetsPerSession
      || state.totalBytes > this.maxBytesPerSession
    ) {
      const oldest = state.assets.values().next().value
      if (!oldest) break
      state.assets.delete(oldest.ref)
      state.byFingerprint.delete(oldest.fingerprint)
      state.totalBytes = Math.max(0, state.totalBytes - oldest.bytes)
    }
  }

  registerParts({ ownerId, sessionId, turnId, parts = [] }) {
    const state = this.state(ownerId, sessionId)
    return parts.map(part => {
      if (part?.type !== 'file') return part
      const hash = fingerprint(part)
      const knownRef = state.byFingerprint.get(hash)
      const known = knownRef ? state.assets.get(knownRef) : null
      if (known) {
        known.lastSeenTurnId = turnId || known.lastSeenTurnId
        known.lastAccessedAt = Date.now()
        return withInputPartRef(part, known.ref)
      }
      const ref = `input_${state.nextRef++}`
      const enriched = withInputPartRef(part, ref)
      const asset = {
        ref,
        part: clonePart(enriched),
        fingerprint: hash,
        bytes: partBytes(part),
        originTurnId: turnId || null,
        lastSeenTurnId: turnId || null,
        createdAt: Date.now(),
        lastAccessedAt: Date.now(),
      }
      state.assets.set(ref, asset)
      state.byFingerprint.set(hash, ref)
      state.totalBytes += asset.bytes
      this.evict(state)
      return enriched
    })
  }

  resolve({ ownerId, sessionId, refs = [] }) {
    if (!Array.isArray(refs) || !refs.length) return []
    this.prune()
    const state = this.peek(ownerId, sessionId)
    if (!state) throw new Error('引用的输入已经失效')
    state.lastAccessedAt = Date.now()
    const unique = [...new Set(refs.map(ref => String(ref || '').trim()).filter(Boolean))]
    return unique.map(ref => {
      const asset = state.assets.get(ref)
      if (!asset) throw new Error(`找不到或无权访问输入引用：${ref}`)
      asset.lastAccessedAt = Date.now()
      return clonePart(asset.part)
    })
  }

  metadataForParts(parts = []) {
    return inputFileParts(parts).flatMap((part, index) => {
      const ref = inputPartRef(part)
      if (!ref) return []
      return [{
        ref,
        type: String(part.mime || '').startsWith('image/') ? 'image' : 'file',
        label: inputPartLabel(part, index),
        filename: part.filename || null,
        mime: part.mime,
      }]
    })
  }
}
