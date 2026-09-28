import {
  createHmac,
  randomUUID,
  timingSafeEqual,
} from 'node:crypto'

const COOKIE_NAME = 'qwen_audio_agent_identity'
const MAX_AGE_SECONDS = 7 * 24 * 60 * 60

function cookies(header = '') {
  return Object.fromEntries(
    header
      .split(';')
      .map(part => part.trim())
      .filter(Boolean)
      .map(part => {
        const separator = part.indexOf('=')
        if (separator < 0) return [part, '']
        try {
          return [part.slice(0, separator), decodeURIComponent(part.slice(separator + 1))]
        } catch {
          return [part.slice(0, separator), '']
        }
      }),
  )
}

export class IdentityManager {
  constructor({
    secret,
    cookieName = COOKIE_NAME,
    mode = 'browser',
    personalOwnerId = 'user_personal',
  } = {}) {
    if (!secret || String(secret).length < 32) {
      throw new Error('QWEN_AUDIO_AGENT_AUTH_SECRET must contain at least 32 characters')
    }
    this.secret = String(secret)
    this.cookieName = cookieName
    this.mode = mode === 'personal' ? 'personal' : 'browser'
    this.personalIdentity = {
      ownerId: String(personalOwnerId || 'user_personal'),
    }
  }

  sign(ownerId) {
    return createHmac('sha256', this.secret).update(ownerId).digest('base64url')
  }

  resolveToken(header) {
    const token = cookies(header)[this.cookieName]
    if (!token) return null
    const separator = token.lastIndexOf('.')
    if (separator < 0) return null
    const ownerId = token.slice(0, separator)
    const supplied = Buffer.from(token.slice(separator + 1))
    const expected = Buffer.from(this.sign(ownerId))
    if (
      !ownerId.startsWith('user_')
      || supplied.length !== expected.length
      || !timingSafeEqual(supplied, expected)
    ) return null
    return { ownerId }
  }

  issue(res, { secure = false } = {}) {
    const identity = { ownerId: `user_${randomUUID()}` }
    const token = `${identity.ownerId}.${this.sign(identity.ownerId)}`
    const attributes = [
      `${this.cookieName}=${encodeURIComponent(token)}`,
      'Path=/',
      'HttpOnly',
      'SameSite=Strict',
      `Max-Age=${MAX_AGE_SECONDS}`,
    ]
    if (secure) attributes.push('Secure')
    res.setHeader('Set-Cookie', attributes.join('; '))
    return identity
  }

  resolveHttp(req, res) {
    if (this.mode === 'personal') return this.personalIdentity
    return this.resolveToken(req.headers.cookie)
      || this.issue(res, {
        secure: req.socket?.encrypted || req.headers['x-forwarded-proto'] === 'https',
      })
  }

  resolveUpgrade(req) {
    if (this.mode === 'personal') return this.personalIdentity
    return this.resolveToken(req.headers.cookie)
  }
}
