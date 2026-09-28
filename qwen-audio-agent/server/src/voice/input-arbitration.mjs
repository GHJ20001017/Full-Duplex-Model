// Input arbitration between an external host and this Gateway.
//
// A host such as a system input method or a platform app owns the microphone
// whenever the user dictates into it. It announces that through the control
// plane, and the Gateway commands its clients to stop capturing. Three
// properties matter:
//
// - Reference counted per holder, so overlapping suspensions from several host
//   subsystems compose instead of racing.
// - Idempotent per holder: a repeated suspend refreshes the deadline rather
//   than incrementing the count, so a host that re-announces on every keypress
//   cannot leak a holder it will never release.
// - Every hold expires. A host that crashes or forgets to resume must not be
//   able to silence the plugin permanently, so a hold releases itself when its
//   ttl runs out.

export const DEFAULT_INPUT_SUSPEND_TTL_MS = 15_000
export const MAX_INPUT_SUSPEND_TTL_MS = 300_000

export class InputArbitration {
  constructor({
    defaultTtlMs = DEFAULT_INPUT_SUSPEND_TTL_MS,
    maxTtlMs = MAX_INPUT_SUSPEND_TTL_MS,
    now = () => Date.now(),
    setTimer = (callback, delay) => {
      const timer = setTimeout(callback, delay)
      timer.unref?.()
      return timer
    },
    clearTimer = timer => clearTimeout(timer),
    logger,
  } = {}) {
    this.defaultTtlMs = defaultTtlMs
    this.maxTtlMs = maxTtlMs
    this.now = now
    this.setTimer = setTimer
    this.clearTimer = clearTimer
    this.logger = logger
    this.holders = new Map()
    this.listeners = new Set()
  }

  get suspended() {
    return this.holders.size > 0
  }

  subscribe(listener) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  notify(change) {
    const status = this.status()
    for (const listener of this.listeners) {
      try {
        listener(status, change)
      } catch (error) {
        this.logger?.error('input.arbitration_listener_failed', { error })
      }
    }
  }

  ttlFor(ttlMs) {
    const requested = Number(ttlMs)
    if (!Number.isFinite(requested) || requested <= 0) return this.defaultTtlMs
    return Math.min(this.maxTtlMs, requested)
  }

  suspend({ owner, reason = '', ttlMs } = {}) {
    const holder = String(owner || '').trim().slice(0, 80)
    if (!holder) {
      const error = new Error('input.suspend 需要 owner')
      error.code = 'QWAUDIO_INPUT_OWNER_REQUIRED'
      throw error
    }
    const wasSuspended = this.suspended
    const ttl = this.ttlFor(ttlMs)
    const existing = this.holders.get(holder)
    if (existing) this.clearTimer(existing.timer)
    const expiresAt = this.now() + ttl
    this.holders.set(holder, {
      owner: holder,
      reason: String(reason || '').slice(0, 200),
      ttlMs: ttl,
      since: existing?.since ?? this.now(),
      expiresAt,
      timer: this.setTimer(() => this.expire(holder), ttl),
    })
    this.logger?.info('input.suspended', {
      owner: holder,
      reason,
      ttlMs: ttl,
      renewed: Boolean(existing),
    })
    if (!wasSuspended) this.notify({ state: 'suspended', owner: holder, reason })
    return this.status()
  }

  resume({ owner } = {}) {
    const holder = String(owner || '').trim().slice(0, 80)
    const existing = this.holders.get(holder)
    if (!existing) return this.status()
    this.clearTimer(existing.timer)
    this.holders.delete(holder)
    this.logger?.info('input.resumed', { owner: holder })
    if (!this.suspended) this.notify({ state: 'resumed', owner: holder })
    return this.status()
  }

  expire(owner) {
    const existing = this.holders.get(owner)
    if (!existing) return
    this.holders.delete(owner)
    this.logger?.warn('input.suspend_expired', {
      owner,
      ttlMs: existing.ttlMs,
    })
    if (!this.suspended) {
      this.notify({ state: 'resumed', owner, expired: true })
    }
  }

  status() {
    const holders = [...this.holders.values()].map(({
      owner,
      reason,
      ttlMs,
      since,
      expiresAt,
    }) => ({ owner, reason, ttlMs, since, expiresAt }))
    return {
      suspended: holders.length > 0,
      holders,
      // Convenience for a host UI that only ever holds one suspension.
      owner: holders[0]?.owner || null,
      reason: holders[0]?.reason || '',
      expiresAt: holders.length
        ? Math.max(...holders.map(holder => holder.expiresAt))
        : null,
    }
  }

  // Drops every hold. A Gateway that stops serving cannot honour a resume, so
  // held state from a previous run must not survive into the next one.
  // Subscribers are kept: they belong to the WebSocket server, which is reused.
  releaseAll() {
    const suspended = this.suspended
    for (const holder of this.holders.values()) this.clearTimer(holder.timer)
    this.holders.clear()
    if (suspended) this.notify({ state: 'resumed', owner: null })
  }

  close() {
    for (const holder of this.holders.values()) this.clearTimer(holder.timer)
    this.holders.clear()
    this.listeners.clear()
  }
}
