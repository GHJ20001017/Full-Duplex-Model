export class ReconnectBackoff {
  constructor({
    baseMs = 500,
    maxMs = 10_000,
    jitterRatio = 0.2,
    random = Math.random,
  } = {}) {
    this.baseMs = baseMs
    this.maxMs = maxMs
    this.jitterRatio = jitterRatio
    this.random = random
    this.attempt = 0
  }

  next() {
    const exponential = Math.min(
      this.maxMs,
      this.baseMs * (2 ** Math.min(this.attempt, 8)),
    )
    this.attempt += 1
    const jitter = exponential * this.jitterRatio * (
      (this.random() * 2) - 1
    )
    return Math.max(0, Math.round(exponential + jitter))
  }

  reset() {
    this.attempt = 0
  }
}
