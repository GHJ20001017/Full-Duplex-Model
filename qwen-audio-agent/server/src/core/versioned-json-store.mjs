import {
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { dirname } from 'node:path'

export class VersionedJsonStore {
  constructor({
    filePath = null,
    version = 1,
    label = '状态',
    now = () => Date.now(),
    onWarning = () => {},
  } = {}) {
    this.filePath = filePath
    this.version = version
    this.label = label
    this.now = now
    this.onWarning = onWarning
    this.warning = null
    this.persistenceDisabled = false
  }

  warn(message, quarantinePath = null) {
    this.warning = { message, quarantinePath, at: this.now() }
    try {
      this.onWarning?.(this.warning)
    } catch {
      // Diagnostics must never prevent startup.
    }
  }

  quarantine(reason) {
    const quarantinePath = `${this.filePath}.corrupt-${this.now()}`
    try {
      renameSync(this.filePath, quarantinePath)
      this.warn(
        `${reason}；原文件已隔离为 ${quarantinePath}。`,
        quarantinePath,
      )
    } catch (error) {
      this.persistenceDisabled = true
      this.warn(
        `${reason}；隔离失败（${error.message}），已禁用持久化以保护原文件。`,
      )
    }
  }

  load({ fallback, validate }) {
    if (!this.filePath || this.persistenceDisabled) return fallback()
    let parsed
    try {
      parsed = JSON.parse(readFileSync(this.filePath, 'utf8'))
    } catch (error) {
      if (error.code === 'ENOENT') return fallback()
      if (error instanceof SyntaxError) {
        this.quarantine(`${this.label}文件不是有效的 JSON：${error.message}`)
      } else {
        this.persistenceDisabled = true
        this.warn(`无法读取${this.label}文件：${error.message}`)
      }
      return fallback()
    }
    if (parsed?.version !== this.version || !validate(parsed)) {
      this.quarantine(`${this.label}文件格式或版本无效`)
      return fallback()
    }
    return parsed
  }

  save(value) {
    if (!this.filePath || this.persistenceDisabled) return false
    try {
      mkdirSync(dirname(this.filePath), { recursive: true })
      const temporary = `${this.filePath}.${process.pid}.tmp`
      writeFileSync(
        temporary,
        `${JSON.stringify({ version: this.version, ...value }, null, 2)}\n`,
        { encoding: 'utf8', mode: 0o600 },
      )
      renameSync(temporary, this.filePath)
      return true
    } catch (error) {
      this.persistenceDisabled = true
      this.warn(`无法保存${this.label}文件：${error.message}`)
      return false
    }
  }

  health() {
    return {
      ok: !this.warning,
      persistenceEnabled: Boolean(this.filePath) && !this.persistenceDisabled,
      warning: this.warning,
    }
  }
}
