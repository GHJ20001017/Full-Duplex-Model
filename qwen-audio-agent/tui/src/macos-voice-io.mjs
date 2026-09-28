import { spawn } from 'node:child_process'
import { stat, mkdir } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const SOURCE_PATH = fileURLToPath(
  new URL('../native/macos-voice-io.swift', import.meta.url),
)
export function macVoiceIOBinaryPath({
  homeDirectory = homedir(),
} = {}) {
  return resolve(
    homeDirectory,
    'Library/Caches/qwaudio/tui/macos-voice-io',
  )
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let stderr = ''
    child.stderr.on('data', chunk => {
      stderr += chunk.toString()
    })
    child.on('error', reject)
    child.on('close', code => {
      if (code === 0) resolvePromise()
      else reject(new Error(stderr.trim() || `${command} exited with ${code}`))
    })
  })
}

export async function ensureMacVoiceIO({
  sourcePath = SOURCE_PATH,
  binaryPath = process.env.QWEN_AUDIO_AGENT_TUI_AEC_BINARY
    || macVoiceIOBinaryPath(),
} = {}) {
  if (process.platform !== 'darwin') {
    throw new Error('CoreAudio Voice Processing 仅支持 macOS')
  }
  const source = await stat(sourcePath)
  const binary = await stat(binaryPath).catch(() => null)
  if (!binary || binary.mtimeMs < source.mtimeMs) {
    await mkdir(dirname(binaryPath), { recursive: true })
    await run('/usr/bin/swiftc', [
      sourcePath,
      '-O',
      '-framework',
      'AudioToolbox',
      '-o',
      binaryPath,
    ])
  }
  return binaryPath
}

export async function startMacVoiceIO({
  captureSampleRate,
  onAudio,
  onPlaybackStarted,
  onPlaybackEnded,
  onError,
  onExit,
  ensureImpl = ensureMacVoiceIO,
  spawnImpl = spawn,
} = {}) {
  const binaryPath = await ensureImpl()
  const child = spawnImpl(binaryPath, [
    '--capture-rate',
    String(captureSampleRate || 16000),
    '--playback-rate',
    '24000',
  ], {
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  let stdout = ''
  let ready = false
  let exited = false
  let readySettled = false
  let settleReady
  let rejectReady
  const readyPromise = new Promise((resolvePromise, reject) => {
    settleReady = value => {
      if (readySettled) return
      readySettled = true
      clearTimeout(readyTimer)
      resolvePromise(value)
    }
    rejectReady = error => {
      if (readySettled) return
      readySettled = true
      clearTimeout(readyTimer)
      reject(error)
    }
  })
  const readyTimer = setTimeout(() => {
    rejectReady(new Error('等待 CoreAudio Voice Processing 启动超时'))
    if (!child.killed) child.kill('SIGTERM')
  }, 15000)
  const send = payload => {
    if (exited || child.stdin.destroyed || !child.stdin.writable) return false
    try {
      child.stdin.write(`${JSON.stringify(payload)}\n`)
      return true
    } catch {
      return false
    }
  }
  const handleLine = line => {
    let event
    try {
      event = JSON.parse(line)
    } catch {
      return
    }
    if (event.type === 'ready') {
      ready = true
      settleReady(event)
    } else if (event.type === 'audio' && event.audio) {
      onAudio?.(Buffer.from(event.audio, 'base64'), event)
    } else if (event.type === 'playback.started' && event.responseId) {
      onPlaybackStarted?.(String(event.responseId))
    } else if (event.type === 'playback.ended' && event.responseId) {
      onPlaybackEnded?.(String(event.responseId))
    } else if (event.type === 'error') {
      const error = new Error(event.message || 'CoreAudio Voice Processing 错误')
      if (!ready) rejectReady(error)
      onError?.(error.message)
    }
  }
  child.stdout.on('data', chunk => {
    stdout += chunk.toString()
    let newline
    while ((newline = stdout.indexOf('\n')) >= 0) {
      const line = stdout.slice(0, newline).trim()
      stdout = stdout.slice(newline + 1)
      if (line) handleLine(line)
    }
  })
  child.stderr.on('data', chunk => {
    const message = chunk.toString().trim()
    if (message) onError?.(message)
  })
  child.stdin.on('error', error => {
    if (!exited) onError?.(`CoreAudio 控制通道错误：${error.message}`)
  })
  child.on('error', error => {
    if (!ready) rejectReady(error)
    onError?.(error.message)
  })
  child.on('close', (code, signal) => {
    exited = true
    if (!ready) rejectReady(new Error(`CoreAudio Voice Processing 启动失败 (${code ?? signal})`))
    onExit?.({ code, signal })
  })

  await readyPromise
  return {
    aec: true,
    write(buffer, sampleRate = 24000, responseId = '') {
      return send({
        type: 'play',
        audio: buffer.toString('base64'),
        sampleRate,
        responseId,
      })
    },
    done(responseId) {
      return send({ type: 'done', responseId })
    },
    clear() {
      send({ type: 'clear' })
    },
    setCaptureEnabled(enabled) {
      send({ type: 'capture', enabled: Boolean(enabled) })
    },
    close() {
      send({ type: 'close' })
      const timer = setTimeout(() => {
        if (!exited && !child.killed) child.kill('SIGTERM')
      }, 500)
      timer.unref?.()
    },
  }
}
