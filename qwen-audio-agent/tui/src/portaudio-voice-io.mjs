import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const HELPER_PATH = fileURLToPath(
  new URL('../native/portaudio-voice-io.py', import.meta.url),
)
const PLAYBACK_SAMPLE_RATE = 24000

export async function startPortAudioVoiceIO({
  captureSampleRate,
  duplexMode = 'half',
  onAudio,
  onPlaybackStarted,
  onPlaybackEnded,
  onError,
  onExit,
  spawnImpl = spawn,
  python = process.env.PYTHON
    || (process.platform === 'win32' ? 'python' : 'python3'),
} = {}) {
  const child = spawnImpl(
    python,
    [
      HELPER_PATH,
      '--capture-rate',
      String(captureSampleRate || 16000),
      '--playback-rate',
      String(PLAYBACK_SAMPLE_RATE),
      '--mode',
      duplexMode,
    ],
    { stdio: ['pipe', 'pipe', 'pipe'] },
  )
  let stdout = ''
  let stderr = ''
  let ready = false
  let closed = false
  let exited = false
  let settleReady
  let rejectReady
  const readyPromise = new Promise((resolvePromise, rejectPromise) => {
    settleReady = resolvePromise
    rejectReady = rejectPromise
  })
  const timeout = setTimeout(() => {
    if (ready) return
    child.kill()
    rejectReady(new Error('PortAudio 音频助手启动超时'))
  }, 5000)

  const failStartup = error => {
    if (ready) {
      onError?.(error.message)
      return
    }
    clearTimeout(timeout)
    rejectReady(error)
  }
  const handleLine = line => {
    if (!line.trim()) return
    let event
    try {
      event = JSON.parse(line)
    } catch {
      onError?.(`PortAudio 返回了无效消息：${line}`)
      return
    }
    if (event.type === 'ready') {
      ready = true
      clearTimeout(timeout)
      settleReady()
    } else if (event.type === 'audio' && event.audio) {
      onAudio?.(Buffer.from(event.audio, 'base64'))
    } else if (event.type === 'playback.started' && event.responseId) {
      onPlaybackStarted?.(String(event.responseId))
    } else if (event.type === 'playback.ended' && event.responseId) {
      onPlaybackEnded?.(String(event.responseId))
    } else if (event.type === 'error' && event.message) {
      onError?.(String(event.message))
    }
  }

  child.stdout.on('data', chunk => {
    stdout += chunk.toString()
    let newline
    while ((newline = stdout.indexOf('\n')) >= 0) {
      const line = stdout.slice(0, newline)
      stdout = stdout.slice(newline + 1)
      handleLine(line)
    }
  })
  child.stderr.on('data', chunk => {
    stderr += chunk.toString()
  })
  child.stdin.on('error', error => {
    if (ready && !closed) onError?.(`PortAudio 通信失败：${error.message}`)
  })
  child.once('error', error => {
    failStartup(new Error(`无法启动 PortAudio 音频助手：${error.message}`))
  })
  child.once('exit', (code, signal) => {
    exited = true
    clearTimeout(timeout)
    if (!ready) {
      failStartup(new Error(
        stderr.trim()
        || `PortAudio 音频助手启动失败：${code ?? signal ?? 'unknown'}`,
      ))
      return
    }
    onExit?.({ code, signal })
  })

  await readyPromise

  const send = event => {
    if (closed || child.stdin.destroyed || !child.stdin.writable) return false
    try {
      child.stdin.write(`${JSON.stringify(event)}\n`)
      return true
    } catch (error) {
      onError?.(`PortAudio 通信失败：${error.message}`)
      return false
    }
  }

  return {
    setCaptureEnabled(enabled) {
      send({ type: 'capture', enabled: Boolean(enabled) })
    },
    write(buffer, sampleRate = PLAYBACK_SAMPLE_RATE, responseId = '') {
      if (Number(sampleRate) !== PLAYBACK_SAMPLE_RATE) {
        onError?.(`PortAudio 只接受 ${PLAYBACK_SAMPLE_RATE} Hz 播放音频`)
        return false
      }
      return send({
        type: 'play',
        audio: Buffer.from(buffer).toString('base64'),
        responseId,
      })
    },
    done(responseId) {
      return send({ type: 'done', responseId })
    },
    clear() {
      send({ type: 'clear' })
    },
    close() {
      if (closed) return
      send({ type: 'close' })
      closed = true
      const timer = setTimeout(() => {
        if (!exited && !child.killed) child.kill('SIGTERM')
      }, 1000)
      timer.unref()
    },
  }
}
