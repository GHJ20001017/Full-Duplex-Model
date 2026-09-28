export function realtimeConnectionStatus({
  provider,
  blockedError = '',
  sleeping = false,
  waking = false,
  ready = false,
  connecting = false,
} = {}) {
  let state = 'disconnected'
  if (blockedError) state = 'unavailable'
  else if (sleeping) state = 'sleeping'
  else if (waking) state = 'waking'
  else if (ready) state = 'connected'
  else if (connecting) state = 'connecting'
  return Object.freeze({
    provider,
    state,
    ...(blockedError ? { error: blockedError } : {}),
  })
}
