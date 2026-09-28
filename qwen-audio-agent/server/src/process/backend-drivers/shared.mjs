export function normalizeServiceEndpoint(value, {
  protocols = ['http:', 'https:', 'ws:', 'wss:'],
} = {}) {
  const target = new URL(value)
  if (!protocols.includes(target.protocol)) {
    throw new Error(`不支持的后台服务地址协议：${target.protocol}`)
  }
  if (target.username || target.password) {
    throw new Error('后台服务地址不能包含用户名或密码，请使用独立认证配置')
  }
  return target.origin
}

export function serviceEndpointPort(value) {
  const target = new URL(value)
  if (target.port) return Number(target.port)
  return ['https:', 'wss:'].includes(target.protocol) ? 443 : 80
}

export function localManagedBackend({
  id,
  env,
  ownership,
  permissionMode,
  baseUrlEnvironment,
  defaultBaseUrl,
  protocols,
}) {
  return {
    protocol: id,
    ownership,
    permissionMode,
    baseUrl: normalizeServiceEndpoint(
      env[baseUrlEnvironment] || defaultBaseUrl,
      { protocols },
    ),
  }
}

export function managedOnlyBackend({
  id,
  ownership,
  permissionMode,
  label,
}) {
  if (ownership !== 'owned') {
    throw new Error(`${label} 后台必须由 Gateway 启动`)
  }
  return {
    protocol: id,
    ownership,
    permissionMode,
    baseUrl: null,
  }
}

export function applyLocalAddress(env, backend, {
  baseUrlEnvironment,
  portEnvironment,
}) {
  const target = new URL(backend.baseUrl)
  env[baseUrlEnvironment] = backend.baseUrl
  env[portEnvironment] = target.port
    || (target.protocol === 'https:' ? '443' : '80')
}
