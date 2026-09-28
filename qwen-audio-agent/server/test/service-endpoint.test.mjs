import assert from 'node:assert/strict'
import test from 'node:test'
import {
  normalizeServiceEndpoint,
  serviceEndpointPort,
} from '../src/process/backend-drivers/shared.mjs'

test('normalizes supported service endpoints without retaining paths', () => {
  assert.equal(
    normalizeServiceEndpoint('wss://agent.example.com/gateway'),
    'wss://agent.example.com',
  )
  assert.equal(
    normalizeServiceEndpoint('https://agent.example.com:8443/path'),
    'https://agent.example.com:8443',
  )
})

test('resolves secure WebSocket and HTTP default ports consistently', () => {
  assert.equal(serviceEndpointPort('wss://agent.example.com'), 443)
  assert.equal(serviceEndpointPort('https://agent.example.com'), 443)
  assert.equal(serviceEndpointPort('ws://agent.example.com'), 80)
  assert.equal(serviceEndpointPort('http://agent.example.com:8080'), 8080)
})

test('rejects unsupported protocols and credentials embedded in URLs', () => {
  assert.throws(
    () => normalizeServiceEndpoint('ftp://agent.example.com'),
    /不支持的后台服务地址协议/,
  )
  assert.throws(
    () => normalizeServiceEndpoint('https://user:secret@agent.example.com'),
    /不能包含用户名或密码/,
  )
})
