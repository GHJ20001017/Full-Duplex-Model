import assert from 'node:assert/strict'
import test from 'node:test'
import { evaluateResponseGuards } from '../src/voice/response-guards/index.mjs'
import { containsTaskStartClaim, containsTaskCompletionClaim } from '../src/voice/response-guards/unsupported-task-start.mjs'

for (const transcript of [
  '已经执行了。', '已完成任务。', '已经重启服务。', '我重启了服务。',
  'I restarted the service.', 'The task is completed.', 'I have executed the task.',
]) {
  test(`completion requires stronger evidence: ${transcript}`, () => {
    assert.equal(containsTaskCompletionClaim(transcript), true)
    const observation = { origin: 'model', transcript, taskAcceptanceVerified: true, taskExecutionVerified: true }
    assert.equal(evaluateResponseGuards(observation)?.guardId, 'unsupported-task-start')
    assert.equal(evaluateResponseGuards({ ...observation, taskCompletionVerified: true }), null)
  })
}

for (const transcript of [
  '后台任务已经开始处理。', 'I have started processing the task.',
  '我没有重启服务。', '例如：“已经重启服务。”',
  'If I restarted the service, the connection would drop.',
]) {
  test(`not a completion receipt: ${transcript}`, () => {
    assert.equal(containsTaskCompletionClaim(transcript), false)
  })
}

for (const transcript of [
  '我已经启动后台任务。', '后台任务已经提交。', '我刚刚安排了后台代理执行。',
  'I have started the background task.', "I've submitted the task.",
  'The background job is running.',
]) {
  test(`detects unsupported task receipt: ${transcript}`, () => {
    assert.equal(containsTaskStartClaim(transcript), true)
    assert.equal(evaluateResponseGuards({ origin: 'model', transcript })?.guardId, 'unsupported-task-start')
    assert.equal(evaluateResponseGuards({ origin: 'model', transcript, hasFunctionCall: true })?.guardId, 'unsupported-task-start')
    assert.equal(evaluateResponseGuards({ origin: 'model', transcript, taskAcceptanceVerified: true, taskExecutionVerified: true }), null)
    assert.equal(evaluateResponseGuards({ origin: 'model', transcript, failed: true }), null)
  })
}

for (const transcript of [
  '后台任务尚未提交。', '如果后台任务已经启动，就可以查看日志。',
  '例如：“我已经启动后台任务。”', 'The example says "I have started the background task."',
  'Background tasks run independently of the user interface.',
  'I have not started the background task.', '我可以帮你安排后台任务。',
]) {
  test(`preserves discussions and non-success statements: ${transcript}`, () => {
    assert.equal(containsTaskStartClaim(transcript), false)
  })
}

test('queued evidence supports acceptance but not execution', () => {
  const evidence = { origin: 'model', taskAcceptanceVerified: true, taskExecutionVerified: false }
  assert.equal(evaluateResponseGuards({ ...evidence, transcript: '后台任务已经提交。' }), null)
  assert.equal(evaluateResponseGuards({ ...evidence, transcript: 'I have started the background task.' })?.guardId, 'unsupported-task-start')
})
