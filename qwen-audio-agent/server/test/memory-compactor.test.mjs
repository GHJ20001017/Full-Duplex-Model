import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { MemoryCompactor } from '../src/memory/learning/compactor.mjs'
import { MarkdownContextStore } from '../src/memory/providers/markdown/context-store.mjs'
import { MarkdownMemoryProvider } from '../src/memory/providers/markdown/provider.mjs'
import { FrontendMemoryRuntime } from '../src/memory/runtime.mjs'

const OWNER = 'owner'
const LONG = `# MEMORY\n\n- 用户${'目前主要居住在杭州。'.repeat(12)}\n- 用户喜欢跑步。`

function fixture(t, reply, { content = LONG, scope = 'memory' } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'qwaudio-compact-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const stores = Object.fromEntries(['user', 'memory'].map(name => [name, new MarkdownContextStore({
    filePath: join(dir, `${name}.md`), scope: name, maxChars: 120, hardMaxChars: 2000, onWarning() {},
  })]))
  stores[scope].persist(OWNER, content)
  const provider = new MarkdownMemoryProvider({ userStore: stores.user, memoryStore: stores.memory })
  const memory = new FrontendMemoryRuntime({ provider })
  const audit = []
  const calls = []
  const compactor = new MemoryCompactor({ memoryService: memory, thresholds: { user: 100, memory: 100 },
    audit: { record: event => audit.push(event) }, logger: { warn() {} },
    llmCall: async payload => { calls.push(payload); return reply(payload, memory) },
  })
  return { stores, provider, memory, compactor, audit, calls }
}

const complete = () => JSON.stringify({ groups: [
  { sources: [0], text: '- 用户住在杭州。' }, { sources: [1], text: '- 用户喜欢跑步。' },
] })

test('compacts the full stored document including facts beyond the injection budget', async t => {
  const f = fixture(t, payload => {
    assert.match(JSON.parse(payload.user).records.at(-1).text, /跑步/)
    return complete()
  })
  assert.doesNotMatch(f.memory.list(OWNER)[0].content, /跑步/)
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.match(f.stores.memory.readRaw(OWNER), /杭州[\s\S]*跑步/)
  assert.ok([...f.stores.memory.readRaw(OWNER)].length <= 100)
  assert.equal(f.audit.at(-1).reason, 'compacted')
  const backup = `${f.stores.memory.pathFor(OWNER)}.pre-compaction`
  assert.equal(readFileSync(backup, 'utf8'), LONG)
  assert.equal(statSync(backup).mode & 0o777, 0o600)
  assert.doesNotMatch(f.memory.list(OWNER)[0].content, /目前主要居住/)
})

test('does not call a model below the threshold or compact other owners', async t => {
  const f = fixture(t, complete, { content: '# MEMORY\n- 用户住杭州。' })
  await f.compactor.maybeRun({ ownerId: OWNER })
  await f.compactor.maybeRun({ ownerId: 'another-owner' })
  assert.equal(f.calls.length, 0)
})

for (const [name, reply] of [
  ['missing sources', () => '{"groups":[{"sources":[0],"text":"- 住杭州"}]}'],
  ['duplicate sources', () => '{"groups":[{"sources":[0,0,1],"text":"- 住杭州，喜欢跑步"}]}'],
  ['unknown sources', () => '{"groups":[{"sources":[0,1,7],"text":"- 住杭州，喜欢跑步"}]}'],
  ['sensitive output', () => '{"groups":[{"sources":[0,1],"text":"- 密码是测试"}]}'],
  ['malformed output', () => 'not JSON'],
  ['empty output', () => '{"groups":[]}'],
]) {
  test(`rejects ${name} without losing the original`, async t => {
    const f = fixture(t, reply)
    await f.compactor.maybeRun({ ownerId: OWNER })
    assert.equal(f.stores.memory.readRaw(OWNER), LONG)
  })
}

test('never rewrites explicit USER requirements', async t => {
  const content = `# USER\n## 用户明确要求\n- 以后请${'先听我说，不要急着建议。'.repeat(10)}`
  const f = fixture(t, () => '{"groups":[{"sources":[0],"text":"- 建议优先"}]}', { scope: 'user', content })
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.equal(f.stores.user.readRaw(OWNER), content)
  assert.equal(f.audit.at(-1).reason, 'compaction_not_safe')
})

test('can merge identical USER entries without rewriting their requirements', async t => {
  const line = '- 用户明确要求安慰时先倾听，不要主动提供建议。'
  const content = `# USER\n## 用户明确要求\n${Array(6).fill(line).join('\n')}`
  const f = fixture(t, () => JSON.stringify({ groups: [{ sources: [0, 1, 2, 3, 4, 5], text: line }] }),
    { scope: 'user', content })
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.equal(f.stores.user.readRaw(OWNER), `# USER\n\n## 用户明确要求\n${line}`)
})

test('does not mix explicit requirements with observed preferences', async t => {
  const content = `# USER\n## 用户明确要求\n- ${'先听我说。'.repeat(20)}\n## 观察推断\n- 温和玩笑`
  const f = fixture(t, () => '{"groups":[{"sources":[0,1],"text":"- 温和玩笑"}]}', { scope: 'user', content })
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.equal(f.stores.user.readRaw(OWNER), content)
})

test('an explicit concurrent edit wins over a delayed compaction', async t => {
  const deferred = Promise.withResolvers()
  const f = fixture(t, () => deferred.promise)
  const pending = f.compactor.maybeRun({ ownerId: OWNER })
  await Promise.resolve()
  await f.memory.apply(OWNER, [{ document: 'memory',
    edits: [{ old_text: LONG, new_text: '# MEMORY\n- 用户住苏州。' }],
  }], { source: 'gateway-memory-api' })
  deferred.resolve(complete())
  await pending
  assert.match(f.stores.memory.readRaw(OWNER), /苏州/)
  assert.equal(f.audit.at(-1).reason, 'compaction_stale')
})

test('invalidated learning cannot commit even if the revision did not change', async t => {
  const deferred = Promise.withResolvers()
  let current = true
  const f = fixture(t, () => deferred.promise)
  const pending = f.compactor.maybeRun({ ownerId: OWNER, isCurrent: () => current })
  await Promise.resolve()
  current = false
  deferred.resolve(complete())
  await pending
  assert.equal(f.stores.memory.readRaw(OWNER), LONG)
})

test('custom providers without full learning snapshots are left untouched', () => {
  assert.equal(new MemoryCompactor({ memoryService: {}, llmCall: async () => '' }).enabled(), false)
})

test('a failed USER compaction does not prevent MEMORY maintenance', async t => {
  const f = fixture(t, payload => JSON.parse(payload.user).document === 'user' ? 'invalid JSON' : complete())
  f.stores.user.persist(OWNER, `# USER\n- ${'始终先倾听，不要急着建议。'.repeat(12)}`)
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.match(f.stores.user.readRaw(OWNER), /始终先倾听/)
  assert.match(f.stores.memory.readRaw(OWNER), /杭州[\s\S]*跑步/)
  assert.equal(f.audit.at(-1).reason, 'compacted')
  assert.equal(f.calls.length, 2)
})

test('same-owner maintenance jobs serialize and recheck capacity before spending another call', async t => {
  const deferred = Promise.withResolvers()
  const f = fixture(t, () => deferred.promise)
  const first = f.compactor.maybeRun({ ownerId: OWNER })
  const second = f.compactor.maybeRun({ ownerId: OWNER })
  await Promise.resolve()
  assert.equal(f.calls.length, 1)
  deferred.resolve(complete())
  await f.compactor.drain()
  await Promise.all([first, second])
  assert.equal(f.calls.length, 1, 'the second job sees the already compacted document')
})

for (const source of ['gateway-memory-api', 'realtime-tool', 'automatic-extraction']) {
  test(`${source}: an explicit correction removes the old compaction backup`, async t => {
    const f = fixture(t, complete)
    await f.compactor.maybeRun({ ownerId: OWNER })
    const backup = `${f.stores.memory.pathFor(OWNER)}.pre-compaction`
    assert.equal(existsSync(backup), true)
    await f.memory.apply(OWNER, [{ document: 'memory',
      edits: [{ old_text: '- 用户住在杭州。', new_text: '- 用户住在苏州。' }],
    }], { source })
    assert.equal(existsSync(backup), false)
    assert.match(f.stores.memory.readRaw(OWNER), /苏州/)
  })
}

test('a failed backup leaves the original document untouched', async t => {
  const f = fixture(t, complete)
  f.stores.memory.saveCompactionBackup = () => { throw new Error('disk unavailable') }
  await f.compactor.maybeRun({ ownerId: OWNER })
  assert.equal(f.stores.memory.readRaw(OWNER), LONG)
  assert.equal(f.audit.at(-1).reason, 'compaction_failed')
})
