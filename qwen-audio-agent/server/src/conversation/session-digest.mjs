// Bounded episodic memory, separate from both durable profile documents and
// the conversation-recovery journal. Existing topic/gist-only entries remain
// valid. Recall ranks lexical matches; the archive is never injected wholesale.

import { containsSensitiveContent } from '../core/sensitive-content.mjs'
import { JsonSnapshotStore } from '../core/json-snapshot-store.mjs'

const FILE_VERSION = 1
const MAX_SESSIONS_PER_OWNER = 60
const MAX_OWNERS = 200
const MAX_TOPICS = 5
const MAX_TOPIC_CHARS = 16
const MAX_GIST_CHARS = 50
const MAX_SUMMARY_CHARS = 400
const MAX_WORK_ITEMS = 5
const MAX_WORK_OBJECTIVE_CHARS = 40
const RETENTION_MS = 90 * 24 * 60 * 60_000
const DEFAULT_LIMIT = 5
const MAX_LIMIT = 10

function text(value, maxChars) {
  return [...String(value || '').replace(/\s+/g, ' ').trim()]
    .slice(0, maxChars)
    .join('')
}

// 匹配前把标点和空白去掉：ASR 转写的标点很随意，用户说「LOCOMO」而记录里
// 是「LOCOMO、压缩评测」，带标点做子串匹配会漏。
function bare(value) {
  return String(value || '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '')
}

const STOP_TERMS = new Set(['的话', '话题', '那次', '上次', '之前', '前几', '几天',
  '我们', '那个', '这个', '记得', '还记', '时候', '怎么', '事情', '关于', '最近',
  '一下', '一场', '聊过', '聊的', '一次', '后来', '用户', '助手'])

function terms(value) {
  const tokens = new Set()
  for (const match of String(value || '').toLowerCase().matchAll(/[\p{Script=Han}]+|[\p{L}\p{N}_-]+/gu)) {
    const word = match[0]
    if (/^\p{Script=Han}+$/u.test(word) && word.length > 1) {
      for (let index = 0; index < word.length - 1; index++) tokens.add(word.slice(index, index + 2))
    } else tokens.add(word)
  }
  return new Set([...tokens].filter(token => !STOP_TERMS.has(token)))
}

function relevance(query, needle, value) {
  const normalized = bare(value)
  if (!normalized) return 0
  if (normalized.includes(needle)) return 2
  const candidates = terms(value)
  const shared = [...query].filter(token => candidates.has(token)).length
  // Partial overlap is useful for natural questions, but a single common
  // character/bigram must not make unrelated experiences look like memories.
  return shared >= Math.min(2, query.size) && shared / query.size >= 0.25
    ? shared / query.size : 0
}

// 取用户本地时区下的日历日：用服务器时区算「今天/昨天」跨时区就错一天。
function calendarParts(timestamp, timeZone) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(timestamp))
  const lookup = Object.fromEntries(parts.map(part => [part.type, part.value]))
  return {
    date: `${lookup.year}-${lookup.month}-${lookup.day}`,
    day: Date.UTC(Number(lookup.year), Number(lookup.month) - 1, Number(lookup.day)),
  }
}

// 相对表述给模型念，绝对日期留着回答「具体哪天」。相对表述是主的：
// 绝对日期会逼模型自己算相对关系，语音场景里这一步很容易算错。
export function describeWhen(at, { now = Date.now(), timeZone = 'UTC' } = {}) {
  let current
  let past
  try {
    current = calendarParts(now, timeZone)
    past = calendarParts(at, timeZone)
  } catch {
    current = calendarParts(now, 'UTC')
    past = calendarParts(at, 'UTC')
  }
  const days = Math.round((current.day - past.day) / 86_400_000)
  let when
  if (days <= 0) when = '今天'
  else if (days === 1) when = '昨天'
  else if (days < 7) when = `${days} 天前`
  else if (days < 14) when = '上周'
  else if (days < 30) when = `${Math.floor(days / 7)} 周前`
  else when = `${Math.floor(days / 30)} 个月前`
  return { when, date: past.date, days_ago: Math.max(days, 0) }
}

// 本场派出去的活。刻意只存 id 与 objective，【绝不存 status】——
// 摘要是冻结的事实，任务状态是活的：今天记「进行中」，过几天任务早完成了，
// 存下来的那个值就是错的，而且不报错。状态一律在检索时从任务台账实时读；
// 台账过期（终态 30 天）后就只答「派过这件活」，自然降级。
export function normaliseWork(work) {
  if (!Array.isArray(work)) return []
  const items = []
  for (const item of work) {
    const objective = text(item?.objective, MAX_WORK_OBJECTIVE_CHARS)
    if (!objective) continue
    if (containsSensitiveContent(objective)) continue
    const id = text(item?.id, 120)
    if (id && items.some(existing => existing.id === id)) continue
    items.push(id ? { id, objective } : { objective })
    if (items.length >= MAX_WORK_ITEMS) break
  }
  return items
}

export function normaliseTopics(topics) {
  if (!Array.isArray(topics)) return []
  const seen = []
  for (const item of topics) {
    const topic = text(item, MAX_TOPIC_CHARS)
    if (!topic) continue
    if (containsSensitiveContent(topic)) continue
    if (seen.some(existing => bare(existing) === bare(topic))) continue
    seen.push(topic)
    if (seen.length >= MAX_TOPICS) break
  }
  return seen
}

export class SessionDigestPool {
  constructor({
    filePath = null,
    store = null,
    now = () => Date.now(),
    onWarning = warning => console.warn(warning.message),
    retentionMs = RETENTION_MS,
    maxPerOwner = MAX_SESSIONS_PER_OWNER,
  } = {}) {
    this.store = store || (filePath
      ? new JsonSnapshotStore({
          filePath,
          fileVersion: FILE_VERSION,
          label: '会话摘要',
          requiredKeys: ['owners'],
          now,
          onWarning,
        })
      : null)
    this.now = now
    this.retentionMs = retentionMs
    this.maxPerOwner = maxPerOwner
    this.owners = new Map()
    this.loaded = false
  }

  load() {
    if (this.loaded) {
      // 别的 Gateway 进程可能也在写同一份文件。
      if (!this.store?.hasChanged()) return
    }
    this.loaded = true
    this.owners = new Map()
    const snapshot = this.store?.load()
    if (!snapshot) return
    for (const [ownerId, digests] of Object.entries(snapshot.owners || {})) {
      if (!Array.isArray(digests)) continue
      const restored = digests
        .map(digest => this.sanitise(digest))
        .filter(Boolean)
      if (restored.length) this.owners.set(String(ownerId), restored)
    }
  }

  // 逐条校验而不是整份信任：手改过或跨版本残留的条目应当被丢弃，而不是带着
  // 半截字段流进工具返回值。
  // work 缺失时给空数组而不是判为无效 —— 它是后加的字段，早先写下的摘要没有它，
  // 这种向后兼容的新增不该升 FILE_VERSION，否则会把已经攒下的摘要全部隔离掉。
  sanitise(digest) {
    const session = text(digest?.session, 120)
    const at = Number(digest?.at)
    if (!session || !Number.isFinite(at) || at <= 0) return null
    const gist = text(digest?.gist, MAX_GIST_CHARS)
    const summary = text(digest?.summary, MAX_SUMMARY_CHARS)
    const topics = normaliseTopics(digest?.topics)
    const work = normaliseWork(digest?.work)
    if (!gist && !summary && !topics.length && !work.length) return null
    if (containsSensitiveContent(`${gist}\n${summary}`)) return null
    const turns = Number(digest?.turns)
    return {
      session,
      at,
      turns: Number.isFinite(turns) && turns > 0 ? Math.floor(turns) : 0,
      topics,
      gist,
      ...(summary ? { summary } : {}),
      work,
    }
  }

  save() {
    if (!this.store?.enabled()) return false
    const owners = {}
    for (const [ownerId, digests] of this.owners) {
      if (digests.length) owners[ownerId] = digests
    }
    return this.store.save({ owners })
  }

  // 记一场会话。同一 sessionId 重复记录时覆盖而不是追加 —— 一场会话只该有
  // 一条摘要，重连产生的第二次调用不能把列表灌满。
  record({ ownerId, sessionId, topics, gist, summary, turns, work } = {}) {
    this.load()
    const safeOwnerId = String(ownerId || '')
    const digest = this.sanitise({
      session: sessionId,
      at: this.now(),
      topics,
      gist,
      summary,
      turns,
      work,
    })
    if (!safeOwnerId || !digest) return null
    if (containsSensitiveContent(digest.gist)) return null

    const digests = this.owners.get(safeOwnerId) || []
    const existing = digests.findIndex(item => item.session === digest.session)
    if (existing >= 0) digests[existing] = digest
    else digests.push(digest)

    digests.sort((left, right) => right.at - left.at)
    this.owners.set(safeOwnerId, digests.slice(0, this.maxPerOwner))
    this.prune()
    this.save()
    return digest
  }

  prune() {
    const cutoff = this.now() - this.retentionMs
    for (const [ownerId, digests] of this.owners) {
      const fresh = digests.filter(digest => digest.at >= cutoff)
      if (fresh.length) this.owners.set(ownerId, fresh)
      else this.owners.delete(ownerId)
    }
    if (this.owners.size <= MAX_OWNERS) return
    // 超过 owner 上限时丢掉最久没有新会话的那些。
    const ranked = [...this.owners.entries()]
      .sort((left, right) => (right[1][0]?.at || 0) - (left[1][0]?.at || 0))
    this.owners = new Map(ranked.slice(0, MAX_OWNERS))
  }

  has({ ownerId, sessionId } = {}) {
    this.load()
    const digests = this.owners.get(String(ownerId || '')) || []
    return digests.some(digest => digest.session === String(sessionId || ''))
  }

  get({ ownerId, sessionId } = {}) {
    this.load()
    this.prune()
    return (this.owners.get(String(ownerId || '')) || [])
      .find(digest => digest.session === String(sessionId || '')) || null
  }

  // Empty queries return recent sessions. Otherwise rank exact/partial lexical
  // overlap across topics, dispatched work and the bounded experience summary.
  search({ ownerId, keyword = '', limit = DEFAULT_LIMIT } = {}) {
    this.load()
    this.prune()
    const digests = this.owners.get(String(ownerId || '')) || []
    const size = Math.min(Math.max(Number(limit) || DEFAULT_LIMIT, 1), MAX_LIMIT)
    const needle = bare(keyword)
    if (!needle) return digests.slice(0, size)
    const queryTerms = terms(keyword)
    const scored = []
    for (const digest of digests) {
      const topicScore = Math.max(0, ...digest.topics.map(topic => relevance(queryTerms, needle, topic)))
      const workScore = Math.max(0, ...digest.work.map(item => relevance(queryTerms, needle, item.objective)))
      const bodyScore = relevance(queryTerms, needle, `${digest.gist} ${digest.summary || ''}`)
      const score = topicScore * 3 + workScore * 2 + bodyScore
      if (score) scored.push({ digest, score })
    }
    return scored
      .sort((left, right) => (right.score - left.score) || (right.digest.at - left.digest.at))
      .slice(0, size)
      .map(item => item.digest)
  }

  count(ownerId) {
    this.load()
    return (this.owners.get(String(ownerId || '')) || []).length
  }

  health() {
    return {
      ...(this.store?.health() || { ok: true, persistenceEnabled: false, warning: null }),
      owners: this.owners.size,
    }
  }
}

export const SESSION_DIGEST_LIMITS = Object.freeze({
  MAX_TOPICS,
  MAX_TOPIC_CHARS,
  MAX_GIST_CHARS,
  MAX_SUMMARY_CHARS,
  MAX_WORK_ITEMS,
  MAX_WORK_OBJECTIVE_CHARS,
  MAX_SESSIONS_PER_OWNER,
  RETENTION_MS,
  DEFAULT_LIMIT,
  MAX_LIMIT,
})
