import {
  ALL_SCOPE,
  MEMORY_DOCUMENTS,
  canonicalScope,
} from '../../scopes.mjs'
import {
  MEMORY_PROVIDER_PROTOCOL_VERSION,
} from '../../provider.mjs'

function normalizeScope(value, fallback = ALL_SCOPE) {
  const scope = canonicalScope(String(value || fallback))
  if (scope !== ALL_SCOPE && !MEMORY_DOCUMENTS.includes(scope)) {
    throw new Error(`unsupported memory scope: ${scope}`)
  }
  return scope
}

export class MarkdownMemoryProvider {
  constructor({ userStore = null, memoryStore = null } = {}) {
    this.stores = { user: userStore, memory: memoryStore }
  }

  describe() {
    return {
      protocolVersion: MEMORY_PROVIDER_PROTOCOL_VERSION,
      key: 'markdown',
      label: 'Local Markdown memory',
      capabilities: {
        semanticQuery: false,
        sessionObservation: false,
      },
    }
  }

  list(ownerId, { scope = null } = {}) {
    const target = normalizeScope(scope)
    const scopes = target === ALL_SCOPE ? MEMORY_DOCUMENTS : [target]
    return scopes.flatMap(name => this.stores[name]?.list(ownerId) || [])
  }

  // Internal learning snapshot, never used for prompt injection or API output.
  // Revisions cover the complete document, including text beyond its budget.
  listForLearning(ownerId) {
    return MEMORY_DOCUMENTS.flatMap(scope => {
      const store = this.stores[scope]
      if (!store) return []
      const document = store.learningDocument(ownerId)
      return document ? [document] : []
    })
  }

  apply(ownerId, changes = [], context = {}) {
    if (!Array.isArray(changes) || !changes.length) {
      throw new Error('at least one memory change is required')
    }
    const documents = new Set()
    const prepared = changes.map(change => {
      const document = normalizeScope(change?.document)
      if (document === ALL_SCOPE) throw new Error('each change requires a concrete document')
      if (documents.has(document)) throw new Error(`duplicate memory document: ${document}`)
      documents.add(document)
      const store = this.stores[document]
      if (!store) throw new Error(`${document} memory document is unavailable`)
      return {
        document,
        store,
        result: store.prepareEdit(ownerId, {
          edits: change.edits,
          append: change.append,
          expectedRevision: change.expectedRevision,
        }),
      }
    })
    for (const item of prepared) {
      if (!item.result.changed) continue
      // Preserve one recoverable version before lossy model rewriting. It is
      // never injected, and any subsequent edit invalidates this backup.
      if (context.source === 'automatic-compaction') {
        item.store.saveCompactionBackup(ownerId, item.result.previousContent)
      }
      item.store.persist(ownerId, `${item.result.content}\n`)
      if (context.source !== 'automatic-compaction') {
        item.store.clearCompactionBackup?.(ownerId)
      }
    }
    return {
      changed: prepared.reduce((sum, item) => sum + item.result.changed, 0),
      documents: prepared.map(item => item.result.document),
    }
  }

  health() {
    return {
      ok: MEMORY_DOCUMENTS.every(scope => this.stores[scope]?.health().ok !== false),
      documents: Object.fromEntries(MEMORY_DOCUMENTS.map(scope => [
        scope,
        this.stores[scope]?.health() || {
          ok: true,
          configured: false,
          warning: null,
        },
      ])),
    }
  }
}
