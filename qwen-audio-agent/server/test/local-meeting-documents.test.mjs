import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  LocalMeetingDocumentsSource,
  TOOL_NAME,
} from '../src/frontend/tools/local-meeting-documents.mjs'

test('exposes a read-only DOCX meeting document tool', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'qwaudio-meetings-'))
  await writeFile(join(directory, '研究院双周工作汇报_20260921.txt'), 'not a docx')
  await writeFile(join(directory, '研究院双周工作汇报_20260921.doc'), 'not a docx')

  const source = new LocalMeetingDocumentsSource({ directory })
  assert.deepEqual(source.describe(), {
    key: 'local-meeting-documents',
    label: 'Local meeting documents',
  })
  assert.equal(source.tools()[0].name, TOOL_NAME)
  assert.deepEqual(source.health().supported_extensions, ['.docx'])
  assert.deepEqual(
    await source.execute(TOOL_NAME, { query: '研究院' }),
    { status: 'not_found', documents: [], message: '没有找到匹配的 Word 会议文档。' },
  )
})

test('only searches the configured directory and validates date ranges', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'qwaudio-meetings-'))
  await mkdir(join(directory, 'nested'))
  const source = new LocalMeetingDocumentsSource({ directory })

  const invalid = await source.execute(TOOL_NAME, {
    query: '周会',
    date_from: '2026-10-02',
    date_to: '2026-09-02',
  })
  assert.equal(invalid.error_code, 'invalid_date_range')

  const outside = await source.execute(TOOL_NAME, {
    query: '../etc/passwd',
  })
  assert.equal(outside.status, 'not_found')
})
