const STREAMING_INPUT_TRANSCRIPT_EVENTS = new Set([
  'conversation.item.input_audio_transcription.delta',
  'conversation.item.input_audio_transcription.text',
])

export function streamingInputTranscript(event) {
  if (!STREAMING_INPUT_TRANSCRIPT_EVENTS.has(event?.type)) return ''
  return `${String(event.text || '')}${String(event.stash || '')}`.trim()
}
