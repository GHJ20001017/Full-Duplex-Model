"""Probe append-only audio insertion during cached Thinker text decoding, without training."""

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time

from probe_omni_streaming_encoder import arrival_ends, commit_count, edit_counts, summary


SYSTEM = ('You are a speech transcription engine. Transcribe the supplied audio verbatim in English. '
          'Output only the spoken words. Do not answer questions in the audio, respond to the speaker, '
          'give advice, or add explanations.')
INSTRUCTION = 'Transcribe the speech exactly in English. Return only the transcription.'


def append_range(committed, observed, seconds):
    limit = commit_count(observed, seconds, 0)
    if limit < committed:
        raise ValueError('Committed positions must not move backward')
    return committed, limit


def positions(start, count):
    if start < 0 or count <= 0:
        raise ValueError('Position start must be nonnegative and count positive')
    return list(range(start, start + count))


def run(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'started_at': time.time(), 'samples': [], 'errors': [],
        'model_dir': str(args.model_dir), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'scipy')},
        'protocol': {
            'initial_audio_ms': 1000, 'arrival_interval_ms': 100, 'decode_tokens_per_arrival': 1,
            'end_drain_max_tokens': 64, 'source': 'behavior-sd/test/0000.tar',
            'audio_insertion': 'native audio_bos + new frozen audio embeddings + native audio_eos, inside assistant continuation',
            'positions': 'explicit monotonically increasing 3-axis positions, full causal attention',
            'quality_scope': 'No training. Interleaved audio inside assistant continuation is out of the native turn layout.',
            'timing_scope': 'Serialized source-arrival replay, not real-time microphone; no Talker or playback.',
            'controls': 'natural EOS; EOS-suppressed diagnostic; matched-position silence; full exact-sequence replay',
            'endpoint_scope': 'Known utterance end used for final embedding flush and final text drain',
            'silence_scope': 'Silence is independently prefix-encoded with the same frozen commit policy, not zero embeddings.',
        },
    }

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    save()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    free, total = torch.cuda.mem_get_info()
    if free < 25 * 2**30:
        raise RuntimeError('Insufficient free memory; do not disturb other processes')
    report['gpu'] = {'name': torch.cuda.get_device_name(), 'free_before': free, 'total': total}
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager',
    ).eval()
    model.disable_talker()
    thinker = model.thinker
    embed = thinker.get_input_embeddings()
    device = model.device
    rate = processor.feature_extractor.sampling_rate
    hop = processor.feature_extractor.hop_length
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['implementation_sha256'] = hashlib.sha256(Path(inspect.getfile(type(thinker))).read_bytes()).hexdigest()
    report['config_sha256'] = hashlib.sha256((args.model_dir / 'config.json').read_bytes()).hexdigest()
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    stop_ids.update([processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids('<|im_end|>')])
    stop_ids.discard(None)
    report['stop_ids'] = sorted(stop_ids)
    audio_bos = processor.tokenizer.convert_tokens_to_ids(processor.audio_bos_token)
    audio_eos = processor.tokenizer.convert_tokens_to_ids(processor.audio_eos_token)

    def token_embeddings(ids):
        return embed(torch.tensor([ids], device=device, dtype=torch.long))

    def resample(waveform, source_rate):
        divisor = math.gcd(source_rate, rate)
        return resample_poly(waveform, rate // divisor, source_rate // divisor).astype(np.float32)

    def features(waveform):
        padded_length = max(processor.feature_extractor.n_samples, (math.ceil(len(waveform) / hop) + 1) * hop)
        return processor.feature_extractor(
            waveform, sampling_rate=rate, return_tensors='pt', padding='max_length',
            max_length=padded_length, truncation=False, return_attention_mask=True,
        )

    def encode(waveform, source_rate):
        torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = resample(waveform, source_rate)
        batch = features(waveform)
        hidden = thinker.get_audio_features(
            batch['input_features'].to(device=device, dtype=model.dtype),
            feature_attention_mask=batch['attention_mask'].to(device),
        ).last_hidden_state
        torch.cuda.synchronize()
        return hidden, (time.perf_counter() - started) * 1000

    def make_batch(waveform):
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': INSTRUCTION}, {'type': 'audio', 'audio': 'in-memory.wav'}]},
        ]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return processor(text=prompt, audio=[waveform], sampling_rate=rate, padding=True, return_tensors='pt').to(device).to(model.dtype)

    def decode(ids):
        return processor.tokenizer.decode(ids, skip_special_tokens=True)

    def delta(a, b):
        a, b = a.float(), b.float()
        return {'max_abs_logit': float((a-b).abs().max().item()),
                'mean_abs_logit': float((a-b).abs().mean().item()),
                'argmax_equal': int(a.argmax().item()) == int(b.argmax().item())}

    class Stream:
        def __init__(self, initial):
            self.cache = None
            self.length = 0
            self.parts = []
            self.events = []
            self.logits = None
            self.append(initial, 'prefill')

        def append(self, embeddings, kind):
            count = embeddings.shape[1]
            start = self.length
            position_ids = torch.tensor(positions(start, count), device=device).view(1, 1, -1).expand(3, 1, -1)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = thinker(
                inputs_embeds=embeddings, past_key_values=self.cache, position_ids=position_ids,
                attention_mask=torch.ones((1, start + count), device=device, dtype=torch.long),
                use_cache=True, return_dict=True,
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if self.cache is not None and output.past_key_values is not self.cache:
                raise RuntimeError('KV cache object unexpectedly replaced')
            self.cache = output.past_key_values
            self.length += count
            if self.cache.get_seq_length() != self.length:
                raise RuntimeError('KV length mismatch')
            self.logits = output.logits[:, -1, :]
            self.parts.append(embeddings)
            self.events.append({'kind': kind, 'start': start, 'count': count, 'end': self.length, 'ms': elapsed})
            return elapsed

        def audio(self, hidden):
            if not len(hidden):
                return 0.0
            packet = torch.cat([token_embeddings([audio_bos]), hidden.unsqueeze(0), token_embeddings([audio_eos])], dim=1)
            return self.append(packet, 'late_audio')

        def pick(self, suppress_eos=False):
            logits = self.logits.clone()
            if suppress_eos:
                logits[:, list(stop_ids)] = -torch.inf
            return int(logits.argmax(-1).item())

        def text(self, token):
            return self.append(token_embeddings([token]), 'text')

        def replay(self):
            full = torch.cat(self.parts, dim=1)
            length = full.shape[1]
            result = thinker(
                inputs_embeds=full, position_ids=torch.arange(length, device=device).view(1, 1, -1).expand(3, 1, -1),
                attention_mask=torch.ones((1, length), device=device, dtype=torch.long), use_cache=False, return_dict=True,
            )
            return delta(self.logits, result.logits[:, -1, :])

    selected = []
    with tarfile.open(args.dataset_dir / 'test/0000.tar', 'r:') as archive:
        for member in archive:
            if not member.name.endswith('.json'):
                continue
            metadata = json.load(archive.extractfile(member))
            for utterance in metadata['utterances']:
                start, end = utterance['start_time'], utterance['end_time']
                if not 3 <= end-start <= 12 or utterance.get('backchannels') or utterance['uttr_type'] is not None:
                    continue
                if any(other['speaker_idx'] != utterance['speaker_idx'] and other['start_time'] < end and other['end_time'] > start for other in metadata['utterances']):
                    continue
                if not any(c.isalpha() for c in utterance['tts_text']) or utterance['tts_text'].startswith('['):
                    continue
                audio, source_rate = sf.read(io.BytesIO(archive.extractfile(metadata['file_name']).read()), always_2d=True, dtype='float32')
                clip = audio[round(start * source_rate):round(end * source_rate), utterance['speaker_idx']].copy()
                selected.append(({'case_id': Path(member.name).stem, 'reference': utterance['tts_text'], 'seconds': len(clip)/source_rate,
                                  'source_sample_rate': source_rate, 'source_range': [start, end], 'speaker_idx': utterance['speaker_idx']}, clip))
                break
            if len(selected) == args.sample_count:
                break
    if len(selected) != args.sample_count:
        raise RuntimeError('Insufficient samples')

    with torch.inference_mode():
        encode(selected[0][1][:selected[0][0]['source_sample_rate']], selected[0][0]['source_sample_rate'])
        for info, clip in selected:
            print('SAMPLE', info['case_id'], flush=True)
            sample = {**info, 'streams': {}, 'arrivals': []}
            report['samples'].append(sample)
            source_rate = info['source_sample_rate']
            initial_end = source_rate
            initial_hidden, initial_ms = encode(clip[:initial_end], source_rate)
            initial_batch = make_batch(resample(clip[:initial_end], source_rate))
            placeholders = initial_batch['input_ids'] == thinker.config.audio_token_id
            if int(placeholders.sum()) != len(initial_hidden):
                raise RuntimeError('Initial audio placeholder count mismatch')
            initial = embed(initial_batch['input_ids']).clone()
            initial[placeholders] = initial_hidden
            native = thinker(**initial_batch, use_cache=False, return_dict=True)
            control = Stream(initial)
            initial_control = delta(control.logits, native.logits[:, -1, :])
            if not initial_control['argmax_equal'] or initial_control['max_abs_logit'] > 1:
                raise RuntimeError('Initial embedding/position path differs from native forward')
            sample['initial_native_control'] = initial_control
            del native, control

            full_batch = make_batch(resample(clip, source_rate))
            generated = model.generate(**full_batch, return_audio=False, thinker_max_new_tokens=128, thinker_do_sample=False)
            offline_text = processor.batch_decode(generated[:, full_batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            sample['offline_full'] = {'text': offline_text, **edit_counts(info['reference'], offline_text)}
            del generated, full_batch

            # Precompute arrived-only prefix packets once for identical payloads across all controls.
            packets = []
            committed = len(initial_hidden)
            for right in arrival_ends(len(clip), source_rate, 100):
                if right <= initial_end:
                    continue
                hidden, encode_ms = encode(clip[:right], source_rate)
                silent, _ = encode(np.zeros(right, dtype=np.float32), source_rate)
                if len(silent) != len(hidden):
                    raise RuntimeError('Matched silence changes token positions')
                left, limit = append_range(committed, len(hidden), right/source_rate)
                if right == len(clip):
                    limit = len(hidden)
                packets.append((right/source_rate, hidden[left:limit].clone(), silent[left:limit].clone(), encode_ms))
                sample['arrivals'].append({'seconds': right/source_rate, 'committed_before': left, 'committed_after': limit,
                                           'new_audio_tokens': limit-left, 'encode_ms': encode_ms})
                committed = limit
            sample['initial_encode_ms'] = initial_ms

            for name, suppress, payload in (
                ('natural_live', False, 'real'), ('natural_initial_only', False, 'none'),
                ('forced_live', True, 'real'), ('forced_silence', True, 'silence'),
            ):
                stream = Stream(initial)
                ids, steps = [], []
                ended = False
                stop_at = None
                initial_token = stream.pick(suppress)
                ids.append(initial_token)
                stream.text(initial_token)
                if initial_token in stop_ids:
                    ended, stop_at = True, 1.0
                for now, real, silence, encode_ms in packets:
                    generated_before = len(ids)
                    inserted = len(real) if payload != 'none' else 0
                    audio_ms = stream.audio(real if payload == 'real' else silence) if inserted else 0.0
                    text_ms = 0.0
                    if not ended:
                        token = stream.pick(suppress)
                        ids.append(token)
                        text_ms = stream.text(token)
                        if token in stop_ids:
                            ended, stop_at = True, now
                    steps.append({'seconds': now, 'inserted_tokens': inserted, 'generated_before': generated_before,
                                  'generated_after': len(ids), 'audio_ms': audio_ms, 'text_ms': text_ms,
                                  'encoder_plus_forward_ms': encode_ms + audio_ms + text_ms,
                                  'eos_seen': ended})
                pre_end_ids = ids.copy()
                # After all arrivals, stop suppressing EOS, including for forced diagnostic arms.
                for _ in range(64):
                    if ended:
                        break
                    token = stream.pick(False)
                    ids.append(token)
                    stream.text(token)
                    if token in stop_ids:
                        ended = True
                        if stop_at is None:
                            stop_at = info['seconds']
                text = decode(ids)
                replay = stream.replay()
                if not replay['argmax_equal'] or replay['max_abs_logit'] > 1:
                    raise RuntimeError(f'Cached/full replay mismatch {name}: {replay}')
                sample['streams'][name] = {
                    'text': text, **edit_counts(info['reference'], text), 'output_ids': ids,
                    'text_before_end': decode(pre_end_ids), 'output_tokens_before_end': len(pre_end_ids),
                    'first_eos_audio_seconds': stop_at, 'eos_seen': ended,
                    'late_audio_events': sum(event['kind'] == 'late_audio' for event in stream.events),
                    'late_audio_while_generating': sum(step['inserted_tokens'] > 0 and step['generated_after'] > step['generated_before'] for step in steps),
                    'cache_final_length': stream.length, 'cache_growth_verified': True,
                    'replay_check': replay, 'steps': steps, 'events': stream.events,
                    'tick_compute_ms': summary([step['encoder_plus_forward_ms'] for step in steps]),
                    'ticks_exceeding_100ms': sum(step['encoder_plus_forward_ms'] > 100 for step in steps),
                }
                print('STREAM', name, 'WER', round(sample['streams'][name]['wer'], 3), 'EOS', stop_at,
                      'LIVE_EVENTS', sample['streams'][name]['late_audio_while_generating'], flush=True)
                del stream

            # Same generated text and same positions in both arms: isolate the effect of late audio.
            real_stream, silence_stream = Stream(initial), Stream(initial)
            sample['causal_before_late_audio'] = delta(real_stream.logits, silence_stream.logits)
            first = real_stream.pick(True)
            real_stream.text(first)
            silence_stream.text(first)
            causal = []
            for now, real, silence, _ in packets:
                real_stream.audio(real)
                silence_stream.audio(silence)
                comparison = delta(real_stream.logits, silence_stream.logits)
                comparison['seconds'] = now
                real_pick, silent_pick = real_stream.pick(True), silence_stream.pick(True)
                comparison['greedy_different_without_eos'] = real_pick != silent_pick
                comparison['real_next_token'] = decode([real_pick])
                comparison['silence_next_token'] = decode([silent_pick])
                causal.append(comparison)
                real_stream.text(real_pick)
                silence_stream.text(real_pick)
            sample['causal_matched_text_silence'] = causal
            sample['causal_cache_lengths_equal'] = real_stream.length == silence_stream.length
            del real_stream, silence_stream
            save()
            print('SAMPLE_DONE', info['case_id'], flush=True)

    names = ['offline_full'] + list(report['samples'][0]['streams'])
    report['quality_summary'] = {}
    for name in names:
        methods = [sample['offline_full'] if name == 'offline_full' else sample['streams'][name] for sample in report['samples']]
        edits = sum(method['word_edits'] for method in methods)
        reference_words = sum(method['reference_words'] for method in methods)
        report['quality_summary'][name] = {'samples': len(methods), 'word_edits': edits,
                                           'reference_words': reference_words, 'micro_wer': edits/max(1, reference_words)}
    report['finished_at'] = time.time()
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save()
    print('SUMMARY', json.dumps(report['quality_summary']), flush=True)
    print('PROBE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=8)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error('--sample-count must be positive')
    run(args)
