"""Compare unchanged Omni audio encoding on arrived-only chunks and prefixes."""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tarfile
import time
import traceback


SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
          'capable of perceiving auditory and visual inputs, as well as generating text and speech.')


def arrival_ends(length, sample_rate, interval_ms):
    step = round(sample_rate * interval_ms / 1000)
    if length <= 0 or step <= 0:
        raise ValueError('Audio length and update interval must be positive')
    return list(range(step, length, step)) + [length]


def commit_count(token_count, arrived_seconds, holdback_ms):
    # Observed encoder output rate is 25 Hz; only full output positions commit.
    return min(token_count, max(0, math.floor((arrived_seconds - holdback_ms / 1000) * 25 + 1e-7)))


def endpoint_padding_samples(length, sample_rate):
    if length <= 0 or sample_rate <= 0:
        raise ValueError('Audio length and sample rate must be positive')
    return max(0, math.ceil(sample_rate * .04) - length)


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace('\u2019', "'"))


def edit_counts(reference, hypothesis):
    left, right = words(reference), words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return {'word_edits': previous[-1], 'reference_words': len(left), 'wer': previous[-1] / max(1, len(left))}


def summary(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': values[min(len(values) - 1, math.ceil(len(values) * .95) - 1)], 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=8)
    parser.add_argument('--strict-transcription', action='store_true',
                        help='Use a transcription-only system prompt and place the instruction before audio')
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error('--sample-count must be positive')
    run(args)


def run(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torch.nn.functional as functional
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'started_at': time.time(), 'model_dir': str(args.model_dir),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'scipy')},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'samples': [], 'errors': [],
        'transcription_protocol': 'strict_text_before_audio' if args.strict_transcription else 'native_system_text_after_audio',
        'endpoint_padding': {'minimum_ms': 40, 'side': 'right', 'value': 0, 'events': []},
        'scope': [
            'Unchanged pretrained weights and n_window; single-stream audio-encoder feasibility, not a trained full-duplex system.',
            'Chunk and prefix waveform slices are taken BEFORE resampling and Mel extraction; no future waveform is supplied.',
            'Final concatenated-embedding transcription uses all arrived embeddings, not incremental Thinker KV insertion.',
            'Prefix-final is the same final input as offline baseline, not independent evidence of online quality.',
            'Final incomplete block and holdback flush assume known utterance end; endpoint detection is not tested.',
            'Residual blocks shorter than 40ms are padded with synthetic silence after resampling; no future real audio is used.',
            'Timing includes Python, resampling, Mel, transfer and synchronized encoding; not Thinker/Talker or capture/network latency.',
            'Small selected English test-set sample; no claim about training behavior, general accuracy or production latency.',
        ],
    }

    def save():
        temporary = output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'report.json')

    def error(where, exception):
        record = {'where': where, 'error': str(exception), 'traceback': traceback.format_exc()}
        report['errors'].append(record)
        print('ERROR', where, repr(exception), flush=True)
        save()

    save()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA device')
    free, total = torch.cuda.mem_get_info(0)
    if free < 25 * 2**30:
        raise RuntimeError('Insufficient free GPU memory; other processes left untouched')
    report['gpu'] = {'name': torch.cuda.get_device_name(0), 'free_bytes_before': free, 'total_bytes': total}
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager',
    ).eval()
    model.disable_talker()
    thinker = model.thinker
    encoder = thinker.audio_tower
    rate = processor.feature_extractor.sampling_rate
    source_file = Path(inspect.getfile(type(encoder)))
    report['encoder'] = {'n_window': encoder.n_window, 'target_sample_rate': rate,
                         'class_source_sha256': hashlib.sha256(source_file.read_bytes()).hexdigest(),
                         'config_sha256': hashlib.sha256((args.model_dir / 'config.json').read_bytes()).hexdigest()}
    report['preprocessing'] = {
        'padding': 'max_length', 'truncation': False,
        'minimum_padded_samples': processor.feature_extractor.n_samples,
        'long_input_policy': 'extend padded length beyond arrived audio; never truncate at native default',
        'reason': 'Match native processor padding, frame masks and right-edge zero context.',
    }
    print('MODEL_LOADED', json.dumps(report['encoder']), flush=True)
    save()

    selected = []
    with tarfile.open(args.dataset_dir / 'test/0000.tar', 'r:') as archive:
        for member in archive:
            if not member.name.endswith('.json'):
                continue
            data = json.load(archive.extractfile(member))
            eligible = []
            for utterance in data['utterances']:
                start, end = utterance['start_time'], utterance['end_time']
                if not 3 <= end - start <= 12 or utterance.get('backchannels') or utterance['uttr_type'] is not None:
                    continue
                if any(other['speaker_idx'] != utterance['speaker_idx'] and other['start_time'] < end and other['end_time'] > start for other in data['utterances']):
                    continue
                if not any(character.isalpha() for character in utterance['tts_text']) or utterance['tts_text'].startswith('['):
                    continue
                eligible.append(utterance)
            if not eligible:
                continue
            # One utterance per dialogue; selection is independent of model output.
            u = eligible[0]
            audio, source_rate = sf.read(io.BytesIO(archive.extractfile(data['file_name']).read()), always_2d=True, dtype='float32')
            clip = audio[round(u['start_time'] * source_rate):round(u['end_time'] * source_rate), u['speaker_idx']].copy()
            selected.append(({'case_id': Path(member.name).stem, 'utterance_index': u['uttr_idx'], 'speaker_idx': u['speaker_idx'],
                              'reference': u['tts_text'], 'seconds': len(clip) / source_rate,
                              'original_time_range': [u['start_time'], u['end_time']], 'source_sample_rate': source_rate}, clip))
            if len(selected) == args.sample_count:
                break
    if len(selected) != args.sample_count:
        raise RuntimeError('Not enough eligible nonoverlapping utterances in selected shard')

    @torch.inference_mode()
    def encode(waveform, source_rate):
        torch.cuda.synchronize()
        started = time.perf_counter()
        divisor = math.gcd(source_rate, rate)
        waveform = resample_poly(waveform, rate // divisor, source_rate // divisor).astype(np.float32)
        padding = endpoint_padding_samples(len(waveform), rate)
        if padding:
            report['endpoint_padding']['events'].append({'original_samples': len(waveform), 'padding_samples': padding})
            waveform = np.pad(waveform, (0, padding))
        hop = processor.feature_extractor.hop_length
        padded_length = max(processor.feature_extractor.n_samples, (math.ceil(len(waveform) / hop) + 1) * hop)
        batch = processor.feature_extractor(
            waveform, sampling_rate=rate, return_tensors='pt', padding='max_length',
            max_length=padded_length, truncation=False, return_attention_mask=True,
        )
        features = batch['input_features'].to(device=model.device, dtype=model.dtype)
        mask = batch['attention_mask'].to(model.device)
        hidden = thinker.get_audio_features(features, feature_attention_mask=mask).last_hidden_state
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - started) * 1000
        return hidden, milliseconds

    @torch.inference_mode()
    def transcribe(hidden, real_waveform=None):
        # Use the native processor to produce exactly the required placeholder/position count.
        # Synthetic silence determines ONLY placeholder shape; its features are overridden below.
        system = ('You are a speech transcription engine. Transcribe the supplied audio verbatim in English. '
                  'Output only the spoken words. Do not answer questions in the audio, respond to the speaker, '
                  'give advice, or add explanations.' if args.strict_transcription else SYSTEM)
        content = [{'type': 'audio', 'audio': 'in-memory.wav'},
                   {'type': 'text', 'text': 'Transcribe the speech exactly in English. Return only the transcription.'}]
        if args.strict_transcription:
            content.reverse()
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
            {'role': 'user', 'content': content},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        dummy = (np.zeros(len(hidden) * 4 * processor.feature_extractor.hop_length, dtype=np.float32)
                 if real_waveform is None else real_waveform)
        batch = processor(text=prompt, audio=[dummy], sampling_rate=rate, padding=True, return_tensors='pt').to(model.device).to(model.dtype)
        count = int((batch['input_ids'] == thinker.config.audio_token_id).sum().item())
        if count != len(hidden):
            raise RuntimeError(f'Placeholder mismatch {count} != {len(hidden)}')
        if real_waveform is not None:
            generated = model.generate(**batch, return_audio=False, thinker_max_new_tokens=128, thinker_do_sample=False)
            return processor.batch_decode(generated[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        original = thinker.get_audio_features
        calls = []

        def supplied(*_positional, **_keywords):
            calls.append(True)
            return BaseModelOutputWithPooling(last_hidden_state=hidden)

        thinker.get_audio_features = supplied
        try:
            generated = model.generate(**batch, return_audio=False, thinker_max_new_tokens=128, thinker_do_sample=False)
        finally:
            thinker.get_audio_features = original
        if len(calls) != 1:
            raise RuntimeError(f'Expected exactly one embedding injection, got {len(calls)}')
        answer = processor.batch_decode(generated[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return answer

    def difference(current, reference):
        length = min(len(current), len(reference))
        if not length:
            return {'tokens_compared': 0}
        current, reference = current[:length].float(), reference[:length].float()
        return {'tokens_compared': length, 'mean_cosine': float(functional.cosine_similarity(current, reference, dim=-1).mean().item()),
                'relative_l2': float(torch.linalg.vector_norm(current - reference).item() / max(torch.linalg.vector_norm(reference).item(), 1e-12))}

    all_timings = defaultdict(list)
    for _ in range(3):
        encode(selected[0][1][:round(selected[0][0]['source_sample_rate'] * .5)], selected[0][0]['source_sample_rate'])

    for info, clip in selected:
        source_rate = info['source_sample_rate']
        sample = {**info, 'methods': {}, 'prefix_updates': [], 'prefix_revision_by_age': {}}
        report['samples'].append(sample)
        reference = info['reference']
        print('SAMPLE', info['case_id'], round(info['seconds'], 3), flush=True)
        with torch.inference_mode():
            baseline, elapsed = encode(clip, source_rate)
            baseline_text = transcribe(baseline)
            sample['methods']['offline_full'] = {'transcript': baseline_text, **edit_counts(reference, baseline_text), 'tokens': len(baseline), 'encode_ms': elapsed}
            if len(report['samples']) == 1:
                divisor = math.gcd(source_rate, rate)
                real_waveform = resample_poly(clip, rate // divisor, source_rate // divisor).astype(np.float32)
                native_text = transcribe(baseline, real_waveform=real_waveform)
                report['native_transcription_control'] = {
                    'case_id': info['case_id'], 'native_transcript': native_text,
                    'injected_transcript': baseline_text, 'exact_match': native_text == baseline_text,
                }
                if native_text != baseline_text:
                    raise RuntimeError('Native waveform control differs from injected baseline')
            for ms in (100, 200, 500, 1000, 2000):
                name = f'independent_{ms}ms'
                blocks, times = [], []
                try:
                    left = 0
                    for right in arrival_ends(len(clip), source_rate, ms):
                        hidden, elapsed = encode(clip[left:right], source_rate)
                        blocks.append(hidden)
                        times.append(elapsed)
                        left = right
                    joined = torch.cat(blocks)
                    answer = transcribe(joined)
                    sample['methods'][name] = {'transcript': answer, **edit_counts(reference, answer), 'tokens': len(joined),
                                               'encode_ms': summary(times), 'encoding_total_ms': sum(times),
                                               'rtf': sum(times) / (1000 * info['seconds']),
                                               'token_count_difference_from_baseline': len(joined) - len(baseline)}
                    all_timings[name].extend(times)
                    print('METHOD', name, 'WER', round(sample['methods'][name]['wer'], 3), flush=True)
                except Exception as exception:
                    error(info['case_id'] + '/' + name, exception)

            pieces = {holdback: [] for holdback in (0, 200, 500, 2000)}
            committed = {holdback: 0 for holdback in pieces}
            first_available = {}
            revisions = defaultdict(list)
            prefix_times = []
            previous = None
            previous_seconds = 0
            selected_prefix_texts = []
            last = None
            for right in arrival_ends(len(clip), source_rate, 100):
                current, elapsed = encode(clip[:right], source_rate)
                now = right / source_rate
                last = current
                prefix_times.append(elapsed)
                step = {'arrived_seconds': now, 'tokens': len(current), 'compute_ms': elapsed}
                if previous is not None:
                    for age_ms in (0, 100, 200, 500, 1000, 2000):
                        stable_count = min(len(previous), len(current), max(0, math.floor((previous_seconds - age_ms / 1000) * 25 + 1e-7)))
                        if stable_count:
                            delta = difference(current[:stable_count], previous[:stable_count])
                            revisions[age_ms].append(delta)
                for holdback in pieces:
                    limit = commit_count(len(current), now, holdback)
                    if limit > committed[holdback]:
                        pieces[holdback].append(current[committed[holdback]:limit].clone())
                        committed[holdback] = limit
                        first_available.setdefault(holdback, now)
                if any(abs(now - checkpoint) < .0001 for checkpoint in (.5, 1, 2, 4)):
                    selected_prefix_texts.append({'arrived_seconds': now, 'transcript': transcribe(current)})
                previous = current
                previous_seconds = now
                sample['prefix_updates'].append(step)
            assert last is not None
            sample['prefix_update_compute_ms'] = summary(prefix_times)
            sample['prefix_compute_total_ms'] = sum(prefix_times)
            sample['prefix_rtf'] = sum(prefix_times) / (1000 * info['seconds'])
            sample['prefix_updates_exceeding_100ms'] = sum(value > 100 for value in prefix_times)
            sample['partial_snapshot_transcripts'] = selected_prefix_texts
            sample['final_prefix_vs_offline'] = difference(last, baseline)
            sample['prefix_revision_by_age'] = {str(age): {
                'mean_cosine': summary([item['mean_cosine'] for item in items]),
                'relative_l2': summary([item['relative_l2'] for item in items]),
            } for age, items in revisions.items()}
            for holdback, segments in pieces.items():
                # End flush uses known end-of-utterance; this is explicitly not an endpoint detector.
                if committed[holdback] < len(last):
                    segments.append(last[committed[holdback]:].clone())
                joined = torch.cat(segments)
                if len(joined) != len(baseline):
                    raise RuntimeError('Append-only prefix output does not match final token count')
                answer = transcribe(joined)
                sample['methods'][f'prefix_append_holdback_{holdback}ms'] = {
                    'transcript': answer, **edit_counts(reference, answer), 'tokens': len(joined),
                    'first_commit_arrival_seconds': first_available.get(holdback), 'holdback_ms': holdback,
                    'embedding_difference_from_offline': difference(joined, baseline),
                    'end_flush': 'oracle_utterance_end',
                }
            all_timings['prefix_recompute_100ms'].extend(prefix_times)
        save()
        print('SAMPLE_DONE', info['case_id'], flush=True)

    # Repeated content is used ONLY for computation scaling, never as a quality sample.
    base_info, base_clip = selected[0]
    source_rate = base_info['source_sample_rate']
    scaling = []
    for seconds in (2, 4, 8, 16, 32):
        size = round(seconds * source_rate)
        synthetic = np.tile(base_clip, math.ceil(size / len(base_clip)))[:size]
        encode(synthetic, source_rate)
        times = [encode(synthetic, source_rate)[1] for _ in range(5)]
        scaling.append({'prefix_seconds': seconds, 'compute_ms': summary(times), 'synthetic_repeated_input': True})
    report['prefix_length_scaling'] = scaling
    report['timing_summary_ms'] = {name: summary(values) for name, values in all_timings.items()}
    totals = defaultdict(lambda: {'word_edits': 0, 'reference_words': 0, 'samples': 0})
    for sample in report['samples']:
        for name, method in sample['methods'].items():
            totals[name]['word_edits'] += method['word_edits']
            totals[name]['reference_words'] += method['reference_words']
            totals[name]['samples'] += 1
    report['quality_summary'] = {name: {**values, 'micro_wer': values['word_edits'] / max(1, values['reference_words'])} for name, values in totals.items()}
    report['finished_at'] = time.time()
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    save()
    print('SUMMARY', json.dumps({'quality': report['quality_summary'], 'timing': report['timing_summary_ms'], 'scaling': scaling, 'errors': len(report['errors'])}), flush=True)
    print('PROBE_COMPLETE', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
