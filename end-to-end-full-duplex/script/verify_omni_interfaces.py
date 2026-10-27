"""Read-only model/data diagnostics; saves small, task-owned validation artifacts."""

import argparse
import difflib
import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time
import traceback


SYSTEM_PROMPT = (
    'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
    'capable of perceiving auditory and visual inputs, as well as generating text and speech.'
)


def normalize_text(text):
    return ' '.join(re.findall(r"[a-z0-9]+", text.lower()))


def asset_audit(model_dir):
    index_path = model_dir / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    names = index['weight_map']
    markers = ('speech_tokenizer', 'audio_tokenizer', 'quantizer', 'codec_encoder')
    return {
        'config_sha256': hashlib.sha256((model_dir / 'config.json').read_bytes()).hexdigest(),
        'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'weight_count': len(names),
        'shards': {name: (model_dir / name).stat().st_size for name in sorted(set(names.values()))},
        'weight_prefixes': sorted({'.'.join(name.split('.')[:2]) for name in names}),
        'waveform_codec_encoder_name_matches': [name for name in names if any(m in name.lower() for m in markers)],
    }


def exception_record(error):
    return {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()}


def run(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists():
        raise FileExistsError(f'Previous report exists; choose a new output directory: {report_path}')
    report = {
        'started_at': time.time(), 'model_dir': str(args.model_dir),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate', 'numpy', 'soundfile')},
        'assets': asset_audit(args.model_dir), 'tests': {},
    }

    def save():
        temporary = output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(report_path)

    def result(name, value):
        report['tests'][name] = value
        save()
        print('RESULT', name, json.dumps(value, ensure_ascii=False), flush=True)

    save()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA device for this diagnostic')
    free, total = torch.cuda.mem_get_info(0)
    report['gpu'] = {'name': torch.cuda.get_device_name(0), 'free_bytes_before': free, 'total_bytes': total}
    if free < 30 * 2**30:
        raise RuntimeError('Less than 30 GiB free; no other process will be interrupted')

    source_file = Path(inspect.getfile(Qwen2_5OmniForConditionalGeneration))
    report['implementation'] = {'path': str(source_file), 'sha256': hashlib.sha256(source_file.read_bytes()).hexdigest()}
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir, local_files_only=True)
    load_start = time.monotonic()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager',
    ).eval()
    report['load_seconds'] = time.monotonic() - load_start
    report['loaded_parameters'] = sum(parameter.numel() for parameter in model.parameters())
    save()
    print('MODEL_LOADED', report['loaded_parameters'], flush=True)

    shard = args.dataset_dir / 'test' / '0000.tar'
    member = '0000000005'
    with tarfile.open(shard, 'r:') as archive:
        metadata = json.load(archive.extractfile(member + '.json'))
        audio, source_rate = sf.read(io.BytesIO(archive.extractfile(member + '.flac').read()), always_2d=True)
    utterance = metadata['utterances'][0]
    clip = audio[round(utterance['start_time'] * source_rate):round(utterance['end_time'] * source_rate), utterance['speaker_idx']]
    target_rate = processor.feature_extractor.sampling_rate
    divisor = math.gcd(source_rate, target_rate)
    clip = resample_poly(clip, target_rate // divisor, source_rate // divisor).astype(np.float32)
    sf.write(output / 'input.wav', clip, target_rate)
    report['input'] = {'shard': str(shard), 'member': member + '.flac', 'speaker_idx': utterance['speaker_idx'], 'seconds': len(clip) / target_rate, 'sample_rate': target_rate, 'reference_text': utterance['tts_text']}
    save()

    def make_inputs(instruction):
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': 'in-memory.wav'}, {'type': 'text', 'text': instruction}]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        batch = processor(text=text, audio=[clip], sampling_rate=target_rate, return_tensors='pt', padding=True)
        return batch.to(model.device).to(model.dtype)

    inputs = make_inputs('Transcribe the speech exactly in English. Return only the transcription.')
    report['input_tensor_shapes'] = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, 'shape')}
    save()
    with torch.inference_mode():
        try:
            torch.cuda.synchronize()
            start = time.monotonic()
            text_ids = model.generate(**inputs, return_audio=False, thinker_max_new_tokens=64, thinker_do_sample=False)
            torch.cuda.synchronize()
            answer = processor.batch_decode(text_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            similarity = difflib.SequenceMatcher(None, normalize_text(utterance['tts_text']), normalize_text(answer)).ratio()
            result('audio_to_text', {'status': 'passed' if answer.strip() else 'failed', 'transcript': answer, 'reference': utterance['tts_text'], 'normalized_character_similarity': similarity, 'seconds': time.monotonic() - start, 'quality_note': 'Single-example transcription comparison, not a benchmark.'})
        except Exception as error:
            result('audio_to_text', exception_record(error))

        try:
            speech_inputs = make_inputs('Reply to the speaker in English with exactly one short sentence.')
            code_trace = {}
            original_generate = model.talker.generate

            @functools.wraps(original_generate)
            def capture_codes(*positional, **keywords):
                generated = original_generate(*positional, **keywords)
                codes = generated.sequences if hasattr(generated, 'sequences') else generated
                code_trace['shape'] = list(codes.shape)
                code_trace['first_ids'] = codes.detach().cpu().reshape(-1)[:12].tolist()
                np.save(output / 'generated_talker_codes.npy', codes.detach().cpu().numpy())
                return generated

            model.talker.generate = capture_codes
            try:
                torch.cuda.synchronize()
                start = time.monotonic()
                text_ids, speech = model.generate(
                    **speech_inputs, thinker_max_new_tokens=48, talker_max_new_tokens=256,
                    thinker_do_sample=False, talker_do_sample=False,
                )
                torch.cuda.synchronize()
            finally:
                model.talker.generate = original_generate
            waveform = speech.detach().float().cpu().numpy().reshape(-1)
            answer = processor.batch_decode(text_ids[:, speech_inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            finite = bool(np.isfinite(waveform).all())
            rms = float(np.sqrt(np.mean(waveform * waveform))) if len(waveform) else 0.0
            if len(waveform) and finite:
                sf.write(output / 'response.wav', waveform, 24000)
            result('audio_to_text_and_speech', {'status': 'passed' if finite and rms > 1e-5 and answer.strip() else 'failed', 'reply_text': answer, 'waveform_samples': len(waveform), 'sample_rate': 24000, 'seconds_of_audio': len(waveform) / 24000, 'rms': rms, 'finite': finite, 'wall_seconds': time.monotonic() - start, 'talker_codes': code_trace, 'quality_note': 'Pass establishes nonblank text/waveform output; intelligibility and voice quality require separate listening/evaluation.'})
        except Exception as error:
            result('audio_to_text_and_speech', exception_record(error))

        try:
            original_prepare = model.thinker.prepare_inputs_for_generation
            prepare_source = inspect.getsource(original_prepare)
            (output / 'prepare_inputs_for_generation.txt').write_text(prepare_source)
            fresh = processor.feature_extractor(
                np.zeros(round(target_rate * 0.1), dtype=np.float32), sampling_rate=target_rate,
                return_tensors='pt', padding='longest', return_attention_mask=True,
            )['input_features'].to(device=model.device, dtype=model.dtype)

            def generate_with_probe(inject):
                trace = []
                encoder_calls = []
                hook = model.thinker.audio_tower.register_forward_pre_hook(lambda _module, _args: encoder_calls.append(time.monotonic()))

                @functools.wraps(original_prepare)
                def prepare(*positional, **keywords):
                    requested = bool(inject and trace)
                    if requested:
                        keywords['input_features'] = fresh
                    prepared = original_prepare(*positional, **keywords)
                    trace.append({'step': len(trace), 'late_audio_requested': requested, 'input_features_forwarded': prepared.get('input_features') is not None})
                    return prepared

                model.thinker.prepare_inputs_for_generation = prepare
                try:
                    generated = model.thinker.generate(**inputs, max_new_tokens=8, min_new_tokens=4, do_sample=False)
                finally:
                    model.thinker.prepare_inputs_for_generation = original_prepare
                    hook.remove()
                return generated, trace, len(encoder_calls)

            control, control_trace, control_calls = generate_with_probe(False)
            changed, changed_trace, changed_calls = generate_with_probe(True)
            late = [step for step in changed_trace if step['late_audio_requested']]
            dropped = bool(late) and all(not step['input_features_forwarded'] for step in late)
            result('incremental_audio_during_cached_generation', {
                'status': 'unsupported_by_unmodified_cached_generation' if dropped else 'requires_further_integration_test',
                'evidence_type': 'runtime_cached_generation_probe_plus_source',
                'injected_audio_seconds': 0.1, 'control_trace': control_trace, 'injection_trace': changed_trace,
                'control_audio_encoder_calls': control_calls, 'injection_audio_encoder_calls': changed_calls,
                'output_token_ids_identical': bool(torch.equal(control, changed)),
                'scope': 'Tests existing Thinker cached generation. It does not establish that a custom scheduler/model adaptation is impossible.',
            })
        except Exception as error:
            result('incremental_audio_during_cached_generation', exception_record(error))

        try:
            original_features = inputs['input_features']
            modified = original_features.clone()
            modified[..., 10:200] = 0
            options = {'feature_attention_mask': inputs['feature_attention_mask']}
            before = model.thinker.get_audio_features(original_features, **options)
            after = model.thinker.get_audio_features(modified, **options)
            def hidden(value):
                if isinstance(value, torch.Tensor):
                    return value
                if hasattr(value, 'last_hidden_state'):
                    return value.last_hidden_state
                return torch.cat(list(value), dim=0)
            before, after = hidden(before), hidden(after)
            delta = float((before[:2].float() - after[:2].float()).abs().max().item())
            result('encoder_prefix_causality', {'status': 'future_dependent_within_supplied_chunk' if delta > 1e-4 else 'no_difference_in_this_probe', 'max_abs_difference_first_two_audio_embeddings': delta, 'unchanged_input_mel_frames': 10, 'modified_future_mel_frames': [10,200], 'scope': 'Fixed-length mel-feature perturbation, not a waveform benchmark or proof of streaming equivalence.'})
        except Exception as error:
            result('encoder_prefix_causality', exception_record(error))

    modules = {'thinker': model.thinker, 'talker': model.talker, 'token2wav': model.token2wav}
    methods = {name: {method: callable(getattr(module, method, None)) for method in ('encode', 'decode', 'forward', 'generate')} for name, module in modules.items()}
    named_codec_matches = report['assets']['waveform_codec_encoder_name_matches']
    result('real_audio_to_native_talker_codes', {
        'status': 'blocked_no_compatible_encoder_found_in_inspected_assets',
        'evidence_type': 'local_asset_and_loaded_interface_inspection',
        'module_methods': methods, 'encoder_weight_name_matches': named_codec_matches,
        'talker_vocab_size': model.talker.config.vocab_size,
        'generated_codes_to_waveform': report['tests'].get('audio_to_text_and_speech',{}).get('status'),
        'scope': 'Only this local checkpoint and installed classes were inspected. Does not claim that no external compatible encoder exists.',
        'not_valid_substitutes': ['Thinker continuous audio embeddings', 'text tokenizer IDs', 'another codec IDs', 'model-generated codes treated as ground-truth waveform encodings'],
    })
    report['finished_at'] = time.time()
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    save()
    print('VERIFICATION_FINISHED', str(report_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
