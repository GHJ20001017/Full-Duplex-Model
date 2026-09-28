"""Raw schema-2 alignments, not quality-approved examples.

Text is tokenized once without special tokens. Fast offsets locate tokens in the
reference; slow tokenizers retain every ID with explicitly estimated positions.
Collisions spill right, never overwrite. End timestamps do not limit frame starts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import torch
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)
_VERSION = 2
_SAMPLE_RATE = 24000
_FRAME_RATE = 12.5


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sample_digest(sample):
    return _digest({key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                          'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
                    if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()})


def _identity(obj, *, required=False):
    explicit = getattr(obj, 'cache_identity', None)
    if explicit is not None:
        return {'class': type(obj).__module__ + '.' + type(obj).__qualname__,
                'identity': explicit}
    backend = getattr(obj, 'backend_tokenizer', None)
    if backend is not None:
        # Backend serialization includes vocabulary, added-token flags and
        # normalization/pre-tokenization. Constructor kwargs are neither the
        # effective backend state nor reliably JSON-safe (AddedToken/Path/dtype).
        special = getattr(obj, 'special_tokens_map', {})
        return {'backend': backend.to_str(),
                'special_tokens': {str(key): [str(token) for token in value]
                                   if isinstance(value, (list, tuple)) else str(value)
                                   for key, value in special.items()}}
    if required:
        raise ValueError('Caching requires tokenizer.cache_identity or a fast backend, and codec.cache_identity')
    return {'class': type(obj).__module__ + '.' + type(obj).__qualname__}


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite nonnegative number')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return result


def _group(identifier):
    # Emilia source/program IDs precede segment (_S...) and optional word (_W...)
    # suffixes. Group conservatively at the source level, not at individual clips.
    match = re.fullmatch(r'([A-Za-z]{2}_.+?)_(?:S\d+|W\d+)(?:_[SW]\d+)*', identifier)
    if not match:
        raise ValueError(f'Cannot parse Emilia program/session ID: {identifier}')
    return match.group(1)


def _canonical(text):
    return ''.join(c.casefold() for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def _spans(row):
    """Match all alignment units sequentially, retaining original character spans."""
    text = row['text']
    canonical, positions = '', []
    for i, char in enumerate(text):
        normalized = _canonical(char)
        canonical += normalized
        positions.extend([i] * len(normalized))
    cursor, spans = 0, []
    for unit in row['timestamps']:
        needle = _canonical(unit['text'])
        if not needle:
            # Punctuation-only units are not lexical anchors.
            continue
        if not canonical.startswith(needle, cursor):
            raise ValueError('Alignment text does not match complete reference text')
        stop = cursor + len(needle)
        spans.append((positions[cursor], positions[stop - 1] + 1, unit['start'], unit['end']))
        cursor = stop
    if cursor != len(canonical) or not spans:
        raise ValueError('Alignment does not cover complete reference text')
    return spans


def _load_audio(path):
    # Optional decoding dependencies are not needed for manifest inspection.
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if rate <= 0 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must be nonempty and finite')
    mono = audio.mean(axis=1)
    if rate != _SAMPLE_RATE:
        divisor = math.gcd(rate, _SAMPLE_RATE)
        mono = resample_poly(mono, _SAMPLE_RATE // divisor, rate // divisor)
    return torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))


class EmiliaDataset(Dataset):
    """Callable codec accepts a mono float32 CPU waveform at 24 kHz.

    It must return integer CPU codes [8,T] and expose eight ``codebook_sizes``.
    With caching enabled, it must also expose a stable JSON ``cache_identity``
    identifying weights/configuration. Mock/slow tokenizers require the same
    identity; fast tokenizer backend serialization is otherwise used.
    ``data_root`` replaces the parent of the configured dataset directory.
    ``records`` contains the selected latest generated rows, with resolved paths
    and a ``group`` key. ``stats`` reports journal selection and filtering.
    """

    def __init__(self, manifest, tokenizer, codec, text_pad_token_id, *, split='train',
                 validation_fraction=0.05, seed=0, cache_dir=None, data_root=None,
                 max_samples=None, min_duration=0.0, max_duration=None):
        self.manifest = Path(manifest).resolve()
        self.tokenizer, self.codec = tokenizer, codec
        if not isinstance(text_pad_token_id, int) or isinstance(text_pad_token_id, bool) or text_pad_token_id < 0:
            raise ValueError('text_pad_token_id must be a nonnegative integer')
        self.text_pad_token_id = text_pad_token_id
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
            raise ValueError('validation_fraction must be in [0, 1)')
        if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0):
            raise ValueError('max_samples must be a positive integer')
        low = _number(min_duration, 'min_duration')
        high = None if max_duration is None else _number(max_duration, 'max_duration')
        if high is not None and high < low:
            raise ValueError('max_duration must be >= min_duration')
        self.codebook_sizes = tuple(getattr(codec, 'codebook_sizes', ()))
        if len(self.codebook_sizes) != 8 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in self.codebook_sizes):
            raise ValueError('codec.codebook_sizes must contain eight positive integers')
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.identities = {'tokenizer': _identity(tokenizer, required=cache_dir is not None),
                           'codec': _identity(codec, required=cache_dir is not None),
                           'codebook_sizes': self.codebook_sizes, 'pad': text_pad_token_id,
                           'version': _VERSION, 'sample_rate': _SAMPLE_RATE,
                           'frame_rate': _FRAME_RATE}
        latest, total = {}, 0
        with self.manifest.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get('metadata'), str) or not row['metadata']:
                        raise ValueError('Missing metadata path')
                    if row.get('config', {}).get('schema') != 2:
                        raise ValueError('Expected schema 2')
                    if row.get('status') not in ('error', 'generated'):
                        raise ValueError('Expected error or generated status')
                    # Validate generated attempts even when superseded: malformed
                    # generated data is never hidden by filtering or later retries.
                    if row['status'] == 'generated':
                        self._validate_row(row)
                    latest[row['metadata']] = row
                    total += 1
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f'{self.manifest}:{line_number} ({locals().get("row", {}).get("id", "unknown") if isinstance(locals().get("row"), dict) else "unknown"}): {exc}') from exc
        self.stats = {'journal_rows': total, 'superseded': total - len(latest),
                      'errors': 0, 'generated': 0, 'duration_filtered': 0}
        candidates = []
        for row in latest.values():
            if row['status'] == 'error':
                self.stats['errors'] += 1
                continue
            self.stats['generated'] += 1
            row = dict(row)
            metadata = Path(row['metadata'])
            roots = [Path(p) for p in row['config'].get('datasets', [])]
            matches = [root for root in roots if metadata.is_relative_to(root)]
            if len(matches) != 1:
                raise ValueError(f'{row["id"]}: metadata must belong to exactly one configured dataset')
            root = matches[0]
            row['group'] = root.name + ':' + _group(row['id'])
            for key in ('metadata', 'audio'):
                path = Path(row[key])
                if data_root is not None:
                    if not path.is_relative_to(root):
                        raise ValueError(f'{row["id"]}: {key} is outside its dataset root')
                    path = Path(data_root) / root.name / path.relative_to(root)
                elif not path.is_absolute():
                    path = self.manifest.parent / path
                row[key] = str(path.resolve())
            if row['duration'] < low or (high is not None and row['duration'] > high):
                self.stats['duration_filtered'] += 1
                continue
            candidates.append(row)
        # A fixed hash threshold is stable under journal order, retries and additions.
        def validation(group):
            return int(_digest([seed, group]), 16) / 2**256 < validation_fraction
        self.records = sorted((r for r in candidates if validation(r['group']) == (split == 'val')),
                              key=lambda r: (r['group'], r['metadata']))
        self.stats['split_records'] = len(self.records)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.stats['selected'] = len(self.records)
        _LOG.info('Emilia manifest selection: %s', self.stats)
        if self.stats['errors']:
            _LOG.warning('Excluded %d latest Emilia error rows (not generated); stats=%s',
                         self.stats['errors'], self.stats)
        if not self.records:
            raise ValueError(f'Empty {split} split: {self.stats}; choose another fraction/seed or provide more groups')
        self.fingerprint = _digest({'records': self.records, 'identities': self.identities,
                                    'split': split, 'fraction': validation_fraction, 'seed': seed})
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_row(row):
        for key in ('id', 'text', 'audio', 'metadata_sha256'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f'Missing/non-string {key}')
        if not re.fullmatch(r'[0-9a-f]{64}', row['metadata_sha256']):
            raise ValueError('Invalid metadata_sha256')
        _group(row['id'])
        row['duration'] = _number(row['duration'], 'duration')
        if row['duration'] == 0 or row.get('timestamp_unit') != 'seconds':
            raise ValueError('Expected positive duration and timestamps in seconds')
        if not isinstance(row.get('timestamps'), list) or not row['timestamps']:
            raise ValueError('Missing timestamps')
        previous = -1.0
        for unit in row['timestamps']:
            if not isinstance(unit.get('text'), str) or not unit['text']:
                raise ValueError('Invalid alignment text')
            start, end = (_number(unit[k], k) for k in ('start', 'end'))
            if start < previous or end < start:
                raise ValueError('Timestamp starts must be ordered and ends >= starts')
            # Ends may overrun duration; only starts are projected to frames.
            unit['start'], unit['end'] = start, end
            previous = start
        _spans(row)

    def __len__(self):
        return len(self.records)

    def _text(self, row, frames, duration):
        text, spans = row['text'], _spans(row)
        try:
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded['offset_mapping']
            mode = 'fast_offsets'
        except (NotImplementedError, TypeError, KeyError):
            encoded = self.tokenizer(text, add_special_tokens=False)
            offsets = None
            mode = 'estimated_uniform_token_positions'
        ids = list(encoded['input_ids'])
        if not ids or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids):
            raise ValueError('Tokenizer must return nonempty flat integer input_ids')
        if offsets is None:
            offsets = [(int(i * len(text) / len(ids)), int((i + 1) * len(text) / len(ids))) for i in range(len(ids))]
        if len(offsets) != len(ids):
            raise ValueError('Tokenizer offsets and IDs differ in length')
        result = torch.full((frames,), self.text_pad_token_id, dtype=torch.long)
        last, collisions, estimated = -1, 0, False
        for token, (begin, end) in zip(ids, offsets):
            if not 0 <= begin <= end <= len(text):
                raise ValueError('Invalid tokenizer offsets')
            # Leading spaces and punctuation use the following lexical anchor;
            # trailing punctuation uses the last anchor.
            position = begin
            while position < end and not _canonical(text[position]):
                position += 1
            anchor = next((s for s in spans if s[1] > position), spans[-1])
            a, b, start, stop = anchor
            # Tokens wholly after the final lexical span (e.g. a separate
            # punctuation token) borrow its final character's anchor, not the
            # utterance-end boundary, which has no corresponding audio frame.
            position = min(position, b - 1)
            fraction = max(0.0, (position - a) / max(1, b - a))
            estimated |= b - a > 1 and fraction > 0
            # Internal word token positions are estimates, not forced token alignment.
            when = start + fraction * max(0.0, min(stop, duration) - start)
            desired = math.floor(when * _FRAME_RATE)
            frame = max(desired, last + 1)
            collisions += int(frame != desired)
            if frame >= frames or frame < 0:
                raise ValueError(f'{row["id"]}: text alignment overflow at token {token}, frame {frame}/{frames}; no clipping')
            result[frame] = token
            last = frame
        return result, {'token_alignment': mode, 'word_internal_start_estimated': estimated,
                        'collision_policy': 'spill_right', 'collision_spills': collisions,
                        'token_count': len(ids)}

    def _validate_codes(self, codes):
        if not isinstance(codes, torch.Tensor) or codes.device.type != 'cpu' or codes.ndim != 2 or codes.shape[0] != 8 or codes.shape[1] == 0:
            raise ValueError('Codec must return CPU [8,T] with T > 0')
        if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('Codec codes must be integers')
        for i, cardinality in enumerate(self.codebook_sizes):
            if bool(((codes[i] < 0) | (codes[i] >= cardinality)).any()):
                raise ValueError(f'Codec codebook {i} is outside its cardinality')

    def __getitem__(self, index):
        row = self.records[index]
        try:
            metadata_hash = _checksum(row['metadata'])
            if metadata_hash != row['metadata_sha256']:
                raise ValueError('Source metadata checksum differs from alignment journal')
            provenance = {'audio_sha256': _checksum(row['audio']),
                          'metadata_sha256': metadata_hash, 'row': row,
                          'identities': self.identities}
            key = _digest(provenance)
            path = None if self.cache_dir is None else self.cache_dir / (key + '.pt')
            if path is not None and path.exists():
                try:
                    cached = torch.load(path, map_location='cpu', weights_only=True)
                    sample = cached['sample']
                    self._validate_codes(sample['audio_tokens'])
                    expected_text, diagnostics = self._text(row, sample['audio_tokens'].shape[1], sample['duration'])
                    if (cached['key'] == key and cached['provenance'] == provenance
                            and sample['id'] == row['id'] and sample['metadata'] == row['metadata']
                            and sample['sample_rate'] == _SAMPLE_RATE
                            and sample['frame_rate'] == _FRAME_RATE
                            and type(sample['num_frames']) is int
                            and sample['num_frames'] == sample['audio_tokens'].shape[1]
                            and sample['text_tokens'].dtype == torch.long
                            and torch.equal(sample['text_tokens'], expected_text)
                            and sample['alignment'] == diagnostics
                            and cached['sample_sha256'] == _sample_digest(sample)):
                        return sample
                except Exception as exc:
                    _LOG.warning('Invalid Emilia cache %s; recomputing: %s', path, exc)
            waveform = _load_audio(row['audio'])
            duration = waveform.numel() / _SAMPLE_RATE
            if abs(duration - row['duration']) > max(0.1, row['duration'] * 0.01):
                raise ValueError('Decoded duration disagrees with aligned utterance')
            with torch.no_grad():
                codes = self.codec(waveform)
            self._validate_codes(codes)
            codes = codes.detach().to(dtype=torch.long).contiguous()
            text, diagnostics = self._text(row, codes.shape[1], duration)
            sample = {'audio_tokens': codes, 'text_tokens': text, 'id': row['id'],
                      'duration': duration, 'metadata': row['metadata'], 'alignment': diagnostics,
                      'sample_rate': _SAMPLE_RATE, 'frame_rate': _FRAME_RATE,
                      'num_frames': codes.shape[1]}
            if path is not None:
                # Reject a source modified during decode rather than caching mismatched data.
                if _checksum(row['audio']) != provenance['audio_sha256'] or _checksum(row['metadata']) != metadata_hash:
                    raise ValueError('Source changed during encoding')
                name = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.tmp', delete=False) as handle:
                        name = handle.name
                        torch.save({'key': key, 'provenance': provenance, 'sample': sample,
                                    'sample_sha256': _sample_digest(sample)}, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(name, path)
                finally:
                    if name is not None and os.path.exists(name):
                        os.unlink(name)
            return sample
        except Exception as exc:
            raise ValueError(f'{row["id"]}: {exc}') from exc


def collate_emilia(batch):
    """Pad time only; -1 denotes batch padding, not within-utterance text silence."""
    if not batch:
        raise ValueError('Cannot collate an empty batch')
    for sample in batch:
        audio, text = sample['audio_tokens'], sample['text_tokens']
        if (audio.ndim != 2 or audio.shape[0] != 8 or audio.shape[1] == 0
                or text.shape != (audio.shape[1],) or audio.device.type != 'cpu'
                or text.device.type != 'cpu' or audio.dtype != torch.long or text.dtype != torch.long):
            raise ValueError('Expected CPU long audio [8,T] and text [T]')
    length = max(s['audio_tokens'].shape[1] for s in batch)
    audio = torch.full((len(batch), 8, length), -1, dtype=torch.long)
    text = torch.full((len(batch), length), -1, dtype=torch.long)
    mask = torch.zeros((len(batch), length), dtype=torch.bool)
    for i, sample in enumerate(batch):
        n = sample['audio_tokens'].shape[1]
        audio[i, :, :n], text[i, :n], mask[i, :n] = sample['audio_tokens'], sample['text_tokens'], True
    return {'audio_tokens': audio, 'text_tokens': text, 'frame_mask': mask,
            'ids': [s['id'] for s in batch], 'durations': [s['duration'] for s in batch]}
