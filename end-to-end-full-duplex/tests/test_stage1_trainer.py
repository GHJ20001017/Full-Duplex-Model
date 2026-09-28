import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.qwen3_moshi import Qwen3Moshi, Qwen3MoshiConfig
from trainer.codec import FrozenMimiCodec
from trainer.engine import TrainConfig, evaluate, load_checkpoint, seed_everything, train


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='qwen3', hidden_size=8, vocab_size=13)
        self.embedding = nn.Embedding(13, 8)
        self.head = nn.Linear(8, 13)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.2)

    @property
    def model(self):
        return self

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(inputs_embeds)))


def make_model():
    seed_everything(12)
    return Qwen3Moshi(TinyQwen(), Qwen3MoshiConfig(13, 8, (7,) * 8, 0,
                      depth_hidden_size=8, depth_layers=1, depth_heads=2,
                      depth_ffn_size=16, num_audio_streams=1))


class TinyData:
    fingerprint = 'deterministic-tiny-v1'
    def __init__(self, count=3):
        self.count = count
        self.records = list(range(count))

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        length = i + 2
        return dict(audio_tokens=torch.arange(8 * length).reshape(8, length) % 7,
                    text_tokens=torch.arange(length) % 13, id=str(i), duration=length / 12.5)


def collate(samples):
    # Exercise the actual dataset collation contract without loading any assets.
    from dataset import collate_emilia
    return collate_emilia(samples)


@pytest.fixture(autouse=True)
def fast_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('dropout', [0., .3, 1.])
def test_one_step_and_resume_matches_uninterrupted(tmp_path, dropout):
    config = TrainConfig(max_steps=3, batch_size=2, accumulation_steps=2, validation_interval=1,
                         text_condition_dropout=dropout)
    data = TinyData()
    uninterrupted = make_model()
    train(uninterrupted, data, collate, tmp_path / 'whole', config, validation_data=data)
    interrupted = make_model()
    first = train(interrupted, data, collate, tmp_path / 'resume', config,
                  validation_data=data, stop_after_steps=1)
    assert first['global_step'] == 1
    saved = load_checkpoint(tmp_path / 'resume' / 'checkpoint.pt')
    assert saved['global_step'] == 1 and saved['batch_position'] == 3
    resumed = make_model()
    train(resumed, data, collate, tmp_path / 'resume', config, validation_data=data, resume=True)
    for key, tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
    whole = load_checkpoint(tmp_path / 'whole' / 'checkpoint.pt')
    final = load_checkpoint(tmp_path / 'resume' / 'checkpoint.pt')
    assert final['scheduler'] == whole['scheduler']
    for key in final['optimizer']['state']:
        for name, value in final['optimizer']['state'][key].items():
            torch.testing.assert_close(value, whole['optimizer']['state'][key][name], rtol=0, atol=0)
    assert (tmp_path / 'whole' / 'metrics.jsonl').read_text() == (tmp_path / 'resume' / 'metrics.jsonl').read_text()


def test_max_steps_one_padding_validation_and_logs(tmp_path):
    model, data = make_model(), TinyData(2)
    config = TrainConfig(max_steps=1, batch_size=2, validation_interval=1)
    batch = collate([data[0], data[1]])
    assert batch['frame_mask'].sum() == 5
    assert batch['frame_mask'].numel() == 6
    model.eval()
    original = model.loss(model.forward_single_stream(batch['audio_tokens'], batch['text_tokens'],
                                                       frame_mask=batch['frame_mask']))['loss']
    batch['audio_tokens'][0, :, -1] = 6
    batch['text_tokens'][0, -1] = 12
    changed = model.loss(model.forward_single_stream(batch['audio_tokens'], batch['text_tokens'],
                                                      frame_mask=batch['frame_mask']))['loss']
    torch.testing.assert_close(original, changed)
    model.train()
    rng = torch.get_rng_state().clone()
    a = evaluate(model, data, collate, config)
    b = evaluate(model, data, collate, config)
    assert model.training and a == b and torch.equal(rng, torch.get_rng_state())
    result = train(model, data, collate, tmp_path, config, validation_data=data)
    assert result['global_step'] == 1
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in rows] == ['train', 'validation']
    assert rows[0]['pad_ratio'] == pytest.approx(1 / 6)
    for row in rows:
        assert row['valid_frames'] == 5
        assert row['text_pad_ratio'] == pytest.approx(2 / 5)
    assert rows[0]['audio_seconds'] == pytest.approx(5 / 12.5)
    assert 'audio_codebook_7_loss' in rows[0]


def test_failures_are_explicit(tmp_path):
    data, config = TinyData(), TrainConfig(max_steps=1, validation_interval=0)
    with pytest.raises(ValueError, match='empty'):
        train(make_model(), TinyData(0), collate, tmp_path / 'empty', config)
    model = make_model()
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        train(model, data, collate, tmp_path / 'nan', config)
    assert not (tmp_path / 'nan' / 'checkpoint.pt').exists()
    train(make_model(), data, collate, tmp_path / 'ok', config)
    with pytest.raises(FileExistsError):
        train(make_model(), data, collate, tmp_path / 'ok', config)
    with pytest.raises(ValueError, match='fingerprint'):
        train(make_model(), data, collate, tmp_path / 'ok', replace(config, batch_size=2), resume=True)
    with pytest.raises(ValueError, match='CPU'):
        replace(config, precision='bf16').validate()


class FakeMimi(nn.Module):
    sample_rate = 24000
    frame_rate = 12.5
    cardinality = 17
    num_codebooks = 8
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def encode(self, wave):
        assert not self.training and not torch.is_grad_enabled()
        assert wave.shape == (1, 1, 3840)
        return torch.full((1, 8, 2), 16, dtype=torch.long)


def test_codec_runtime_cardinality_freezing_identity(tmp_path):
    checkpoint = tmp_path / 'mimi.safetensors'
    checkpoint.write_bytes(b'fake local bytes')
    model = FakeMimi()
    codec = FrozenMimiCodec(checkpoint, loader=lambda *a, **k: model)
    assert codec.codebook_sizes == (17,) * 8
    assert not model.weight.requires_grad
    codes = codec(torch.zeros(3840))
    assert codes.shape == (8, 2) and codes.device.type == 'cpu' and not codes.is_inference()
    for invalid in (torch.zeros(1, 1, 3840), torch.empty(0), torch.tensor([float('nan')])):
        with pytest.raises(ValueError, match='waveform'):
            codec(invalid)
    previous = codec.cache_identity
    checkpoint.write_bytes(b'different bytes')
    assert FrozenMimiCodec(checkpoint, loader=lambda *a, **k: model).cache_identity != previous
    model.cardinality = None
    with pytest.raises(ValueError, match='cardinality'):
        FrozenMimiCodec(checkpoint, loader=lambda *a, **k: model)


def test_cli_requires_explicit_padding():
    from trainer.train_stage1 import parser
    with pytest.raises(SystemExit):
        parser().parse_args(['--manifest', 'm', '--qwen-model', 'q', '--mimi-checkpoint', 'c', '--output-dir', 'o'])


def test_cli_one_step_with_injected_local_assets(tmp_path, monkeypatch):
    import os
    import dataset
    import transformers
    import trainer.codec
    from trainer.train_stage1 import main

    class Tokenizer:
        is_fast = True
        def get_vocab(self):
            return {'pad': 0}
        def __call__(self, text, **kwargs):
            return {'offset_mapping': [(0, len(text))]}

    qwen_dir = tmp_path / 'local-qwen'
    qwen_dir.mkdir()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}\n')
    checkpoint = tmp_path / 'codec.safetensors'
    checkpoint.write_bytes(b'local test only')
    calls = []
    def local_factory(value):
        def load(path, **kwargs):
            assert kwargs['local_files_only'] is True
            assert kwargs['trust_remote_code'] is False
            assert Path(path).is_dir()
            calls.append(path)
            return value
        return load
    from pathlib import Path
    qwen = TinyQwen()
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', local_factory(qwen.config))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', local_factory(Tokenizer()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', local_factory(qwen))
    monkeypatch.setattr(trainer.codec, 'FrozenMimiCodec', lambda *a, **k: SimpleNamespace(
        codebook_sizes=(7,) * 8, cache_identity='fake-local-codec'))
    splits = []
    def data_factory(*args, **kwargs):
        splits.append(kwargs['split'])
        return TinyData(2)
    monkeypatch.setattr(dataset, 'EmiliaDataset', data_factory)
    args = ['--manifest', str(manifest), '--qwen-model', str(qwen_dir),
            '--mimi-checkpoint', str(checkpoint), '--output-dir', str(tmp_path / 'out'),
            '--text-pad-token-id', '0', '--max-steps', '1', '--validation-interval', '1',
            '--depth-hidden-size', '8', '--depth-layers', '1', '--depth-heads', '2', '--depth-ffn-size', '16']
    assert main(args) == 0
    assert len(calls) == 3 and splits == ['train', 'val']
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    assert load_checkpoint(tmp_path / 'out' / 'checkpoint.pt')['global_step'] == 1


@pytest.mark.parametrize('dropout', [-.1, 1.1, float('nan'), float('inf')])
def test_dropout_validation(dropout):
    with pytest.raises(ValueError, match='text_condition_dropout'):
        TrainConfig(text_condition_dropout=dropout).validate()


def test_conditioning_dropout_is_per_utterance_and_eval_is_unmasked(monkeypatch):
    from trainer.engine import _loss
    model = make_model()
    data = TinyData(2)
    batch = collate([data[0], data[1]])
    targets = batch['text_tokens'].clone()
    masks = []
    original = model.forward_single_stream
    def observe(audio, text, **kwargs):
        torch.testing.assert_close(text, targets)
        masks.append(kwargs['text_condition_mask'])
        return original(audio, text, **kwargs)
    monkeypatch.setattr(model, 'forward_single_stream', observe)
    draws = []
    def draw(*shape, **kwargs):
        draws.append(shape)
        assert shape == (2, 1)
        return torch.tensor([[.2], [.8]], device=kwargs['device'])
    monkeypatch.setattr(torch, 'rand', draw)
    config = TrainConfig(text_condition_dropout=.3)
    model.train()
    _loss(model, batch, config)
    assert len(draws) == 1
    assert masks[0].dtype == torch.bool and masks[0].shape == targets.shape
    assert not masks[0][0].any() and masks[0][1].all()
    model.eval()
    _loss(model, batch, config)
    assert len(draws) == 1 and masks[1] is None
    torch.testing.assert_close(batch['text_tokens'], targets)


@pytest.mark.parametrize('suffix', [b'{"kind":', b'{"kind":"\xe4\xb8'])
def test_resume_repairs_only_incomplete_final_metrics_line(tmp_path, suffix):
    config, data = TrainConfig(max_steps=2, validation_interval=0), TinyData()
    train(make_model(), data, collate, tmp_path, config, stop_after_steps=1)
    path = tmp_path / 'metrics.jsonl'
    committed = path.read_bytes()
    path.write_bytes(committed + suffix)
    train(make_model(), data, collate, tmp_path, config, resume=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['global_step'] for row in rows] == [1, 2]
    # An earlier malformed row or a newline-terminated malformed last row is
    # corruption, not a partial append. The log must remain untouched on failure.
    for corrupt in (b'not-json\n' + committed, committed + b'not-json\n'):
        path.write_bytes(corrupt)
        with pytest.raises(json.JSONDecodeError):
            train(make_model(), data, collate, tmp_path, config, resume=True)
        assert path.read_bytes() == corrupt


def test_permutation_is_cached_across_updates_and_rebuilt_on_resume(tmp_path, monkeypatch):
    original = torch.randperm
    calls = []
    def record(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'randperm', record)
    config = TrainConfig(max_steps=5, batch_size=1, validation_interval=0)
    data = TinyData(3)
    train(make_model(), data, collate, tmp_path / 'whole', config)
    assert calls == [3, 3]  # Five updates span two epochs, not five permutations.
    calls.clear()
    train(make_model(), data, collate, tmp_path / 'resume', config, stop_after_steps=1)
    train(make_model(), data, collate, tmp_path / 'resume', config, resume=True)
    assert calls == [3, 3, 3]  # Once initially, once on resume, once at epoch change.
    whole = load_checkpoint(tmp_path / 'whole' / 'checkpoint.pt')
    resumed = load_checkpoint(tmp_path / 'resume' / 'checkpoint.pt')
    for key, value in whole['model'].items():
        torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
