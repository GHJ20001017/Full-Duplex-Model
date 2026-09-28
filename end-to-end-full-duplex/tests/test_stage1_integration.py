"""Exercise real audio I/O, dataset, codec adapter and tiny Qwen3 together."""
import hashlib
import json

import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from dataset import EmiliaDataset, collate_emilia
from models.qwen3_moshi import load_qwen3_moshi
from trainer.codec import FrozenMimiCodec
from trainer.engine import TrainConfig, load_checkpoint, seed_everything, train


class LocalTokenizer:
    cache_identity = 'integration-tokenizer-v1'

    def __call__(self, text, **kwargs):
        assert text == 'hi there'
        return {'input_ids': [3, 4], 'offset_mapping': [(0, 2), (3, 8)]}


class LocalMimi(nn.Module):
    sample_rate = 24000
    frame_rate = 12.5
    cardinality = 7
    num_codebooks = 8

    def encode(self, waveform):
        assert waveform.shape == (1, 1, 15360)
        assert not self.training and not torch.is_grad_enabled()
        return torch.arange(64).reshape(1, 8, 8) % self.cardinality


def test_emilia_codec_real_qwen_train_and_resume(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path / 'emilia-en-100h'
        extracted = root / 'extracted'
        extracted.mkdir(parents=True)
        identifier = 'EN_Y000001_S00001_W000001'
        metadata = extracted / (identifier + '.json')
        metadata.write_text(json.dumps({'id': identifier, 'text': 'hi there'}))
        audio = metadata.with_suffix('.wav')
        sf.write(audio, np.zeros((10240, 2), dtype=np.float32), 16000)
        row = {'config': {'schema': 2, 'datasets': [str(root)]},
               'metadata': str(metadata), 'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
               'id': identifier, 'audio': str(audio), 'text': 'hi there', 'duration': .64,
               'status': 'generated', 'quality_status': 'not_evaluated', 'timestamp_unit': 'seconds',
               'timestamps': [{'text': 'hi', 'start': 0., 'end': .16},
                              {'text': 'there', 'start': .24, 'end': .60}]}
        manifest = tmp_path / 'manifest.jsonl'
        manifest.write_text(json.dumps(row) + '\n')
        checkpoint = tmp_path / 'mimi.safetensors'
        checkpoint.write_bytes(b'injected codec test fixture, not real weights')
        codec = FrozenMimiCodec(checkpoint, loader=lambda *a, **k: LocalMimi())
        data = EmiliaDataset(manifest, LocalTokenizer(), codec, 0,
                             validation_fraction=0, cache_dir=tmp_path / 'cache')
        sample = data[0]
        assert sample['audio_tokens'].shape == (8, 8)
        assert sample['text_tokens'][[0, 3]].tolist() == [3, 4]

        def model():
            seed_everything(7)
            qwen = Qwen3ForCausalLM(Qwen3Config(vocab_size=13, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, attention_dropout=.1))
            qwen.config.use_cache = False
            return load_qwen3_moshi(qwen, mimi_codebook_sizes=codec.codebook_sizes,
                text_pad_token_id=0, num_audio_streams=1, acoustic_delay=2,
                depth_hidden_size=8, depth_layers=1, depth_heads=2, depth_ffn_size=16)

        config = TrainConfig(max_steps=2, validation_interval=0)
        complete, partial = model(), model()
        train(complete, data, collate_emilia, tmp_path / 'complete', config)
        train(partial, data, collate_emilia, tmp_path / 'resumed', config, stop_after_steps=1)
        restored = model()
        result = train(restored, data, collate_emilia, tmp_path / 'resumed', config, resume=True)
        assert result['global_step'] == 2
        for name, value in complete.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        assert load_checkpoint(tmp_path / 'resumed' / 'checkpoint.pt')['global_step'] == 2
    finally:
        torch.set_num_threads(previous)
