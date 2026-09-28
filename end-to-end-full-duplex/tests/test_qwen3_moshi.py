import pytest
import torch
from torch import nn

from models.qwen3_moshi import (
    Qwen3Moshi,
    Qwen3MoshiConfig,
    apply_delay,
    align_timestamped_tokens,
    remove_delay,
    shift_history,
)


class _Output:
    def __init__(self, hidden):
        self.last_hidden_state = hidden
        self.past_key_values = None


class _TinyCore(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds=None, **_kwargs):
        return _Output(self.projection(inputs_embeds))


class _TinyQwen3(nn.Module):
    def __init__(self, vocab_size=31, hidden_size=16):
        super().__init__()
        self.config = type("Config", (), {
            "model_type": "qwen3",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
        })()
        self.core = _TinyCore(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.core.embedding

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def model(self):
        return self.core


def _model(num_audio_streams=2):
    config = Qwen3MoshiConfig(
        31, 16, (7, 8, 9, 10, 11, 12, 13, 14), 30,
        depth_hidden_size=16, depth_layers=1, depth_heads=4, depth_ffn_size=32,
        num_audio_streams=num_audio_streams,
    )
    return Qwen3Moshi(_TinyQwen3(), config)


def test_delay_preserves_all_frames_and_semantic_codebook():
    tokens = torch.arange(2 * 8 * 4).reshape(2, 8, 4)
    delayed = apply_delay(tokens, 2)
    assert delayed.shape == (2, 8, 6)
    assert torch.equal(delayed[:, 0, :4], tokens[:, 0])
    assert torch.equal(delayed[:, 1:, 2:], tokens[:, 1:])
    assert torch.equal(remove_delay(delayed, 2), tokens)


def test_shift_history_is_fixed_length():
    tokens = torch.tensor([[3, 4, 5]])
    assert torch.equal(shift_history(tokens), torch.tensor([[-1, 3, 4]]))


def test_timestamp_alignment_and_rejection():
    aligned = align_timestamped_tokens([4, 5], [0.0, 0.08], 3, 30)
    assert torch.equal(aligned, torch.tensor([4, 5, 30]))


def test_model_returns_text_and_eight_audio_heads_and_loss():
    model = _model()
    user = torch.randint(0, 7, (2, 8, 4))
    assistant = torch.randint(0, 7, (2, 8, 4))
    text = torch.randint(0, 30, (2, 4))
    output = model(user, assistant, text)
    assert output["text_logits"].shape == (2, 6, 31)
    assert len(output["audio_logits"]) == 8
    assert output["audio_logits"][0].shape[:2] == (2, 6)
    losses = model.loss(output)
    assert losses["audio_codebook_loss"].shape == (8,)
    assert torch.isfinite(losses["loss"])


def test_single_stream_masks_flush_and_backward():
    model = _model(1)
    assert model.user_audio_embeddings is None
    assert not any(key.startswith('user_audio_embeddings') for key in model.state_dict())
    audio = torch.randint(0, 7, (2, 8, 4))
    text = torch.randint(0, 30, (2, 4))
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    output = model.forward_single_stream(audio, text, frame_mask=mask)
    batch = output['batch']
    assert batch['user'] is None
    assert output['text_logits'].shape == (2, 6, 31)
    restored = remove_delay(batch['audio'], 2)
    assert torch.equal(restored, audio.masked_fill(~mask[:, None], -1))
    assert batch['audio_mask'].sum().item() == 6 * 8
    assert batch['text_mask'].sum().item() == 6
    assert torch.equal(batch['audio'][0, 1:, -1], audio[0, 1:, -1])
    loss = model.loss(output)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    for table in model.assistant_audio_embeddings:
        assert table.weight.grad is not None
        assert torch.isfinite(table.weight.grad).all()
    altered = audio.clone()
    altered[1, :, 2:] = 6
    altered_text = text.clone()
    altered_text[1, 2:] = 29
    other = model.forward_single_stream(altered, altered_text, frame_mask=mask)
    torch.testing.assert_close(model.loss(other)['loss'], loss)


def test_single_stream_history_and_depth_do_not_leak_targets():
    model = _model(1).eval()
    audio = torch.zeros(1, 8, 4, dtype=torch.long)
    text = torch.zeros(1, 4, dtype=torch.long)
    original = model.forward_single_stream(audio, text)
    changed = audio.clone()
    changed[:, 0, 2] = 1
    result = model.forward_single_stream(changed, text)
    torch.testing.assert_close(original['text_logits'][:, :3], result['text_logits'][:, :3])
    torch.testing.assert_close(original['audio_logits'][0][:, :3], result['audio_logits'][0][:, :3])
    hidden = torch.randn(1, 4, 16)
    baseline = model.depth_logits(hidden, text, audio)
    for k in range(8):
        changed = audio.clone()
        changed[:, k:] = 1
        logits = model.depth_logits(hidden, text, changed)
        for j in range(k + 1):
            torch.testing.assert_close(baseline[j], logits[j])


def test_single_stream_upgrade_preserves_outputs_and_checkpoint():
    model = _model(1).eval()
    restored = _model(1).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    audio = torch.randint(0, 7, (1, 8, 4))
    text = torch.randint(0, 30, (1, 4))
    before = model.forward_single_stream(audio, text)
    torch.testing.assert_close(before['text_logits'], restored.forward_single_stream(audio, text)['text_logits'])
    with pytest.raises(ValueError, match='forward_single_stream'):
        model(audio, audio, text)
    model._temporal_cache = object()
    model.enable_dual_stream()
    assert model.config.num_audio_streams == 2
    assert model._temporal_cache is None
    for user, assistant in zip(model.user_audio_embeddings, model.assistant_audio_embeddings):
        torch.testing.assert_close(user.weight, assistant.weight)
        assert user.weight.data_ptr() != assistant.weight.data_ptr()
    after = model(torch.full_like(audio, -1), audio, text)
    torch.testing.assert_close(before['text_logits'], after['text_logits'])
    for first, second in zip(before['audio_logits'], after['audio_logits']):
        torch.testing.assert_close(first, second)
    branch = model.user_audio_embeddings
    model.enable_dual_stream()
    assert model.user_audio_embeddings is branch
    _model(2).load_state_dict(model.state_dict(), strict=True)


def test_single_stream_rejects_invalid_shapes_and_masks():
    model = _model(1)
    audio = torch.zeros(1, 8, 4, dtype=torch.long)
    text = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match='Stream shapes'):
        model.forward_single_stream(audio[:, :7], text)
    with pytest.raises(ValueError, match='right-padded'):
        model.forward_single_stream(audio, text, frame_mask=torch.tensor([[1, 0, 1, 0]]))
    with pytest.raises(ValueError, match='nonempty'):
        model.forward_single_stream(audio[:, :, :0], text[:, :0])
    with pytest.raises(ValueError, match='cardinality'):
        model.forward_single_stream(audio + 20, text)
    with pytest.raises(ValueError, match='one pre-training'):
        _model(3)


def test_reset_clears_temporal_cache():
    model = _model()
    model._temporal_cache = object()
    model.reset_stream_state()
    assert model._temporal_cache is None
