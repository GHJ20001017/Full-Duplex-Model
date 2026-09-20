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


def _model():
    config = Qwen3MoshiConfig(
        31, 16, (7, 8, 9, 10, 11, 12, 13, 14), 30,
        depth_hidden_size=16, depth_layers=1, depth_heads=4, depth_ffn_size=32,
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


def test_reset_clears_temporal_cache():
    model = _model()
    model._temporal_cache = object()
    model.reset_stream_state()
    assert model._temporal_cache is None
