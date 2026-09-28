import pytest
import torch

from test_qwen3_moshi import _model


def test_text_condition_dropout_keeps_supervision_and_hides_both_paths():
    model = _model(1).eval()
    audio = torch.zeros(1, 8, 4, dtype=torch.long)
    text = torch.tensor([[1, 2, 3, 30]])
    keep = torch.zeros_like(text, dtype=torch.bool)
    first = model.forward_single_stream(audio, text, text_condition_mask=keep)
    second = model.forward_single_stream(audio, text + (text != 30).long(), text_condition_mask=keep)
    assert torch.equal(first['batch']['text'][:, :4], text)
    assert first['batch']['text_mask'].sum() == 4
    assert (first['batch']['text_condition'] == -1).all()
    torch.testing.assert_close(first['text_logits'], second['text_logits'])
    for left, right in zip(first['audio_logits'], second['audio_logits']):
        torch.testing.assert_close(left, right)
    model.loss(first)['loss'].backward()
    assert model.text_head.weight.grad is not None


def test_short_audio_retains_shifted_history_attention_and_generation():
    model = _model(1).eval()
    audio = torch.zeros(1, 8, 1, dtype=torch.long)
    text = torch.zeros(1, 1, dtype=torch.long)
    output = model.forward_single_stream(audio, text)
    assert output['batch']['attention_mask'].tolist() == [[True, True, True]]
    generated_text, generated_audio = model.generate_step(None, audio - 1, text - 1)
    assert generated_text.shape == (1, 1)
    assert generated_audio.shape == (1, 8, 1)


def test_real_qwen_short_history_and_cached_steps_agree():
    transformers = pytest.importorskip('transformers')
    from models.qwen3_moshi import load_qwen3_moshi, shift_history

    torch.manual_seed(7)
    qwen = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=31, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    ))
    model = load_qwen3_moshi(qwen, mimi_codebook_sizes=(7,) * 8,
                            text_pad_token_id=30, num_audio_streams=1,
                            depth_hidden_size=16, depth_layers=1,
                            depth_heads=4, depth_ffn_size=32).eval()
    batch = model.prepare_batch(None, torch.zeros(1, 8, 1, dtype=torch.long),
                                torch.zeros(1, 1, dtype=torch.long))
    audio, text = shift_history(batch['audio']), shift_history(batch['text'])
    with torch.no_grad():
        full = model.temporal(None, audio, text, attention_mask=batch['attention_mask'])
        steps = [model.temporal(None, audio[:, :, i:i+1], text[:, i:i+1],
                                attention_mask=batch['attention_mask'][:, :i+1], use_cache=True)
                 for i in range(3)]
        torch.testing.assert_close(full, torch.cat(steps, dim=1), atol=1e-5, rtol=1e-4)
        changed_audio = audio.clone()
        changed_audio[:, 0, 1] = 1
        changed = model.temporal(None, changed_audio, text, attention_mask=batch['attention_mask'])
        assert not torch.allclose(full[:, 2], changed[:, 2])
    model.reset_stream_state()


def test_keep_all_text_condition_is_equivalent_and_shape_checked():
    model = _model(1).eval()
    audio = torch.zeros(1, 8, 4, dtype=torch.long)
    text = torch.zeros(1, 4, dtype=torch.long)
    first = model.forward_single_stream(audio, text)
    second = model.forward_single_stream(audio, text, text_condition_mask=torch.ones_like(text, dtype=torch.bool))
    torch.testing.assert_close(first['text_logits'], second['text_logits'])
    for left, right in zip(first['audio_logits'], second['audio_logits']):
        torch.testing.assert_close(left, right)
    with pytest.raises(ValueError, match='boolean'):
        model.forward_single_stream(audio, text, text_condition_mask=text)
