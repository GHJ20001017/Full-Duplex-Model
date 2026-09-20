"""Token-level Qwen3 Temporal + causal, text-conditioned Depth architecture.

Mimi is an external frozen codec. Audio has shape [B, K, T], text [B, T].
-1 is an absent-stream sentinel (zero embedding), never a codec token.
"""
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Qwen3MoshiConfig:
    vocab_size: int
    hidden_size: int
    codebook_sizes: tuple[int, ...]
    text_pad_token_id: int
    depth_hidden_size: int = 1024
    depth_layers: int = 6
    depth_heads: int = 16
    depth_ffn_size: int = 4096
    acoustic_delay: int = 2

    def __post_init__(self):
        self.codebook_sizes = tuple(self.codebook_sizes)
        if len(self.codebook_sizes) != 8 or min(self.codebook_sizes) <= 0:
            raise ValueError("Expected eight positive Mimi cardinalities")
        if self.acoustic_delay not in (1, 2):
            raise ValueError("Use acoustic delay 2 for pre-training or 1 for later stages")
        if min(self.hidden_size, self.vocab_size, self.depth_hidden_size,
               self.depth_heads, self.depth_layers, self.depth_ffn_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.depth_hidden_size % self.depth_heads:
            raise ValueError("Depth hidden size must be divisible by head count")
        if not 0 <= self.text_pad_token_id < self.vocab_size:
            raise ValueError("Text padding token must belong to Qwen vocabulary")

    @property
    def num_codebooks(self):
        return len(self.codebook_sizes)


def apply_delay(tokens: Tensor, delay: int, pad_value: int = -1) -> Tensor:
    """Keep semantic codebook at t; move acoustic codebooks to t+delay.

    Append flush frames rather than discarding the utterance tail.
    """
    if tokens.ndim != 3 or delay < 0:
        raise ValueError("Expected [B,K,T] and non-negative delay")
    out = tokens.new_full((*tokens.shape[:2], tokens.shape[-1] + delay), pad_value)
    out[:, 0, :tokens.shape[-1]] = tokens[:, 0]
    out[:, 1:, delay:] = tokens[:, 1:]
    return out


def remove_delay(tokens: Tensor, delay: int) -> Tensor:
    """Invert apply_delay, including the retained tail frames."""
    if tokens.ndim != 3 or delay < 0 or delay > tokens.shape[-1]:
        raise ValueError("Invalid delayed sequence or delay")
    length = tokens.shape[-1] - delay
    return torch.cat((tokens[:, :1, :length], tokens[:, 1:, delay:]), dim=1)


def shift_history(tokens: Tensor) -> Tensor:
    """One Temporal step of teacher-forcing shift, independent of codec delay."""
    return torch.cat((tokens.new_full((*tokens.shape[:-1], 1), -1), tokens), -1)[..., :tokens.shape[-1]]


def align_timestamped_tokens(token_ids, timestamps, num_frames, pad_token_id):
    """Place pre-tokenized text at 12.5 Hz; reject overflow instead of truncating.

    Collisions spill into the next free frame. Caller supplies token timestamps,
    not word timestamps; no implicit tokenizer or ASR is invoked.
    """
    import math
    if len(token_ids) != len(timestamps) or num_frames < 0:
        raise ValueError("Invalid alignment sizes")
    out = torch.full((num_frames,), pad_token_id, dtype=torch.long)
    last_frame, last_time = -1, -1.0
    for token, timestamp in zip(token_ids, timestamps):
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp < last_time:
            raise ValueError("Timestamps must be finite, non-negative and ordered")
        frame = max(math.floor(timestamp * 12.5), last_frame + 1)
        if frame >= num_frames:
            raise ValueError("Text does not fit the audio timeline")
        out[frame] = token
        last_frame, last_time = frame, timestamp
    return out


class DepthTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            config.depth_hidden_size, config.depth_heads, config.depth_ffn_size,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.depth_layers, enable_nested_tensor=False)
        # TransformerEncoder clones equal initial weights; initialize layers independently.
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.positions = nn.Embedding(config.num_codebooks, config.depth_hidden_size)
        self.norm = nn.LayerNorm(config.depth_hidden_size)

    def forward(self, inputs):
        k = inputs.shape[1]
        mask = torch.ones(k, k, dtype=torch.bool, device=inputs.device).triu(1)
        return self.norm(self.transformer(inputs + self.positions.weight[:k], mask=mask))


class Qwen3Moshi(nn.Module):
    def __init__(self, qwen_model: nn.Module, config: Qwen3MoshiConfig):
        super().__init__()
        if qwen_model.config.model_type != "qwen3":
            raise ValueError("Expected a Qwen3 causal language model")
        if (qwen_model.config.hidden_size, qwen_model.config.vocab_size) != (config.hidden_size, config.vocab_size):
            raise ValueError("Qwen3 configuration mismatch")
        self.config = config
        self.qwen_model = qwen_model
        self.user_audio_embeddings = nn.ModuleList(nn.Embedding(n, config.hidden_size) for n in config.codebook_sizes)
        self.assistant_audio_embeddings = nn.ModuleList(nn.Embedding(n, config.hidden_size) for n in config.codebook_sizes)
        self.temporal_to_depth = nn.Linear(config.hidden_size, config.depth_hidden_size)
        self.depth_text_embedding = nn.Embedding(config.vocab_size, config.depth_hidden_size)
        self.depth_audio_embeddings = nn.ModuleList(nn.Embedding(n, config.depth_hidden_size) for n in config.codebook_sizes[:-1])
        self.depth = DepthTransformer(config)
        self.audio_heads = nn.ModuleList(nn.Linear(config.depth_hidden_size, n) for n in config.codebook_sizes)
        self._temporal_cache: Any = None
        self.to(device=self.text_embedding.weight.device, dtype=self.text_embedding.weight.dtype)

    @property
    def text_embedding(self):
        return self.qwen_model.get_input_embeddings()

    @property
    def text_head(self):
        return self.qwen_model.get_output_embeddings()

    @staticmethod
    def _embed(table, tokens):
        if tokens.dtype != torch.long or torch.any(tokens < -1) or torch.any(tokens >= table.num_embeddings):
            raise ValueError("Tokens must be int64 and in [-1, cardinality)")
        return table(tokens.clamp_min(0)) * (tokens != -1).unsqueeze(-1)

    def temporal(self, user_history, assistant_history, text_history, *, attention_mask=None, use_cache=False):
        """Low-level already-delayed, already-shifted inputs; cache is session-local.

        During cached inference pass only new time steps and a full-prefix mask.
        Use a separate model/session per concurrent conversation.
        """
        expected = (text_history.shape[0], self.config.num_codebooks, text_history.shape[-1])
        if text_history.ndim != 2 or user_history.shape != expected or assistant_history.shape != expected:
            raise ValueError("Expected matching [B,8,T] audio and [B,T] text")
        inputs = self._embed(self.text_embedding, text_history)
        for k in range(self.config.num_codebooks):
            inputs = inputs + self._embed(self.user_audio_embeddings[k], user_history[:, k])
            inputs = inputs + self._embed(self.assistant_audio_embeddings[k], assistant_history[:, k])
        result = self.qwen_model.model(
            inputs_embeds=inputs, attention_mask=attention_mask,
            past_key_values=self._temporal_cache if use_cache else None,
            use_cache=use_cache, return_dict=True,
        )
        if use_cache:
            self._temporal_cache = result.past_key_values
        return result.last_hidden_state

    def depth_logits(self, hidden, text_targets, audio_targets):
        """Teacher forcing: codebook k sees W_t and A_{t,<k}, never A_{t,>=k}."""
        b, t, _ = hidden.shape
        base = self.temporal_to_depth(hidden)
        inputs = [base + self._embed(self.depth_text_embedding, text_targets)]
        for k, table in enumerate(self.depth_audio_embeddings):
            inputs.append(base + self._embed(table, audio_targets[:, k]))
        states = self.depth(torch.stack(inputs, 2).reshape(b * t, self.config.num_codebooks, -1))
        return tuple(head(states[:, k]).reshape(b, t, -1) for k, head in enumerate(self.audio_heads))

    def prepare_batch(self, user, assistant, text, frame_mask=None):
        if text.ndim != 2 or text.shape[-1] == 0:
            raise ValueError("Expected nonempty [B,T] text")
        if user.shape != assistant.shape or user.shape != (text.shape[0], 8, text.shape[1]):
            raise ValueError("Stream shapes do not match")
        valid = torch.ones_like(text, dtype=torch.bool) if frame_mask is None else frame_mask.bool()
        if valid.shape != text.shape or torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("frame_mask must be right-padded [B,T]")
        d = self.config.acoustic_delay
        user = apply_delay(user.masked_fill(~valid[:, None], -1), d)
        audio = apply_delay(assistant.masked_fill(~valid[:, None], -1), d)
        text = F.pad(text.masked_fill(~valid, -1), (0, d), value=-1)
        audio_mask = audio != -1
        text_mask = text != -1
        attention = audio_mask.any(1) | (user != -1).any(1) | text_mask
        return dict(user=user, audio=audio, text=text, audio_mask=audio_mask,
                    text_mask=text_mask, attention_mask=attention)

    def forward(self, user_audio_tokens, assistant_audio_tokens, text_tokens, *, frame_mask=None):
        batch = self.prepare_batch(user_audio_tokens, assistant_audio_tokens, text_tokens, frame_mask)
        hidden = self.temporal(shift_history(batch['user']), shift_history(batch['audio']),
                               shift_history(batch['text']), attention_mask=batch['attention_mask'])
        return dict(text_logits=self.text_head(hidden),
                    audio_logits=self.depth_logits(hidden, batch['text'], batch['audio']), batch=batch)

    @staticmethod
    def _ce(logits, targets, mask, weights=None):
        valid = mask & (targets != -1)
        safe = targets.masked_fill(~valid, 0)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), safe.reshape(-1), reduction='none').reshape_as(targets)
        values = valid.to(ce.dtype)
        if weights is not None:
            values = values * weights
        return (ce * values).sum() / valid.sum().clamp_min(1)

    def loss(self, outputs):
        batch = outputs['batch']
        weights = torch.where(batch['text'] == self.config.text_pad_token_id, 0.5, 1.0)
        text = self._ce(outputs['text_logits'], batch['text'], batch['text_mask'], weights)
        codebooks = torch.stack([self._ce(logits, batch['audio'][:, k], batch['audio_mask'][:, k])
                                for k, logits in enumerate(outputs['audio_logits'])])
        alpha = codebooks.new_tensor([100.] + [1.] * 7)
        audio = (codebooks * alpha).sum() / alpha.sum()
        return dict(loss=text + audio, text_loss=text, audio_loss=audio, audio_codebook_loss=codebooks)

    def forward_text(self, input_ids, attention_mask=None):
        """Pure-text preservation branch; caller owns its separate optimizer state."""
        return self.qwen_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.masked_fill(
            attention_mask == 0, -100) if attention_mask is not None else input_ids, use_cache=False)

    @torch.no_grad()
    def generate_step(self, user_history, assistant_history, text_history):
        """Greedy one-step sampling on delayed history; no live codec/playback loop."""
        if self.training or text_history.shape[-1] != 1:
            raise ValueError("Call eval() and supply one already-shifted delayed frame")
        hidden = self.temporal(user_history, assistant_history, text_history, use_cache=True)
        text = self.text_head(hidden).argmax(-1)
        audio = user_history.new_full(user_history.shape, -1)
        for k in range(8):
            audio[:, k] = self.depth_logits(hidden, text, audio)[k].argmax(-1)
        return text, audio

    def reset_stream_state(self):
        self._temporal_cache = None


def load_qwen3_moshi(qwen_model, *, mimi_codebook_sizes, text_pad_token_id, **depth_options):
    """Use verified codec cardinalities, never Qwen vocabulary IDs for audio."""
    config = Qwen3MoshiConfig(qwen_model.config.vocab_size, qwen_model.config.hidden_size,
                             tuple(mimi_codebook_sizes), text_pad_token_id, **depth_options)
    return Qwen3Moshi(qwen_model, config)
