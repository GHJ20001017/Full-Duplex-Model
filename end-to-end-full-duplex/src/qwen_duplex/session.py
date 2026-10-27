"""Model-native action decoding on one append-only Thinker cache; no decision controller."""

from dataclasses import dataclass

import numpy as np
import torch

from .audio import PrefixAudioEncoder
from .sequence import audio_packet, text_vocabulary, token_embeddings
from .tokens import validate_action_tokens


@dataclass(frozen=True)
class ActionEvent:
    token: str
    token_id: int
    audio_end_ms: int
    sequence_length: int


class ThinkerSession:
    """Single-session, single-batch synchronous audio/text inference protocol.

    Call prefill() once, then push_audio()/append_audio() and decide() once per
    fixed audio window. Optionally generate_text() after speak/backchannel.
    New audio may be appended between any two generated text tokens, without
    resetting the cache. The clock here is the audio sample timeline, not a
    wall-clock scheduler; callers handle capture and scheduling, never actions.

    Decisions use Thinker's existing LM head restricted to the action vocabulary,
    not an auxiliary classifier. Untrained action rows have no learned policy.
    This class neither calls Talker nor interprets interrupt as a playback command.
    """

    def __init__(self, model, tokenizer, *, processor=None, decision_interval_ms=1000,
                 max_context_tokens=None, max_audio_seconds=30.0, record_sequence=False):
        if isinstance(decision_interval_ms, bool) or not isinstance(decision_interval_ms, int) or decision_interval_ms <= 0:
            raise ValueError("decision_interval_ms must be a positive integer")
        self.thinker = getattr(model, "thinker", model)
        self.tokenizer = tokenizer
        self.actions = validate_action_tokens(model, tokenizer)
        self.stop_ids, self.text_forbidden = text_vocabulary(self.thinker, tokenizer, self.actions)
        self.interval_ms = decision_interval_ms
        limit = self.thinker.config.text_config.max_position_embeddings
        if max_context_tokens is not None:
            if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int) or not 0 < max_context_tokens <= limit:
                raise ValueError("Invalid context budget")
            limit = max_context_tokens
        self.max_context_tokens = limit
        self.encoder = PrefixAudioEncoder(model, processor, max_audio_seconds=max_audio_seconds) if processor else None
        self.record_sequence = record_sequence
        self.reset()

    def reset(self):
        self.cache = None
        self.length = 0
        self.logits = None
        self.audio_end_ms = 0
        self.pending_action = False
        self.last_action = None
        self.text_finished = False
        self.broken = False
        self.events = []
        self._parts = []
        if self.encoder is not None:
            self.encoder.reset()

    def _ready(self):
        if self.broken:
            raise RuntimeError("A forward failed and may have mutated KV state; reset() is required")
        if self.thinker.training:
            raise ValueError("Call model.eval() before cached inference")

    def _capacity(self, count):
        if self.length + count > self.max_context_tokens:
            raise ValueError("Context budget exceeded; rebuild/reset explicitly, do not silently trim KV")

    @torch.inference_mode()
    def _append(self, embeddings, kind):
        self._ready()
        count = embeddings.shape[1]
        self._capacity(count)
        start = self.length
        position = torch.arange(start, start + count, device=embeddings.device)
        try:
            output = self.thinker(
                inputs_embeds=embeddings, past_key_values=self.cache,
                position_ids=position.view(1, 1, -1).expand(3, 1, -1), cache_position=position,
                attention_mask=torch.ones((1, start + count), device=embeddings.device, dtype=torch.long),
                use_cache=True, return_dict=True,
            )
            if output.past_key_values is None or output.past_key_values.get_seq_length() != start + count:
                raise RuntimeError("Thinker returned inconsistent KV length")
            if self.cache is not None and output.past_key_values is not self.cache:
                raise RuntimeError("Thinker unexpectedly replaced its cache object")
            logits = output.logits[:, -1, :].detach()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Thinker returned non-finite logits")
        except Exception:
            # DynamicCache may be partially updated before a native forward raises.
            self.broken = True
            raise
        self.cache = output.past_key_values
        self.length += count
        self.logits = logits
        self.events.append({"kind": kind, "start": start, "end": self.length})
        if self.record_sequence:
            self._parts.append(embeddings.detach().clone())

    def prefill(self, prefix_ids):
        """Append a text-only chat/system prefix once; no audio placeholder expansion here."""
        self._ready()
        if self.length:
            raise ValueError("prefill() is only allowed on an empty session")
        self._append(token_embeddings(self.thinker, prefix_ids), "prefill")

    def append_audio(self, new_embeddings, *, end_ms):
        """Append ONLY newly committed audio positions for the next fixed window."""
        self._ready()
        if not self.length:
            raise ValueError("prefill() must precede audio")
        if self.pending_action:
            raise ValueError("Generate the pending action before appending another window")
        if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms != self.audio_end_ms + self.interval_ms:
            raise ValueError("Audio windows must be contiguous and exactly one decision interval")
        packet = audio_packet(self.thinker, new_embeddings)
        self._capacity(packet.shape[1] + 1)  # Reserve the mandatory action position.
        self._append(packet, "audio")
        self.audio_end_ms = end_ms
        self.pending_action = True
        self.text_finished = False

    def push_audio(self, pcm, *, sampling_rate):
        """Encode exactly one arrived window at the native rate, then append its new positions."""
        self._ready()
        if self.encoder is None:
            raise ValueError("Pass processor=... to enable waveform input")
        if not self.length or self.pending_action:
            raise ValueError("prefill() and resolve the previous decision before push_audio()")
        expected = sampling_rate * self.interval_ms
        if sampling_rate != self.encoder.sample_rate or expected % 1000 or np.asarray(pcm).shape != (expected // 1000,):
            raise ValueError("pcm must be exactly one interval of mono audio at the native sampling rate")
        # Reserve before changing encoder state; packet size is checked again after encoding.
        self._capacity(4)
        hidden = self.encoder.push(pcm, sampling_rate=sampling_rate)
        try:
            self.append_audio(hidden, end_ms=self.audio_end_ms + self.interval_ms)
        except Exception:
            self.broken = True
            raise

    def decide(self):
        """Choose and write back one action using the native Thinker LM-head logits."""
        self._ready()
        if not self.pending_action:
            raise ValueError("No newly appended audio window needs a decision")
        choices = torch.tensor(list(self.actions.values()), device=self.logits.device)
        index = self.logits[0, choices].argmax().item()
        token = list(self.actions)[index]
        token_id = self.actions[token]
        self._append(token_embeddings(self.thinker, [token_id]), "action")
        self.pending_action = False
        self.last_action = token
        self.text_finished = token not in ("<|speak|>", "<|backchannel|>")
        return ActionEvent(token, token_id, self.audio_end_ms, self.length)

    def generate_text(self, *, max_new_tokens=1):
        """Decode a bounded text chunk; yield control so the next audio can be appended.

        Only end_speech is permitted as an action inside text. Native EOS ends
        this text burst, NOT the persistent session. Termination tokens are
        returned to the caller and written to KV just like other outputs.
        """
        self._ready()
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.pending_action or self.last_action not in ("<|speak|>", "<|backchannel|>"):
            raise ValueError("Text requires a completed speak/backchannel decision")
        if self.text_finished:
            return []
        self._capacity(max_new_tokens)
        result = []
        for _ in range(max_new_tokens):
            logits = self.logits.clone()
            if self.text_forbidden:
                logits[:, list(self.text_forbidden)] = -torch.inf
            if not torch.isfinite(logits).any():
                raise RuntimeError("No valid text token remains after vocabulary masking")
            token_id = int(logits.argmax(-1).item())
            self._append(token_embeddings(self.thinker, [token_id]), "text")
            result.append(token_id)
            if token_id in self.stop_ids:
                self.text_finished = True
                break
        return result

    def replay_inputs(self):
        """Return exact committed embeddings for cache-replay diagnostics, not offline audio equivalence."""
        self._ready()
        if not self.record_sequence or not self._parts:
            raise ValueError("Set record_sequence=True and prefill before replay")
        embeddings = torch.cat(self._parts, dim=1)
        return {
            "inputs_embeds": embeddings,
            "position_ids": torch.arange(self.length, device=embeddings.device).view(1, 1, -1).expand(3, 1, -1),
            "attention_mask": torch.ones((1, self.length), device=embeddings.device, dtype=torch.long),
            "use_cache": False, "return_dict": True,
        }
