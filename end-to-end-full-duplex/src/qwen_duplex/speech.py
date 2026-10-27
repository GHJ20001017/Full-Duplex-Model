"""Differentiable Thinker-to-Talker conditioning and supervised speech-code loss.

This is a training interface, not a waveform tokenizer or a playback runtime.
Targets must come from a speech tokenizer compatible with the pretrained Talker;
Thinker input-audio features and arbitrary codec IDs are not interchangeable.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F

from .sequence import _prepare_interleaved_inputs, text_vocabulary, token_embeddings
from .tokens import validate_action_tokens


@dataclass(frozen=True)
class SpeechTarget:
    """One complete speaking window paired with raw output codes, without specials.

    ``window_index`` indexes the supplied TrainingWindow list. Its text must end
    in EOS/end_speech and contain at least one content token. A reply continued
    across windows must first be explicitly segmented/aligned by the caller;
    this interface does not invent word-to-audio timestamps or truncate codes.
    """

    window_index: int
    codec_ids: tuple[int, ...] | torch.Tensor


@dataclass
class SpeechTrainingOutput:
    loss: torch.Tensor
    thinker_loss: torch.Tensor
    speech_loss: torch.Tensor
    thinker_output: Any
    talker_outputs: tuple[Any, ...]


def _loss_weight(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _codec_targets(talker, value):
    weight = talker.get_input_embeddings().weight
    try:
        ids = torch.as_tensor(value, device=weight.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("codec_ids must be a nonempty integer [length] sequence") from error
    if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("codec_ids must be a nonempty integer [length] sequence")
    if ((ids < 0) | (ids >= talker.config.vocab_size)).any():
        raise ValueError("Speech code outside Talker vocabulary")
    for special in (talker.codec_bos_token, talker.codec_eos_token,
                    talker.codec_pad_token, talker.codec_mask_token):
        if (ids == special).any():
            raise ValueError("codec_ids must not include BOS/EOS/PAD/MASK; EOS is added automatically")
    return ids.long()


def _prepare_talker_inputs(thinker, talker, thinker_output, spans,
                           window_index, codes, text_bos_token_id, detach_thinker):
    span = spans[window_index]
    prompt_length = span.text_start
    # The final Thinker stop token is a boundary, not spoken text conditioning.
    text_end = span.text_end - 1
    embeddings = thinker_output.hidden_states[0][:, :text_end]
    # Native device_map dispatch may place the first and final decoder states
    # on different devices. Keep transfers differentiable and join one span only.
    hidden = thinker_output.hidden_states[-1][:, :text_end].to(
        device=embeddings.device, dtype=embeddings.dtype,
    )
    audio_mask = torch.zeros((1, prompt_length, 1), dtype=torch.bool, device=embeddings.device)
    for previous in spans[:window_index + 1]:
        audio_mask[:, previous.audio_start + 1:previous.audio_end - 1] = True
    # Match native Omni: raw modality embeddings are removed, contextual hidden
    # states remain. Audio boundary markers and action history remain intact.
    prompt = hidden[:, :prompt_length] + embeddings[:, :prompt_length].masked_fill(audio_mask, 0)
    reply = hidden[:, span.text_start:text_end] + embeddings[:, span.text_start:text_end]
    bos = token_embeddings(thinker, [text_bos_token_id]).to(embeddings)
    eos = token_embeddings(thinker, [talker.text_eos_token]).to(embeddings)
    pad = token_embeddings(thinker, [talker.text_pad_token]).to(embeddings)
    conditions = torch.cat((reply, eos, pad), dim=1)
    # One text condition per codec input, not one codec per word. After text EOS,
    # the native incremental path keeps reusing the last (text PAD) condition.
    indices = torch.arange(codes.numel() + 1, device=conditions.device).clamp_max(conditions.shape[1] - 1)
    conditions = conditions.index_select(1, indices)
    conditions = torch.cat((prompt, bos, conditions), dim=1)
    if detach_thinker:
        conditions = conditions.detach()
    weight = talker.get_input_embeddings().weight
    conditions = conditions.to(device=weight.device, dtype=weight.dtype)
    codec_inputs = torch.cat((codes.new_tensor([talker.codec_pad_token, talker.codec_bos_token]), codes))
    codec_embeddings = talker.get_input_embeddings()(codec_inputs.unsqueeze(0))
    inputs = torch.cat((conditions[:, :prompt_length],
                        conditions[:, prompt_length:] + codec_embeddings), dim=1)
    length = inputs.shape[1]
    # These labels are already NEXT-token targets aligned to output logits.
    # Unlike Thinker's native labels, they must not undergo another causal shift.
    labels = codes.new_full((1, length), -100)
    labels[:, prompt_length + 1:] = torch.cat((codes, codes.new_tensor([talker.codec_eos_token])))
    return {
        "inputs_embeds": inputs,
        "attention_mask": torch.ones((1, length), device=inputs.device, dtype=torch.long),
        # Explicit positions bypass native forward's in-place prefill BOS/PAD
        # insertion; those codec embeddings have already been added above.
        "position_ids": torch.arange(length, device=inputs.device).view(1, 1, -1).expand(3, 1, -1),
        "use_cache": False,
        "return_dict": True,
    }, labels


def forward_with_speech_loss(model, tokenizer, prefix_ids, windows, speech_targets, *,
                             speaker="Chelsie", thinker_loss_weight=1.0,
                             speech_loss_weight=1.0, detach_thinker=False):
    """Teacher-force Thinker actions/text and native Talker speech codes.

    Requires a composite exposing native ``thinker``, ``talker`` and
    ``speaker_map[speaker]['bos_token']``. No generate(), argmax, or inference
    session is used. Call ``result.loss.backward()`` in a normal training loop.
    Module train/eval modes and parameter freezing remain caller-controlled.

    Each SpeechTarget supplies an independently terminated utterance for one
    speak/backchannel window. Only its causal history conditions Talker; future
    windows and the custom end_speech token are not reply conditions. Talker
    receives native text BOS/codec PAD, first text/codec BOS, then preceding
    ground-truth codes plus successive text conditions (EOS, then repeated PAD).
    Speech CE includes exactly the raw codes and one codec EOS, averaged over
    all supervised codec tokens across targets, not over utterance means.

    ``detach_thinker`` disconnects ONLY speech conditioning; the Thinker loss
    still trains Thinker. Targets may be empty for action/text-only examples.
    Unpaired speaking windows receive no speech loss. The returned Talker
    outputs follow speech_targets order. This does not train Token2Wav or infer
    compatible target codes from waveforms, nor provide cross-window streaming
    speech alignment. Codec vocabulary validation cannot establish provenance.

    Example (codes must come from a compatible target speech encoder)::

        result = forward_with_speech_loss(
            model, tokenizer, prefix_ids, windows,
            [SpeechTarget(window_index=1, codec_ids=target_codes)],
        )
        result.loss.backward()
    """
    thinker_loss_weight = _loss_weight(thinker_loss_weight, "thinker_loss_weight")
    speech_loss_weight = _loss_weight(speech_loss_weight, "speech_loss_weight")
    if thinker_loss_weight == speech_loss_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    if not isinstance(detach_thinker, bool):
        raise ValueError("detach_thinker must be a bool")
    thinker, talker = getattr(model, "thinker", None), getattr(model, "talker", None)
    if thinker is None or talker is None or getattr(talker.config, "model_type", None) != "qwen2_5_omni_talker":
        raise TypeError("Speech training requires native Thinker and Talker modules on model")
    if (thinker.get_input_embeddings().weight.shape[1] != talker.config.embedding_size
            or talker.get_input_embeddings().weight.shape[1] != talker.config.embedding_size):
        raise ValueError("Thinker and Talker conditioning embedding sizes must match")
    speaker_map = getattr(model, "speaker_map", {})
    if not isinstance(speaker, str) or speaker not in speaker_map:
        raise ValueError(f"Unknown Talker speaker: {speaker!r}")
    bos = speaker_map[speaker].get("bos_token")
    text_specials = (bos, talker.text_eos_token, talker.text_pad_token)
    rows = thinker.get_input_embeddings().num_embeddings
    if any(type(token) is not int or not 0 <= token < rows for token in text_specials):
        raise ValueError("Talker text special IDs must fit Thinker embeddings")
    codec_specials = (talker.codec_bos_token, talker.codec_eos_token,
                      talker.codec_pad_token, talker.codec_mask_token)
    if (len(set(codec_specials)) != 4
            or any(type(token) is not int or not 0 <= token < talker.config.vocab_size for token in codec_specials)):
        raise ValueError("Talker codec special IDs must be distinct and inside its vocabulary")
    actions = validate_action_tokens(model, tokenizer)
    if set(text_specials) & set(actions.values()):
        raise ValueError("Talker text specials must not be duplex action tokens")
    stop_ids, _ = text_vocabulary(thinker, tokenizer, actions)
    windows = list(windows)
    thinker_inputs, spans = _prepare_interleaved_inputs(model, tokenizer, prefix_ids, windows)
    targets = []
    seen = set()
    for target in speech_targets:
        if not isinstance(target, SpeechTarget):
            raise TypeError("speech_targets must contain SpeechTarget values")
        index = target.window_index
        if type(index) is not int or not 0 <= index < len(windows):
            raise ValueError("Speech target window_index is outside the training windows")
        if index in seen:
            raise ValueError("Duplicate speech target window_index")
        seen.add(index)
        window = windows[index]
        if window.action not in ("<|speak|>", "<|backchannel|>"):
            raise ValueError("Speech targets require a speak/backchannel window")
        if len(window.text_ids) < 2 or window.text_ids[-1] not in stop_ids:
            raise ValueError("Speech targets require nonempty text terminated by EOS/end_speech")
        codes = _codec_targets(talker, target.codec_ids)
        if spans[index].text_start + 2 + codes.numel() > talker.config.max_position_embeddings:
            raise ValueError("Speech teacher-forcing sequence exceeds Talker context budget")
        targets.append((index, codes))
    thinker_output = thinker(**thinker_inputs, output_hidden_states=True)
    speech_sum = None
    speech_count = 0
    talker_outputs = []
    for index, codes in targets:
        talker_inputs, labels = _prepare_talker_inputs(
            thinker, talker, thinker_output, spans, index, codes, bos, detach_thinker,
        )
        output = talker(**talker_inputs)
        # Talker.forward in Transformers 4.57.6 returns logits but no native loss.
        loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                               labels.to(output.logits.device).reshape(-1), ignore_index=-100, reduction="sum")
        speech_sum = loss if speech_sum is None else speech_sum + loss
        speech_count += codes.numel() + 1
        talker_outputs.append(output)
    thinker_loss = thinker_output.loss
    speech_loss = (speech_sum / speech_count if speech_count
                   else thinker_loss.new_zeros(()))
    loss = thinker_loss_weight * thinker_loss + speech_loss_weight * speech_loss.to(thinker_loss.device)
    return SpeechTrainingOutput(loss, thinker_loss, speech_loss, thinker_output, tuple(talker_outputs))
