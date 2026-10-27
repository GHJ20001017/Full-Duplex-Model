"""Shared audio/action/text serialization for teacher forcing and cached decoding."""

from dataclasses import dataclass

import torch

from .tokens import validate_action_tokens


def token_embeddings(thinker, token_ids):
    ids = torch.as_tensor(token_ids, device=thinker.get_input_embeddings().weight.device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be nonempty integer [length] or [1, length]")
    if ((ids < 0) | (ids >= thinker.get_input_embeddings().num_embeddings)).any():
        raise ValueError("Token ID outside Thinker vocabulary")
    return thinker.get_input_embeddings()(ids.long())


def audio_packet(thinker, embeddings):
    weight = thinker.get_input_embeddings().weight
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or embeddings.shape[0] == 0 or embeddings.shape[1] != weight.shape[1]
            or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()):
        raise ValueError("Audio embeddings must be finite, nonempty [positions, hidden_size]")
    return torch.cat((
        token_embeddings(thinker, [thinker.config.audio_start_token_id]),
        embeddings.to(device=weight.device, dtype=weight.dtype).unsqueeze(0),
        token_embeddings(thinker, [thinker.config.audio_end_token_id]),
    ), dim=1)


def text_vocabulary(thinker, tokenizer, actions):
    """The shared text-phase grammar (actions are selected only at window boundaries)."""
    eos = thinker.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    stop_ids.add(actions["<|end_speech|>"])
    known_ids = set(tokenizer.get_vocab().values())
    forbidden = (set(tokenizer.all_special_ids) | set(actions.values())) - stop_ids
    forbidden.update(set(range(thinker.get_input_embeddings().num_embeddings)) - known_ids)
    return stop_ids, forbidden


@dataclass
class TrainingWindow:
    """One newly committed audio window, one action, then optional target text IDs."""

    audio_embeddings: torch.Tensor
    action: str
    text_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _WindowSpan:
    audio_start: int
    audio_end: int
    text_start: int
    text_end: int


def _prepare_interleaved_inputs(model, tokenizer, prefix_ids, windows):
    """Build native Thinker inputs and spans from the same serialization pass."""
    thinker = getattr(model, "thinker", model)
    actions = validate_action_tokens(model, tokenizer)
    stop_ids, forbidden = text_vocabulary(thinker, tokenizer, actions)
    parts = [token_embeddings(thinker, prefix_ids)]
    labels = [torch.full(parts[0].shape[:2], -100, device=parts[0].device, dtype=torch.long)]
    windows = list(windows)
    if not windows:
        raise ValueError("At least one training window is required")
    spans = []
    length = parts[0].shape[1]
    for window in windows:
        if window.action not in actions:
            raise ValueError(f"Unknown action: {window.action}")
        if window.text_ids and window.action not in ("<|speak|>", "<|backchannel|>"):
            raise ValueError("Text targets require speak or backchannel")
        if any(token in forbidden for token in window.text_ids):
            raise ValueError("Text targets contain tokens forbidden by the text-phase grammar")
        if any(token in stop_ids for token in window.text_ids[:-1]):
            raise ValueError("Text targets cannot continue after EOS/end_speech")
        packet = audio_packet(thinker, window.audio_embeddings)
        targets = [actions[window.action], *window.text_ids]
        audio_end = length + packet.shape[1]
        spans.append(_WindowSpan(length, audio_end, audio_end + 1, audio_end + len(targets)))
        length = spans[-1].text_end
        parts.extend((packet, token_embeddings(thinker, targets)))
        labels.extend((torch.full(packet.shape[:2], -100, device=packet.device, dtype=torch.long),
                       torch.tensor([targets], device=packet.device, dtype=torch.long)))
    embeddings = torch.cat(parts, dim=1)
    length = embeddings.shape[1]
    if length > thinker.config.text_config.max_position_embeddings:
        raise ValueError("Interleaved sequence exceeds the model context budget")
    return {
        "inputs_embeds": embeddings,
        "labels": torch.cat(labels, dim=1),
        "attention_mask": torch.ones((1, length), device=embeddings.device, dtype=torch.long),
        "position_ids": torch.arange(length, device=embeddings.device).view(1, 1, -1).expand(3, 1, -1),
        "use_cache": False,
        "return_dict": True,
    }, tuple(spans)


def prepare_interleaved_inputs(model, tokenizer, prefix_ids, windows):
    """Build batch-size-one differentiable inputs for native Thinker.forward().

    Prefix/audio labels are -100. Action and optional text labels use native
    causal shifting (do not shift them again). All three RoPE axes are monotonic
    sequence positions, matching ThinkerSession; this protocol is audio/text
    only and does not implement video timestamps or image grids.
    """
    inputs, _ = _prepare_interleaved_inputs(model, tokenizer, prefix_ids, windows)
    return inputs
