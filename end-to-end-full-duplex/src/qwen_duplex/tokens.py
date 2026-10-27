"""Native Thinker action tokens and locally reloadable Qwen2.5-Omni checkpoints.

Actions are ordinary autoregressive LM outputs, not an external controller. Extend
before creating an optimizer: Transformers may replace the embedding parameters.
Reserved model rows receive explicit special-token placeholders before *new*
actions are appended, so no pretrained row is repurposed as an action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTION_TOKENS = (
    "<|wait|>",
    "<|speak|>",
    "<|interrupt|>",
    "<|backchannel|>",
    "<|yield|>",
    "<|end_speech|>",
)

_METADATA_FILE = "duplex_tokens.json"
_FORMAT_VERSION = 1
_THINKER_TYPE = "qwen2_5_omni_thinker"
_COMPOSITE_TYPE = "qwen2_5_omni"


def _thinker(model: Any) -> Any:
    """Resolve only the text-producing Thinker, never composite embeddings."""
    thinker = getattr(model, "thinker", model)
    if getattr(getattr(thinker, "config", None), "model_type", None) != _THINKER_TYPE:
        raise TypeError("Expected a native Qwen2.5-Omni Thinker or a model.thinker")
    return thinker


def _vocab(tokenizer: Any) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Tokenizer must have nonnegative integer vocabulary IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Tokenizer vocabulary IDs must be unique")
    return vocab


def _dimensions(thinker: Any) -> int:
    """Check both vocabulary matrices and the native text configuration."""
    embedding = thinker.get_input_embeddings()
    head = thinker.get_output_embeddings()
    if embedding is None or head is None:
        raise ValueError("Thinker needs both input embeddings and an LM head")
    input_shape, output_shape = embedding.weight.shape, head.weight.shape
    if len(input_shape) != 2 or len(output_shape) != 2 or input_shape != output_shape:
        raise ValueError("Thinker embedding and LM head row dimensions must match")
    rows, hidden = input_shape
    config = thinker.config.text_config
    if rows != config.vocab_size or hidden != config.hidden_size:
        raise ValueError("Thinker text config does not match embedding/LM head dimensions")
    if getattr(thinker, "vocab_size", rows) != rows:
        raise ValueError("Thinker vocab_size does not match embedding rows")
    return rows


def _action_ids(tokenizer: Any) -> dict[str, int]:
    vocab = _vocab(tokenizer)
    result = {}
    special_tokens = set(tokenizer.all_special_tokens)
    for token in ACTION_TOKENS:
        if token not in vocab or token not in special_tokens:
            raise ValueError(f"Missing action special token: {token}")
        token_id = vocab[token]
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Action token must encode atomically: {token}")
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Action token does not round-trip: {token}")
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Action token must not be the unknown token: {token}")
        result[token] = token_id
    if len(set(result.values())) != len(ACTION_TOKENS):
        raise ValueError("Action token IDs must be unique")
    # Adjacent actions must remain distinct without needing spaces or a controller.
    if tokenizer.encode("".join(ACTION_TOKENS), add_special_tokens=False) != list(result.values()):
        raise ValueError("Adjacent action tokens must encode atomically")
    return result


def validate_action_tokens(model: Any, tokenizer: Any) -> dict[str, int]:
    """Validate atomic special actions and Thinker vocabulary dimensions.

    Accepts either a native Thinker or a composite exposing ``model.thinker``.
    Returns the six exact token IDs without changing tokenizer or model. Padded
    embedding rows are allowed, but every tokenizer ID must fit both matrices.
    Raises ``ValueError`` for missing/non-atomic actions or inconsistent shapes.
    """
    thinker = _thinker(model)
    rows = _dimensions(thinker)
    ids = _action_ids(tokenizer)
    if max(_vocab(tokenizer).values()) >= rows:
        raise ValueError("Tokenizer vocabulary exceeds Thinker embedding/LM head rows")
    if thinker is not model:
        config = getattr(getattr(model, "config", None), "thinker_config", None)
        if config is None or config.text_config.to_dict() != thinker.config.text_config.to_dict():
            raise ValueError("Composite config.thinker_config is out of sync with Thinker")
    return ids


def extend_action_tokens(model: Any, tokenizer: Any) -> dict[str, int]:
    """Append six action specials and resize only the native Thinker.

    Existing vocabulary IDs, special-token registrations and pretrained rows are
    preserved. If the model has more rows than the tokenizer, explicit
    ``<|duplex_reserved_ID|>`` special placeholders occupy that gap *before* new
    actions are added. Existing action tokens retain their IDs and are promoted
    to specials if necessary. Repeating this function does not grow the vocab or
    reinitialize weights. Talker and Token2Wav are never resized.
    """
    from transformers import AddedToken

    thinker = _thinker(model)
    rows = _dimensions(thinker)
    vocab = _vocab(tokenizer)
    old_special_ids = {token: vocab[token] for token in tokenizer.all_special_tokens}
    missing = [token for token in ACTION_TOKENS if token not in vocab]
    if missing:
        # Tokenizer length is not the model vocabulary size (e.g. Qwen 152064).
        # Use the highest actual ID as well, so sparse vocabularies cannot shrink
        # the model or place new actions inside its reserved rows.
        next_id = max(vocab.values()) + 1
        placeholders = []
        for token_id in range(next_id, rows):
            token = f"<|duplex_reserved_{token_id}|>"
            while token in vocab:
                token = token[:-2] + "_reserved|>"
            placeholders.append(AddedToken(token, special=True, normalized=False))
        if placeholders:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": placeholders},
                replace_additional_special_tokens=False,
            )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(t, special=True, normalized=False) for t in ACTION_TOKENS]},
        replace_additional_special_tokens=False,
    )
    ids = _action_ids(tokenizer)
    if any(ids[token] < rows for token in missing):
        raise ValueError("Tokenizer allocated a new action inside reserved Thinker rows")
    vocab = _vocab(tokenizer)
    if any(token not in tokenizer.all_special_tokens or vocab.get(token) != i for token, i in old_special_ids.items()):
        raise ValueError("Tokenizer failed to preserve existing special tokens")
    new_rows = max(rows, max(vocab.values()) + 1)
    if new_rows != rows:
        # Disabling covariance-based initialization avoids a large decomposition
        # on production checkpoints; native random initialization trains normally.
        thinker.resize_token_embeddings(new_rows, mean_resizing=False)
    if thinker is not model:
        model.config.thinker_config = thinker.config
    return validate_action_tokens(model, tokenizer)


def save_duplex_checkpoint(model: Any, processor: Any, path: str | Path) -> None:
    """Save a native model, complete processor and exact action-ID metadata.

    ``path`` must not exist or must be an empty directory; files, symlinks and
    nonempty directories are rejected instead of overwritten. Composite saves
    retain the full model (including Talker/Token2Wav) and native speaker map.
    The metadata completion marker is written last. A failed save may leave an
    incomplete directory, which must not be used as a duplex checkpoint.
    """
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration

    target = Path(path)
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError(f"Checkpoint target must be a new or empty directory: {target}")
    if isinstance(model, Qwen2_5OmniForConditionalGeneration):
        model_type = _COMPOSITE_TYPE
        if bool(model.config.enable_audio_output) != bool(model.has_talker):
            raise ValueError("Composite enable_audio_output config is inconsistent with its Talker")
    elif isinstance(model, Qwen2_5OmniThinkerForConditionalGeneration):
        model_type = _THINKER_TYPE
    else:
        raise TypeError("Checkpoint saving requires a native Transformers Qwen2.5-Omni model")
    ids = validate_action_tokens(model, processor.tokenizer)
    metadata = {
        "format_version": _FORMAT_VERSION,
        "model_type": model_type,
        "action_token_ids": ids,
        "thinker_vocab_size": _dimensions(_thinker(model)),
    }
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    # Unified native serialization avoids image/audio preprocessors overwriting
    # the same legacy preprocessor_config.json in Transformers 4.57.
    processor.save_pretrained(target, legacy_serialization=False)
    if model_type == _COMPOSITE_TYPE:
        # Native composite from_pretrained unconditionally reads this file;
        # save_pretrained alone does not persist the speaker map in Transformers.
        torch.save(model.speaker_map, target / "spk_dict.pt")
    (target / _METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_duplex_checkpoint(path: str | Path, **model_kwargs: Any) -> tuple[Any, Any]:
    """Reload a local native Thinker/composite and its full Omni processor.

    Native Transformers classes are selected using versioned metadata, not
    architecture guessing. Exact saved action IDs and Thinker row counts must
    match; loading never silently extends or repairs a checkpoint. Model kwargs
    such as ``dtype`` and ``device_map`` are passed to native ``from_pretrained``.
    Network fetching and mismatched-size reinitialization are not permitted.
    """
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    target = Path(path)
    if not target.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {target}")
    metadata = json.loads((target / _METADATA_FILE).read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("format_version")) is not int
        or metadata["format_version"] != _FORMAT_VERSION
    ):
        raise ValueError("Unsupported duplex checkpoint metadata version")
    model_classes = {
        _THINKER_TYPE: Qwen2_5OmniThinkerForConditionalGeneration,
        _COMPOSITE_TYPE: Qwen2_5OmniForConditionalGeneration,
    }
    model_type = metadata.get("model_type")
    if not isinstance(model_type, str) or model_type not in model_classes:
        raise ValueError("Unsupported duplex checkpoint model type")
    # Check before native construction: treating a Thinker config as a composite
    # could otherwise allocate default (production-size) audio components.
    native_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if not isinstance(native_config, dict) or native_config.get("model_type") != model_type:
        raise ValueError("Checkpoint native model type does not match metadata")
    expected = metadata.get("action_token_ids")
    if (
        not isinstance(expected, dict)
        or set(expected) != set(ACTION_TOKENS)
        or any(type(i) is not int or i < 0 for i in expected.values())
        or len(set(expected.values())) != len(ACTION_TOKENS)
    ):
        raise ValueError("Invalid action token IDs in checkpoint metadata")
    expected_rows = metadata.get("thinker_vocab_size")
    if type(expected_rows) is not int or expected_rows <= max(expected.values()):
        raise ValueError("Invalid Thinker vocabulary size in checkpoint metadata")
    if model_kwargs.get("ignore_mismatched_sizes"):
        raise ValueError("Duplex checkpoints must not reinitialize mismatched weight sizes")
    model_kwargs["local_files_only"] = True
    processor = Qwen2_5OmniProcessor.from_pretrained(target, local_files_only=True)
    if _action_ids(processor.tokenizer) != expected:
        raise ValueError("Checkpoint action token IDs do not match metadata")
    model = model_classes[model_type].from_pretrained(target, **model_kwargs)
    if validate_action_tokens(model, processor.tokenizer) != expected:
        raise ValueError("Checkpoint action token IDs do not match metadata")
    if _dimensions(_thinker(model)) != expected_rows:
        raise ValueError("Checkpoint Thinker vocabulary size does not match metadata")
    return model, processor
