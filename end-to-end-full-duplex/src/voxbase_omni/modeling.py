"""VoxBase-omni input-side model for audio-conditioned causal LM pretraining."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .audio import AudioProjector, GatedAudioFusion, MimiCodeEmbedding, align_continuous, require_integer


@dataclass
class PreparedInputs:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor | None


class VoxBaseOmni(nn.Module):
    """Text and one embedding per Mimi frame, interleaved in one causal sequence.

    audio_positions maps each Mimi frame to an existing slot in input_ids, not
    to a new vocabulary ID. Use -1 for padded frames. Audio/padding labels are
    always ignored; the caller additionally masks prompts with -100.
    """

    def __init__(
        self,
        llm: nn.Module,
        *,
        audio_encoder: nn.Module | None = None,
        audio_hidden_size: int = 1024,
        codebook_size: int = 2048,
        adapter_dim: int = 256,
    ):
        super().__init__()
        if audio_hidden_size <= 0:
            raise ValueError("audio_hidden_size must be positive")
        self.llm = llm
        self.audio_encoder = audio_encoder
        embedding = llm.get_input_embeddings().weight
        if embedding.device.type == "meta":
            raise ValueError("Initialize the LLM on a real device before attaching audio modules")
        self.hidden_size = embedding.shape[1]
        self.audio_hidden_size = audio_hidden_size
        self.discrete_encoder = MimiCodeEmbedding(self.hidden_size, codebook_size, adapter_dim)
        self.projector = AudioProjector(audio_hidden_size, self.hidden_size)
        self.fusion = GatedAudioFusion(self.hidden_size)
        for module in (self.discrete_encoder, self.projector, self.fusion):
            module.to(device=embedding.device, dtype=embedding.dtype)

    @classmethod
    def from_pretrained(
        cls,
        llm_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        with_continuous_encoder: bool = False,
        audio_name_or_path: str = "Qwen/Qwen3-ASR-0.6B",
        llm_kwargs: dict[str, Any] | None = None,
        audio_kwargs: dict[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> "VoxBaseOmni":
        """Initialize pretrained backbones; new input adapters start untrained.

        This is not a VoxBase checkpoint loader. Resume trained weights with
        load_state_dict after reconstructing the same module configuration.
        """
        from transformers import AutoModelForCausalLM

        llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, **(llm_kwargs or {}))
        audio_encoder = None
        if with_continuous_encoder:
            from .encoders import Qwen3ASRAudioEncoder

            audio_encoder = Qwen3ASRAudioEncoder.from_pretrained(
                audio_name_or_path, **(audio_kwargs or {})
            )
            model_kwargs.setdefault("audio_hidden_size", audio_encoder.output_dim)
            if model_kwargs["audio_hidden_size"] != audio_encoder.output_dim:
                raise ValueError("audio_hidden_size does not match the loaded AuT")
        return cls(llm, audio_encoder=audio_encoder, **model_kwargs)

    def encode_audio(
        self,
        codes: Tensor,
        frame_mask: Tensor,
        *,
        continuous_features: Tensor | None = None,
        continuous_frame_indices: Tensor | None = None,
        continuous_feature_mask: Tensor | None = None,
        audio_inputs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        if audio_inputs is not None:
            if continuous_features is not None or continuous_feature_mask is not None:
                raise ValueError("Supply either audio_inputs or cached continuous features/mask")
            if self.audio_encoder is None:
                raise ValueError("audio_inputs requires an attached continuous audio encoder")
            continuous_features, continuous_feature_mask = self.audio_encoder(**audio_inputs)
        discrete = self.discrete_encoder(codes, frame_mask)
        if continuous_features is None:
            if continuous_frame_indices is not None or continuous_feature_mask is not None:
                raise ValueError("Continuous alignment/mask requires continuous features")
            return discrete
        if continuous_frame_indices is None:
            raise ValueError("Explicit continuous_frame_indices are required for frame alignment")
        if continuous_features.ndim != 3 or not continuous_features.is_floating_point():
            raise ValueError("continuous_features must be floating point [batch, time, width]")
        if continuous_features.shape[-1] != self.audio_hidden_size:
            raise ValueError("Continuous feature width does not match audio_hidden_size")
        if any(t.device != codes.device for t in (
            continuous_features, continuous_frame_indices,
            *(() if continuous_feature_mask is None else (continuous_feature_mask,)),
        )):
            raise ValueError("Audio features, masks and indices must be on the same device")
        aligned, available = align_continuous(
            continuous_features, continuous_frame_indices, frame_mask, continuous_feature_mask
        )
        if not available.any():
            return discrete
        # Select before projecting: absent/NaN padded features never enter fusion.
        projected = self.projector(aligned[available].to(dtype=discrete.dtype))
        fused = discrete.clone()
        fused[available] = self.fusion(discrete[available], projected)
        return fused

    def prepare_inputs(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        audio_codes: Tensor | None = None,
        audio_positions: Tensor | None = None,
        continuous_features: Tensor | None = None,
        continuous_frame_indices: Tensor | None = None,
        continuous_feature_mask: Tensor | None = None,
        audio_inputs: dict[str, Tensor] | None = None,
    ) -> PreparedInputs:
        require_integer(input_ids, "input_ids")
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must be nonempty [batch, sequence]")
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape or not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must be binary [batch, sequence]")
        if attention_mask.device != input_ids.device:
            raise ValueError("attention_mask and input_ids must be on the same device")
        attention_mask = attention_mask.bool()
        if not attention_mask.any(dim=1).all():
            raise ValueError("Every sample must have at least one non-padding position")
        if labels is not None:
            require_integer(labels, "labels")
            if labels.shape != input_ids.shape or labels.device != input_ids.device:
                raise ValueError("labels must have input_ids shape and device")
            labels = labels.long().clone().masked_fill(~attention_mask, -100)
        audio_slots = torch.zeros_like(attention_mask)
        audio_hidden = None
        if audio_codes is None:
            if any(value is not None for value in (
                audio_positions, continuous_features, continuous_frame_indices,
                continuous_feature_mask, audio_inputs,
            )):
                raise ValueError("Audio arguments require audio_codes; continuous-only input is not implemented")
        else:
            if audio_positions is None:
                raise ValueError("audio_codes requires audio_positions")
            require_integer(audio_positions, "audio_positions")
            if audio_codes.ndim != 3 or audio_codes.shape[:2] != (batch, 8):
                raise ValueError("audio_codes must have shape [batch, 8, frames]")
            if audio_positions.shape != (batch, audio_codes.shape[2]):
                raise ValueError("audio_positions must have shape [batch, frames]")
            if audio_positions.device != input_ids.device or audio_codes.device != input_ids.device:
                raise ValueError("Audio codes, positions and input_ids must be on the same device")
            if ((audio_positions < -1) | (audio_positions >= length)).any():
                raise ValueError("Audio positions must be sequence indices or -1 for padding")
            frame_mask = audio_positions >= 0
            for row in range(batch):
                positions = audio_positions[row, frame_mask[row]].long()
                if positions.numel() > 1 and (positions[1:] <= positions[:-1]).any():
                    raise ValueError("Audio positions must be unique and strictly increasing per sample")
                if not attention_mask[row, positions].all():
                    raise ValueError("Audio positions cannot point into sequence padding")
                audio_slots[row, positions] = True
            audio_hidden = self.encode_audio(
                audio_codes, frame_mask,
                continuous_features=continuous_features,
                continuous_frame_indices=continuous_frame_indices,
                continuous_feature_mask=continuous_feature_mask,
                audio_inputs=audio_inputs,
            )
        # Audio slots need no special vocabulary token; their IDs are never embedded.
        safe_ids = input_ids.masked_fill(audio_slots | ~attention_mask, 0).long()
        embeddings = self.llm.get_input_embeddings()(safe_ids)
        if audio_hidden is not None:
            embeddings = embeddings.clone()
            embeddings[audio_slots] = audio_hidden[frame_mask].to(dtype=embeddings.dtype)
        if labels is not None:
            labels = labels.masked_fill(audio_slots, -100)
            vocab_size = self.llm.get_input_embeddings().num_embeddings
            if ((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any():
                raise ValueError("Text labels must be vocabulary indices or -100")
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        return PreparedInputs(embeddings, attention_mask, position_ids, labels)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        audio_codes: Tensor | None = None,
        audio_positions: Tensor | None = None,
        continuous_features: Tensor | None = None,
        continuous_frame_indices: Tensor | None = None,
        continuous_feature_mask: Tensor | None = None,
        audio_inputs: dict[str, Tensor] | None = None,
        output_hidden_states: bool = False,
    ):
        prepared = self.prepare_inputs(
            input_ids, attention_mask=attention_mask, labels=labels,
            audio_codes=audio_codes, audio_positions=audio_positions,
            continuous_features=continuous_features,
            continuous_frame_indices=continuous_frame_indices,
            continuous_feature_mask=continuous_feature_mask, audio_inputs=audio_inputs,
        )
        output = self.llm(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if prepared.labels is not None:
            targets = prepared.labels[:, 1:].clone()
            # Padding cannot supply a causal prediction of the next token.
            targets.masked_fill_(~prepared.attention_mask[:, :-1], -100)
            if (targets != -100).any():
                loss = F.cross_entropy(
                    output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1), ignore_index=-100,
                )
            else:
                # HF's mean CE is NaN for all-ignored batches; preserve a zero gradient.
                loss = output.logits.float().sum() * 0.0
            output["loss"] = loss
        return output
