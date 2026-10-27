"""Offline native-model integration tests; run with stdlib unittest, no pytest needed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    WhisperFeatureExtractor,
)

# Use the local source tree without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_duplex.tokens import (  # noqa: E402
    ACTION_TOKENS,
    extend_action_tokens,
    load_duplex_checkpoint,
    save_duplex_checkpoint,
    validate_action_tokens,
)


def make_processor(extra_vocab: tuple[str, ...] = ()) -> Qwen2_5OmniProcessor:
    """Build a real native processor with an entirely local fast WordLevel tokenizer."""
    words = ("[UNK]", "hello", "world", "<|endoftext|>", "<|old_special|>", *extra_vocab)
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|old_special|>"],
        extra_special_tokens={
            "image_token": "<|IMAGE|>",
            "audio_token": "<|AUDIO|>",
            "video_token": "<|VIDEO|>",
            "vision_bos_token": "<|vision_bos|>",
            "vision_eos_token": "<|vision_eos|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        },
    )
    return Qwen2_5OmniProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(feature_size=4),
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )


def make_thinker(rows: int = 24, tied: bool = False) -> Qwen2_5OmniThinkerForConditionalGeneration:
    """Instantiate a real one-layer native Thinker (roughly ten thousand weights)."""
    config = Qwen2_5OmniThinkerConfig(
        text_config=dict(
            vocab_size=rows,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            tie_word_embeddings=tied,
            pad_token_id=3,
            bos_token_id=3,
            eos_token_id=3,
        ),
        audio_config=dict(
            num_mel_bins=4,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            d_model=16,
            max_source_positions=16,
            n_window=4,
            output_dim=16,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
            window_size=4,
            out_hidden_size=16,
            fullatt_block_indexes=[0],
        ),
        pad_token_id=3,
        bos_token_id=3,
        eos_token_id=3,
        vision_start_token_id=8,
        vision_end_token_id=9,
    )
    return Qwen2_5OmniThinkerForConditionalGeneration(config)


class CompositeWithUntouchableTalker:
    """Exercise composite routing with real Thinker weights and inert audio state."""

    def __init__(self, thinker):
        self.thinker = thinker
        self.config = SimpleNamespace(thinker_config=copy.deepcopy(thinker.config))
        self.talker = torch.nn.Embedding(11, 16)
        self.token2wav = torch.nn.Linear(4, 4)
        self.talker.resize_token_embeddings = self.forbidden

    def forbidden(self, *args, **kwargs):
        raise AssertionError("Composite/Talker embedding APIs must not be used")

    get_input_embeddings = forbidden
    get_output_embeddings = forbidden
    resize_token_embeddings = forbidden


class ActionTokenTests(unittest.TestCase):
    """Use actual Transformer weights/tokenizers; only the audio sentinel is a stub."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        import os

        self.env = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        temporary = tempfile.TemporaryDirectory(prefix="qwen-duplex-test-")
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        torch.manual_seed(7)
        self.model = make_thinker()
        self.processor = make_processor()

    def test_extension_preserves_every_old_row_and_special(self):
        for padded in (False, True):
            for tied in (False, True):
                with self.subTest(padded=padded, tied=tied):
                    tokenizer = make_processor().tokenizer
                    rows = len(tokenizer) + (12 if padded else 0)
                    model = make_thinker(rows, tied=tied)
                    inputs = model.get_input_embeddings().weight.detach().clone()
                    outputs = model.get_output_embeddings().weight.detach().clone()
                    old_vocab = tokenizer.get_vocab().copy()
                    old_specials = set(tokenizer.all_special_tokens)
                    old_additional = list(tokenizer.additional_special_tokens)
                    ids = extend_action_tokens(model, tokenizer)
                    self.assertEqual(ACTION_TOKENS, (
                        "<|wait|>", "<|speak|>", "<|interrupt|>", "<|backchannel|>", "<|yield|>", "<|end_speech|>",
                    ))
                    self.assertEqual(ids, dict(zip(ACTION_TOKENS, range(rows, rows + 6))))
                    self.assertEqual(len(tokenizer), rows + 6)
                    self.assertTrue(all(tokenizer.get_vocab()[token] == i for token, i in old_vocab.items()))
                    self.assertLessEqual(old_specials, set(tokenizer.all_special_tokens))
                    self.assertLessEqual(set(old_additional), set(tokenizer.additional_special_tokens))
                    self.assertTrue(torch.equal(model.get_input_embeddings().weight[:rows], inputs))
                    self.assertTrue(torch.equal(model.get_output_embeddings().weight[:rows], outputs))
                    self.assertEqual(model.config.text_config.vocab_size, rows + 6)
                    self.assertEqual(model.get_output_embeddings().weight.shape, (rows + 6, 16))
                    if tied:
                        self.assertEqual(model.get_input_embeddings().weight.data_ptr(), model.get_output_embeddings().weight.data_ptr())
                    for token, i in ids.items():
                        self.assertEqual(tokenizer.encode(token, add_special_tokens=False), [i])
                        self.assertEqual(tokenizer.decode([i], skip_special_tokens=True), "")
                    self.assertEqual(tokenizer.encode("".join(ACTION_TOKENS), add_special_tokens=False), list(ids.values()))
                    self.assertEqual(validate_action_tokens(model, tokenizer), ids)

    def test_idempotent_without_reinitializing_weights(self):
        model, processor = self.model, self.processor
        ids = extend_action_tokens(model, processor.tokenizer)
        inputs = model.get_input_embeddings().weight
        outputs = model.get_output_embeddings().weight
        input_copy, output_copy = inputs.detach().clone(), outputs.detach().clone()
        vocab = processor.tokenizer.get_vocab().copy()
        specials = processor.tokenizer.all_special_tokens.copy()
        self.assertEqual(extend_action_tokens(model, processor.tokenizer), ids)
        self.assertIs(model.get_input_embeddings().weight, inputs)
        self.assertIs(model.get_output_embeddings().weight, outputs)
        self.assertTrue(torch.equal(inputs, input_copy))
        self.assertTrue(torch.equal(outputs, output_copy))
        self.assertEqual(processor.tokenizer.get_vocab(), vocab)
        self.assertEqual(processor.tokenizer.all_special_tokens, specials)

    def test_existing_action_is_promoted_without_changing_its_id(self):
        model = self.model
        tokenizer = make_processor((ACTION_TOKENS[0],)).tokenizer
        existing_id = tokenizer.get_vocab()[ACTION_TOKENS[0]]
        existing_row = model.get_input_embeddings().weight[existing_id].detach().clone()
        ids = extend_action_tokens(model, tokenizer)
        self.assertEqual(ids[ACTION_TOKENS[0]], existing_id)
        self.assertGreaterEqual(min(ids[token] for token in ACTION_TOKENS[1:]), 24)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight[existing_id], existing_row))
        self.assertEqual(extend_action_tokens(model, tokenizer), ids)

    def test_reserved_placeholder_name_collision(self):
        tokenizer = make_processor(("<|duplex_reserved_13|>",)).tokenizer
        old_id = tokenizer.convert_tokens_to_ids("<|duplex_reserved_13|>")
        ids = extend_action_tokens(self.model, tokenizer)
        self.assertEqual(min(ids.values()), 24)
        self.assertEqual(tokenizer.convert_tokens_to_ids("<|duplex_reserved_13|>"), old_id)

    def test_tokenizer_larger_than_model_never_shrinks_or_drops_old_rows(self):
        model, processor = self.model, self.processor
        processor.tokenizer.add_tokens([f"word_{i}" for i in range(20)])
        old_size = len(processor.tokenizer)
        old_rows = model.get_input_embeddings().weight.detach().clone()
        ids = extend_action_tokens(model, processor.tokenizer)
        self.assertEqual(min(ids.values()), old_size)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight[:24], old_rows))
        self.assertEqual(model.config.text_config.vocab_size, old_size + 6)

    def test_composite_resizes_thinker_only_and_syncs_config(self):
        model, processor = self.model, self.processor
        composite = CompositeWithUntouchableTalker(model)
        talker = composite.talker.weight.detach().clone()
        wav = composite.token2wav.weight.detach().clone()
        ids = extend_action_tokens(composite, processor.tokenizer)
        self.assertIs(composite.config.thinker_config, model.config)
        self.assertEqual(validate_action_tokens(composite, processor.tokenizer), ids)
        self.assertTrue(torch.equal(composite.talker.weight, talker))
        self.assertTrue(torch.equal(composite.token2wav.weight, wav))
        self.assertEqual(composite.talker.num_embeddings, 11)
        self.assertEqual(extend_action_tokens(composite, processor.tokenizer), ids)

    def test_actions_are_trainable_native_lm_outputs(self):
        model, processor = self.model, self.processor
        ids = extend_action_tokens(model, processor.tokenizer)
        model.train()
        inputs = torch.tensor([[1, *ids.values(), 2]])
        old_input = model.get_input_embeddings().weight.detach().clone()
        old_output = model.get_output_embeddings().weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(input_ids=inputs, labels=inputs, attention_mask=torch.ones_like(inputs))
        self.assertEqual(output.logits.shape, (1, inputs.shape[1], len(processor.tokenizer)))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for token_id in ids.values():
            self.assertGreater(model.get_input_embeddings().weight.grad[token_id].abs().sum(), 0)
            self.assertGreater(model.get_output_embeddings().weight.grad[token_id].abs().sum(), 0)
        optimizer.step()
        for token_id in ids.values():
            self.assertFalse(torch.equal(model.get_input_embeddings().weight[token_id], old_input[token_id]))
            self.assertFalse(torch.equal(model.get_output_embeddings().weight[token_id], old_output[token_id]))

    def test_native_save_reload_round_trip(self):
        for composite in (False, True):
            with self.subTest(composite=composite):
                model, processor = make_thinker(), make_processor()
                if composite:
                    # Native composite without allocating production Talker/vocoder.
                    config = Qwen2_5OmniConfig(thinker_config=model.config.to_dict(), enable_audio_output=False)
                    model = Qwen2_5OmniForConditionalGeneration(config)
                    model.speaker_map = {"test-speaker": {"embedding": torch.tensor([1.0, 2.0])}}
                ids = extend_action_tokens(model, processor.tokenizer)
                thinker = model.thinker if composite else model
                before = {key: value.detach().clone() for key, value in model.state_dict().items()}
                target = self.tmp_path / ("composite" if composite else "thinker")
                save_duplex_checkpoint(model, processor, target)
                self.assertTrue((target / "model.safetensors").is_file())
                self.assertTrue((target / "processor_config.json").is_file())
                self.assertTrue((target / "duplex_tokens.json").is_file())
                if composite:
                    self.assertTrue((target / "spk_dict.pt").is_file())
                loaded, loaded_processor = load_duplex_checkpoint(target, device_map="cpu")
                self.assertIs(type(loaded), type(model))
                self.assertIs(type(loaded_processor), Qwen2_5OmniProcessor)
                self.assertEqual(validate_action_tokens(loaded, loaded_processor.tokenizer), ids)
                self.assertEqual(loaded_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())
                self.assertCountEqual(loaded_processor.tokenizer.all_special_tokens, processor.tokenizer.all_special_tokens)
                self.assertEqual(loaded_processor.chat_template, processor.chat_template)
                self.assertEqual(loaded_processor.feature_extractor.feature_size, 4)
                self.assertEqual(before.keys(), loaded.state_dict().keys())
                for name, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)
                loaded_thinker = loaded.thinker if composite else loaded
                x = torch.tensor([[1, *ids.values(), 2]])
                thinker.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(thinker(input_ids=x).logits, loaded_thinker(input_ids=x).logits))
                self.assertEqual(extend_action_tokens(loaded, loaded_processor.tokenizer), ids)
                if composite:
                    self.assertEqual(loaded.config.thinker_config.text_config.vocab_size, thinker.config.text_config.vocab_size)
                    self.assertTrue(torch.equal(loaded.speaker_map["test-speaker"]["embedding"], torch.tensor([1.0, 2.0])))

    def test_save_accepts_existing_empty_directory(self):
        extend_action_tokens(self.model, self.processor.tokenizer)
        save_duplex_checkpoint(self.model, self.processor, self.tmp_path)
        self.assertEqual(load_duplex_checkpoint(self.tmp_path)[0].config.text_config.vocab_size, 30)

    def test_save_rejects_existing_content_without_overwrite(self):
        extend_action_tokens(self.model, self.processor.tokenizer)
        sentinel = self.tmp_path / "sentinel"
        sentinel.write_text("keep me")
        link = self.tmp_path / "link"
        link.symlink_to(self.tmp_path, target_is_directory=True)
        for target in (self.tmp_path, sentinel, link):
            with self.subTest(target=target):
                with self.assertRaisesRegex(FileExistsError, "new or empty"):
                    save_duplex_checkpoint(self.model, self.processor, target)
                self.assertEqual(sentinel.read_text(), "keep me")
                self.assertFalse((self.tmp_path / "duplex_tokens.json").exists())

    def test_load_rejects_corrupt_metadata(self):
        extend_action_tokens(self.model, self.processor.tokenizer)
        save_duplex_checkpoint(self.model, self.processor, self.tmp_path)
        metadata_path = self.tmp_path / "duplex_tokens.json"
        original = json.loads(metadata_path.read_text())
        for corruption in ("ids", "rows", "version", "bool_version", "model_type", "swapped_model_type", "duplicate_ids", "bool_id"):
            with self.subTest(corruption=corruption):
                metadata = copy.deepcopy(original)
                if corruption == "ids":
                    a, b = ACTION_TOKENS[:2]
                    metadata["action_token_ids"][a], metadata["action_token_ids"][b] = metadata["action_token_ids"][b], metadata["action_token_ids"][a]
                elif corruption == "rows":
                    metadata["thinker_vocab_size"] += 1
                elif corruption == "version":
                    metadata["format_version"] = 999
                elif corruption == "bool_version":
                    metadata["format_version"] = True
                elif corruption == "model_type":
                    metadata["model_type"] = "arbitrary_remote_class"
                elif corruption == "swapped_model_type":
                    metadata["model_type"] = "qwen2_5_omni"
                elif corruption == "duplicate_ids":
                    metadata["action_token_ids"][ACTION_TOKENS[0]] = metadata["action_token_ids"][ACTION_TOKENS[1]]
                else:
                    metadata["action_token_ids"][ACTION_TOKENS[0]] = True
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "metadata|model type"):
                    load_duplex_checkpoint(self.tmp_path)

    def test_load_requires_local_completed_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            load_duplex_checkpoint(self.tmp_path / "missing")
        with self.assertRaises(FileNotFoundError):
            load_duplex_checkpoint(self.tmp_path)

    def test_load_disallows_mismatched_weight_reinitialization(self):
        extend_action_tokens(self.model, self.processor.tokenizer)
        save_duplex_checkpoint(self.model, self.processor, self.tmp_path)
        with self.assertRaisesRegex(ValueError, "reinitialize"):
            load_duplex_checkpoint(self.tmp_path, ignore_mismatched_sizes=True)

    def test_validation_missing_special_and_non_atomic_actions(self):
        model, processor = self.model, self.processor
        with self.assertRaisesRegex(ValueError, "Missing action"):
            validate_action_tokens(model, processor.tokenizer)
        extend_action_tokens(model, processor.tokenizer)
        processor.tokenizer.split_special_tokens = True
        with self.assertRaisesRegex(ValueError, "atomically"):
            validate_action_tokens(model, processor.tokenizer)
        processor.tokenizer.split_special_tokens = False
        processor.tokenizer.additional_special_tokens = [ACTION_TOKENS[0]]
        with self.assertRaisesRegex(ValueError, "Missing action"):
            validate_action_tokens(model, processor.tokenizer)

    def test_validation_rejects_inconsistent_dimensions(self):
        for corruption in ("head_rows", "head_hidden", "config", "tokenizer_range", "composite_config"):
            with self.subTest(corruption=corruption):
                model, processor = make_thinker(), make_processor()
                extend_action_tokens(model, processor.tokenizer)
                if corruption == "head_rows":
                    model.lm_head = torch.nn.Linear(16, 29, bias=False)
                elif corruption == "head_hidden":
                    model.lm_head = torch.nn.Linear(15, 30, bias=False)
                elif corruption == "config":
                    model.config.text_config.vocab_size += 1
                elif corruption == "tokenizer_range":
                    processor.tokenizer.add_tokens(["new_out_of_range_word"])
                else:
                    model = CompositeWithUntouchableTalker(model)
                    model.config.thinker_config.text_config.vocab_size -= 1
                with self.assertRaisesRegex(ValueError, "dimensions|config|exceeds"):
                    validate_action_tokens(model, processor.tokenizer)


if __name__ == "__main__":
    unittest.main()
