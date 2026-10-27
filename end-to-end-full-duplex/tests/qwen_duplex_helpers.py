"""Small real native Omni fixture; no downloads or pretrained weights."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration


def tiny_omni():
    words = ["<unk>", "<bos>", "<eos>", "hello", "<audio_bos>", "<audio_eos>",
             "<audio>", "<image>", "<video>"] + [f"word{i}" for i in range(23)]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>",
                                       eos_token="<eos>", additional_special_tokens=words[4:9])
    config = Qwen2_5OmniThinkerConfig(
        text_config=dict(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                         bos_token_id=1, eos_token_id=2, pad_token_id=0),
        audio_config=dict(num_mel_bins=8, encoder_layers=1, encoder_attention_heads=2,
                          encoder_ffn_dim=32, d_model=16, output_dim=32),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           patch_size=2, temporal_patch_size=2, out_hidden_size=32,
                           window_size=8, fullatt_block_indexes=[0]),
        audio_token_index=6, image_token_index=7, video_token_index=8,
        audio_start_token_id=4, audio_end_token_id=5,
    )
    config._attn_implementation = "eager"
    return Qwen2_5OmniThinkerForConditionalGeneration(config).eval(), tokenizer
