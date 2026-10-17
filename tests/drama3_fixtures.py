"""Synthetic tokenizer and source config shared by prompt and API tests."""
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers


def make_prompt_assets(directory, vocabulary, *, normalizer=None):
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    if normalizer is not None:
        tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|audio_pad|>"]
    special += [f"<|speaker:{i}|>" for i in range(16)]
    special += [f"<|semantic:{i}|>" for i in range(4)]
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in special])
    tokenizer.save(str(directory / "tokenizer.json"))
    config = {
        "model_type": "fish_omni",
        "audio_decoder_config": {"num_codebooks": 10, "vocab_size": 4},
        "semantic_start_token_id": tokenizer.token_to_id("<|semantic:0|>"),
        "semantic_end_token_id": tokenizer.token_to_id("<|semantic:3|>"),
        "eos_token_id": tokenizer.token_to_id("<|im_end|>"),
        "audio_pad_token_id": tokenizer.token_to_id("<|audio_pad|>"),
    }
    return tokenizer, config
