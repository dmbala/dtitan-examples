# Debug tokenizer (vocab 2016)

`tokenizer.json` / `tokenizer_config.json` are the small test tokenizer vendored from
[pytorch/torchtitan](https://github.com/pytorch/torchtitan) (`tests/assets/tokenizer/`, BSD-3-Clause).

**Why it's here:** the `llama3` **`debugmodel`** flavor hardcodes `vocab_size=2048`, and
torchtitan 0.2.2 does **not** resize the embedding from the tokenizer
(`TransformerModelArgs.update_from_config` only sets `max_seq_len`). So the debug model
can only be paired with a tokenizer whose vocab ≤ 2048. This tokenizer (vocab **2016**) is
that match and is what the Level 1 debug/fast-path labs use:

```
--model.hf_assets_path=assets/test_tokenizer
```

The **real Llama-3.1 tokenizer** (`$MODELS/Llama-3.1-8B-Instruct`, vocab 128256) is only valid
with a real-vocab flavor (`--model.flavor 8B`) — use it for the "real-model taste", not the
debug model (a 128K-vocab tokenizer overflows the debug model's embedding → CUDA gather OOB).
