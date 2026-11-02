# T9: sd-cli two-phase batch (encode all → unload TE → generate all)

Type: task
Status: resolved
Blocked by: 05, 06, 07, 08

## Task

Modify `examples/cli/main.cpp` + `common.cpp`: (1) change `--prompt-file` to split lines into a vector of prompts; (2) encode phase: sd_encode_prompt for each prompt into in-RAM vector; (3) unload phase: sd_unload_text_encoder; (4) generate phase: for each cached condition call sd_generate_image_from_condition (via params.condition) + save. Log phase transitions and memory. `--batch-count` keeps per-prompt N samples with seed_for_batch.

## Acceptance

- `sd-cli --prompt-file prompts.txt ...` produces one image per prompt; peak RAM < 6GB after unload; no quality regression vs single-stage.

## Answer

Landed in `32e8b8a2` (merged in `2f00a13f`, hardened in `b71effb5`).

- `--prompt-file` now reads one prompt per line, drops blank/whitespace-only lines and trailing `\r`, and errors on a file with no prompts. A single-line file behaves like before.
- `SDGenerationParams::prompts` carries the list; per-prompt `<lora:...>` tags are extracted into per-prompt LoRA vectors (`prompt_lora_vecs`, path strings owned by `prompt_lora_maps`), and the CLI applies them to both the encode and generate call for that prompt. A tag on line *i* applies only to image *i*.
- CLI two-phase flow in `examples/cli/main.cpp`: encode all prompts via `sd_encode_prompt` (logging `encode phase: N prompts in X.Xs`) → `sd_unload_text_encoder` (`text encoder unloaded`) → generate each handle via `params.condition` (`generate phase: prompt i/N`), freeing handles as it goes and on every error path.
- `--batch-count` keeps per-prompt N samples with the existing `seed_for_batch` seeds; output indices stay unique across prompts and batches.
- Precedence: `--prompt-file` wins over `-p` (the file handler clears `prompt`), matching the previous argv-order-agnostic behaviour of the single-shot flag.

Verified: probe run over a 5-line file with blank and whitespace-only lines produced `prompt-file parsed 3 prompts`; the 100-prompt batch produced exactly 100 PNGs; single-stage parity is byte-identical (see T8/T10).
