# T9: sd-cli two-phase batch (encode all → unload TE → generate all)

Type: task
Status: open
Blocked by: 05, 06, 07, 08

## Task

Modify `examples/cli/main.cpp` + `common.cpp`: (1) change `--prompt-file` to split lines into a vector of prompts; (2) encode phase: sd_encode_prompt for each prompt into in-RAM vector; (3) unload phase: sd_unload_text_encoder; (4) generate phase: for each cached condition call sd_generate_image_from_condition (via params.condition) + save. Log phase transitions and memory. `--batch-count` keeps per-prompt N samples with seed_for_batch.

## Acceptance

- `sd-cli --prompt-file prompts.txt ...` produces one image per prompt; peak RAM < 6GB after unload; no quality regression vs single-stage.
