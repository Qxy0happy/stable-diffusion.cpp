# T7: Add optional condition field to sd_img_gen_params_t

Type: task
Status: resolved
Blocked by: 04

## Task

Add `sd_condition_t* condition` (NULL = encode from prompt as today) to `sd_img_gen_params_t` in `include/stable-diffusion.h`. Keep backward compatible (default NULL). Document ownership: caller frees after all generations using it.

## Acceptance

- Existing callers compile unchanged; setting condition skips prompt encoding.

## Answer

Landed in `c60bfba1` (merged in `88e03f60`).

- `sd_condition_t* condition` appended as the LAST field of `sd_img_gen_params_t` (append-only), documented: NULL encodes from `prompt`; when set, `prompt`/`negative_prompt` are not used for encoding, the handle is caller-owned and must outlive the call and be freed with `sd_free_condition` after the last generation using it.
- `sd_img_gen_params_init` zeroes the field; the CLI's `sd_img_gen_params_t{}` construction and every reinit site route through it, so existing callers keep encoding-from-prompt behaviour.
- Consume-side branching is T8 (`f1c0bde0`).
