# T7: Add optional condition field to sd_img_gen_params_t

Type: task
Status: open
Blocked by: 04

## Task

Add `sd_condition_t* condition` (NULL = encode from prompt as today) to `sd_img_gen_params_t` in `include/stable-diffusion.h`. Keep backward compatible (default NULL). Document ownership: caller frees after all generations using it.

## Acceptance

- Existing callers compile unchanged; setting condition skips prompt encoding.
