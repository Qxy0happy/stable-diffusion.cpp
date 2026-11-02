# T5: Add sd_condition_t opaque handle + sd_encode_prompt / sd_free_condition

Type: task
Status: resolved
Blocked by: 04

## Task

Add to `include/stable-diffusion.h`: `typedef struct sd_condition_t sd_condition_t;` and declarations `sd_condition_t* sd_encode_prompt(sd_ctx_t* ctx, const char* prompt, const sd_img_gen_params_t* params)` and `void sd_free_condition(sd_condition_t* cond)`. Implement in `src/stable-diffusion.cpp`: wrap `StableDiffusionGGML::get_learned_condition` to build a handle owning `SDCondition` (c_crossattn, c_vector, c_concat, id tensors, embeds) plus uncond/img_uncond variants. Handle must be freed before ctx teardown.

## Acceptance

- Header compiles; `sd_encode_prompt` returns non-null for a valid prompt; `sd_free_condition` frees without leak (ASan/valgrind optional).

## Answer

Landed in `a2626cb7` (merged to `spec/llm-encode-swap` in `88e03f60`).

- `include/stable-diffusion.h`: opaque `sd_condition_t` (forward declaration before `sd_img_gen_params_t`), `sd_encode_prompt`, `sd_free_condition`, with ownership/lifetime comments.
- `src/pipeline/condition_handle.h`: private definition owning `sd::pipeline::ImageGenerationEmbeds` (cond/uncond/img_uncond) plus the encode dimensions and the `size_dependent` flag used for ADetailer reuse.
- `prepare_image_generation_embeds` is no longer file-static and is declared in `src/pipeline/generation.h`, so encode and generate share one implementation — no duplicated encode path.
- `sd_encode_prompt` runs inside `ExecutionScope`, applies the params' LoRAs/circular axes, resolves the ref-image params, encodes with empty latents (so latents-dependent `c_concat` stays unfilled), and registers the handle in `sd_ctx_t::live_conditions`; `free_sd_ctx` frees leftovers and `sd_free_condition(NULL)` is a no-op.

Verified: CPU build green (`sd-cli`, `sd-server`); library harness (`D:/Cache/Agents/sdcpp-smoke/reload/reload.cpp`) encodes a prompt, unloads the TE, encodes again, and frees both handles; image output is byte-identical to the single-stage path (see T10).
