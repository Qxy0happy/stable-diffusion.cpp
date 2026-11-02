# T5: Add sd_condition_t opaque handle + sd_encode_prompt / sd_free_condition

Type: task
Status: open
Blocked by: 04

## Task

Add to `include/stable-diffusion.h`: `typedef struct sd_condition_t sd_condition_t;` and declarations `sd_condition_t* sd_encode_prompt(sd_ctx_t* ctx, const char* prompt, const sd_img_gen_params_t* params)` and `void sd_free_condition(sd_condition_t* cond)`. Implement in `src/stable-diffusion.cpp`: wrap `StableDiffusionGGML::get_learned_condition` to build a handle owning `SDCondition` (c_crossattn, c_vector, c_concat, id tensors, embeds) plus uncond/img_uncond variants. Handle must be freed before ctx teardown.

## Acceptance

- Header compiles; `sd_encode_prompt` returns non-null for a valid prompt; `sd_free_condition` frees without leak (ASan/valgrind optional).
