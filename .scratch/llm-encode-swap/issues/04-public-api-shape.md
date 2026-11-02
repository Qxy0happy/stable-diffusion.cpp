# Public API shape for encode/generate separation

Type: grilling
Status: resolved

## Question

Given the findings of tickets 01-03, design the public C API additions in `include/stable-diffusion.h` for (A) encoding a prompt into a reusable conditioning handle and (B) generating an image from a precomputed conditioning, plus (C) any flags for sequential model residency. Grilling session to settle names, ownership/lifetime, and whether encode and generate can share one context.

## Answer

### Decisions (settled 2026-11-02)

**D1. Handle ownership (A from grilling).**
- Add `typedef struct sd_condition_t sd_condition_t;` (opaque).
- `sd_condition_t* sd_encode_prompt(sd_ctx_t* ctx, const char* prompt, const sd_img_gen_params_t* params);`
  - Returns a library-owned handle; user frees with `sd_free_condition(sd_condition_t*)`.
  - Handle wraps `SDCondition` (host-RAM `sd::Tensor<float>` fields: `c_crossattn`, `c_vector`, `c_concat`, id tensors, embeds) plus the uncond/img_uncond variants needed by CFG. No ggml handle embedded, so it is backend-agnostic and survives TE eviction.
  - Lifetime: bound to the creating context; must be freed before `free_sd_ctx` (or freed by context teardown).

**D2. Single context, sequential residency (A from grilling).**
- Encode and generate share one `sd_ctx_t`.
- Flow: batch-encode all prompts → `sd_encode_prompt` for each (TE on CPU, cached in RAM as `sd_condition_t*`) → call `sd_end_runners(ctx)` (or a new `sd_unload_text_encoder(ctx)`) to evict TE params (via `GGMLRunner::runner_end` → `evict_compute_backend_params`; requires TE registered with `ResidencyMode::Disk` OR CPU params that become evictable — see caveat below) → generate each image from its cached handle → TE reloads lazily on next encode.
- Caveat (from ticket 02): CPU-resident `ParamBackend` TE (the default 4.9 GB case) cannot be evicted by `evict_compute_backend_params` (demotion rejects CPU). To actually drop TE memory, TE must be registered as `ResidencyMode::Disk` at context creation (`--params-backend te=disk`) OR we add an explicit `unregister_param_tensors(ModelComponent::Conditioner)` + re-register path. Decision: expose `sd_unload_text_encoder(ctx)` that unregisters TE tensors and frees their params blocks; re-register lazily on next encode. This is the only way to free the 4.9 GB.

**D3. CLI batch flow (in-RAM cache, user override).**
- User clarified: cache stays in RAM (not files); even 100 prompts = a few hundred MB of conditioning. No file-based handoff.
- `--prompt-file`: change semantics to split lines into a `std::vector<std::string>` of prompts (currently slurps whole file into one prompt). Keep backward compat? Decision: split on `\n`, skip blanks, one image per prompt.
- New behavior for batch: single process, two phases:
  1. Encode phase: for each prompt, `sd_encode_prompt` → store `sd_condition_t*` in a `std::vector`. TE resident (CPU).
  2. Unload phase: `sd_unload_text_encoder(ctx)` (frees 4.9 GB).
  3. Generate phase: for each cached condition, `sd_generate_image_from_condition(ctx, cond, params)` → VAE decode → save. DiT+VAE on GPU.
- `--batch-count` keeps meaning: N samples per prompt (seeds via `seed_for_batch`).
- Conditioning cache: raise default or let `--conditioning-cache-size` control the LRU; the new in-RAM handle vector is the explicit batch store and bypasses the LRU for the batch lifetime.
- ADetailer (`detailer.cpp`) also benefits: reuse cached cond instead of re-encoding per region.

**D4. Generate-from-conditioning API.**
- `sd_image_t* sd_generate_image_from_condition(sd_ctx_t* ctx, sd_condition_t* cond, const sd_img_gen_params_t* params);` (or extend `sd_img_gen_params_t` with an optional `sd_condition_t* condition` override). Decision: add optional field `sd_condition_t* condition` to `sd_img_gen_params_t` (NULL = encode from `prompt` as today). This keeps one entry point and is backward compatible.
- `sd_condition_t*` must be freed by caller after all generations using it complete.

**D5. Memory reporting.**
- Log phase transitions (encode done, TE unloaded, generate start) so users see the memory drop.

### Implementation tickets to create next (task, not grilling)

- T5: Add `sd_condition_t` opaque type + `sd_encode_prompt` / `sd_free_condition` to `include/stable-diffusion.h` and `src/stable-diffusion.cpp`.
- T6: Implement `sd_unload_text_encoder` (unregister TE tensors + free params blocks) in `diffusion_engine`.
- T7: Add `condition` field to `sd_img_gen_params_t`; branch in `sd::pipeline::generate_image` to skip encode when condition provided.
- T8: Refactor `prepare_image_generation_embeds` to accept optional precomputed cond (from ticket 03 seam).
- T9: Update `sd-cli` `--prompt-file` to split lines + two-phase batch encode/generate.
- T10: Tests/manual verification: 100-prompt batch, memory profile, quality parity vs baseline.

### Status

Ticket 04 resolved. Ready to create implementation task tickets or hand off.