# T6: Implement sd_unload_text_encoder (free TE params)

Type: task
Status: resolved
Blocked by: 02, 04

## Task

Add `void sd_unload_text_encoder(sd_ctx_t* ctx)` to public API. Implementation: call `StableDiffusionGGML::unregister_param_tensors(ModelComponent::Conditioner)` (or equivalent) to drop TE tensors and free their params blocks, so the 4.9GB LLM memory is released. Re-register lazily on next `sd_encode_prompt` via existing `prepare_params` reload path. Must not break conditioning cache (host-RAM SDCondition copies survive).

## Acceptance

- After unload, RSS drops by ~4.9GB; next encode reloads TE and succeeds.

## Answer

Landed in `c4a5e8b1` (merged in `88e03f60`).

- `StableDiffusionGGML::unload_text_encoder()` runs under the context's `ContextOperation` guard, calls `end_runners()` so nothing is pinned/staged, then `model_manager->unregister_param_tensors(ModelComponent::Conditioner, &released)`; a second call is a no-op; the released byte count is logged.
- Reload is lazy: `ensure_text_encoder_registered()` re-registers through the same `register_runner_params(..., SDBackendModule::TE, ...)` path used at init and guards `get_learned_condition` (the single TE entry funnel).
- `conditioning_cache_` and the conditioner object are left intact — host-RAM `SDCondition` copies and the graph code survive eviction.

Verified with a throwaway library harness (`D:/Cache/Agents/sdcpp-smoke/reload/reload.cpp`, CPU build): `encode #1 -> PASS`, `sd_unload_text_encoder returned`, `encode #2 after unload -> PASS` (all 426 TE tensors reloaded from the model file), `free_sd_ctx -> PASS`.
RSS evidence from the 100-prompt batch (see T10): peak during encode 5.21GB, 1.03GB immediately after the `text encoder unloaded` phase marker, peak during generate 5.16GB.
