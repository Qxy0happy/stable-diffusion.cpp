# T6: Implement sd_unload_text_encoder (free TE params)

Type: task
Status: open
Blocked by: 02, 04

## Task

Add `void sd_unload_text_encoder(sd_ctx_t* ctx)` to public API. Implementation: call `StableDiffusionGGML::unregister_param_tensors(ModelComponent::Conditioner)` (or equivalent) to drop TE tensors and free their params blocks, so the 4.9GB LLM memory is released. Re-register lazily on next `sd_encode_prompt` via existing `prepare_params` reload path. Must not break conditioning cache (host-RAM SDCondition copies survive).

## Acceptance

- After unload, RSS drops by ~4.9GB; next encode reloads TE and succeeds.
