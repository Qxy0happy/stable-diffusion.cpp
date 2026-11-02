# Map: LLM batch encode + sequential model residency

## Destination

Implement (A) batch LLM prompt encoding that produces reusable embedding vectors stored in RAM, and (B) sequential model residency so only one large model is resident at a time, reducing peak memory. Deliver as library API extensions in `stable-diffusion.cpp` plus `sd-cli` support (prompt-file batch flow). Scope: code changes in `src/` and `examples/cli/`; no new model formats.

## Notes

- Repo: stable-diffusion.cpp (C/C++17, CMake). Public API: `include/stable-diffusion.h`. Core engine: `src/pipeline/diffusion_engine.{h,cpp}`. Batch entry: `examples/cli/main.cpp` + `examples/common/common.cpp`.
- Skills to consult: `codebase-design` (seams/testability), `domain-modeling`.
- LLM is Qwen3.5 9B (~4.9GB). DiT Q6_K ~4.4GB. VAE ~0.12GB. Peak today ~9.4GB RAM (all resident).
- Existing `conditioning_cache_size` (default 4) caches conditioning per context; only helps repeated prompts.
- No public encode-prompt / generate-from-conditioning API exists today.
- Constraint: keep CUDA DiT+VAE on GPU (Q6_K fits 8GB with workspace); LLM stays on CPU.

## Decisions so far

- [Public API shape for encode/generate separation](issues/04-public-api-shape.md): opaque `sd_condition_t*` handle owned by lib; single context with `sd_unload_text_encoder` to drop TE; `sd_img_gen_params_t.condition` optional field for generate-from-condition; CLI two-phase in-RAM batch (no file handoff); ADetailer reuses cached cond.
- Spec published: [spec.md](spec.md) (status: ready-for-agent). Consolidates decisions from tickets 01-04 into implementation-ready spec.
- Implemented on `spec/llm-encode-swap`: T5 `a2626cb7`, T6 `c4a5e8b1`, T7 `c60bfba1`, T8 `f1c0bde0`, T9 `32e8b8a2`, review fixes `b71effb5`, per-prompt LoRA lifetime fix `d10669a0`; merges `88e03f60` (05-07) and `2f00a13f` (08-09). All tickets 05-10 resolved with verification in their `## Answer` sections.

## Not yet specified

- How conditioning/embeddings are represented internally and whether they can be exported/serialized.
- Whether ModelManager can evict TE params and lazily reload within a live context.
- CLI UX: new flags vs new mode; how prompt-file batch feeds cached embeddings.
- Serialization format for cross-process embedding handoff (if any).
- Whether to keep a single context or split encode-context / generate-context.
- Concurrency: can encode (CPU) overlap with diffusion (GPU)? Context is serialized via ExecutionScope.

## Out of scope

- New model formats / quantization schemes (reuse existing Q4_K/Q6_K).
- GPU offload of LLM (Vulkan 780M tested, slower + unstable).
- Changing ggml or thirdparty.

## Tickets

- `issues/01-conditioning-representation.md` — research (resolved)
- `issues/02-modelmanager-te-eviction.md` — research (resolved)
- `issues/03-sdcli-batch-prompt-flow.md` — research (resolved)
- `issues/04-public-api-shape.md` — grilling (resolved)
- `issues/05-add-sd-condition-handle-api.md` — task (T5)
- `issues/06-implement-sd-unload-text-encoder.md` — task (T6)
- `issues/07-add-condition-field-to-gen-params.md` — task (T7)
- `issues/08-branch-generate-on-condition.md` — task (T8)
- `issues/09-sdcli-two-phase-batch.md` — task (T9)
- `issues/10-verify-batch-memory-and-quality.md` — task (T10)