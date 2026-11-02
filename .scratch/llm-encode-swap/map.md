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

<!-- one line per closed ticket: [title](link): gist -->

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