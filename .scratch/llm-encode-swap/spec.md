Status: ready-for-agent

# Spec: LLM batch encode + sequential model residency

## Problem Statement

For batch image generation in stable-diffusion.cpp, every prompt pays a ~2s LLM (Qwen3.5-9B) encode on the CPU before diffusion can start, and the LLM stays resident (4.9 GB) alongside the diffusion model (4.4 GB) for the whole run, peaking ~9.4 GB RAM. On the 8 GB VRAM target, the text encoder cannot be moved to the GPU, so encoding is the bottleneck for large batches and memory pressure is high. There is no way to precompute and reuse prompt conditioning, nor to drop the text encoder between the encode and generate phases.

## Solution

Expose a public C API to (1) encode a prompt into a reusable, library-owned conditioning handle, (2) generate an image from a precomputed conditioning handle, and (3) unload the text encoder to free its memory. Add an optional `condition` field to the existing image-generation params so a single entry point covers both encode-from-prompt and generate-from-condition. Refactor `sd-cli`'s prompt-file batch flow into two in-memory phases: encode all prompts (LLM on CPU, conditioning cached in RAM), unload the text encoder, then generate all images from cached conditioning (DiT+VAE on GPU). Conditioning handles are host-RAM `sd::Tensor` copies, backend-agnostic, and survive text-encoder eviction. ADetailer reuses cached conditioning instead of re-encoding per region.

## User Stories

1. As a batch user, I want to encode all prompts up front on the CPU, so that the LLM cost is paid once per unique prompt.
2. As a batch user, I want the text encoder unloaded before diffusion starts, so that peak RAM stays near the diffusion model size (~4.4 GB) instead of 9.4 GB.
3. As a batch user, I want to generate images from cached conditioning, so that diffusion runs on the GPU without waiting on the CPU encoder.
4. As a library user, I want an opaque conditioning handle I can free, so that I can manage lifetime explicitly.
5. As a library user, I want to pass a precomputed condition into the existing generate call, so that I don't need a second entry point.
6. As a library user, I want the handle to survive text-encoder eviction, so that cached embeddings remain valid after unload.
7. As an ADetailer user, I want region inpainting to reuse cached conditioning, so that detection+inpaint loops don't re-encode the prompt.
8. As a CLI user, I want `--prompt-file` to process one prompt per line, so that batch jobs are easy to author.
9. As a CLI user, I want phase-transition logs showing encode done / TE unloaded / generate start, so that I can observe memory behavior.
10. As a CLI user, I want `--batch-count` to still produce N samples per prompt with deterministic seeds, so that existing batch semantics are preserved.
11. As a developer, I want the internal `prepare_image_generation_embeds` seam to accept an optional precomputed condition, so that the generate path can skip encoding cleanly.
12. As a developer, I want text-encoder params to reload lazily on the next encode, so that a second batch in the same process works.

## Implementation Decisions

- **Modules modified**: public C API header and implementation, diffusion engine (context/runners/residency), image pipeline, conditioning cache, ADetailer, CLI common/main.
- **Opaque handle**: introduce `sd_condition_t` owning `SDCondition` (host-RAM tensors: cross-attention, vector, concat, id tensors, image/audio embeds, reference blocks, extra cross-attentions) plus the uncond and img_uncond variants needed for CFG. No ggml handle is embedded, so the handle is backend-agnostic and survives text-encoder eviction.
- **Encode API**: `sd_encode_prompt(ctx, prompt, params)` returns a library-owned handle; `sd_free_condition(cond)` frees it. Handle lifetime is bound to the creating context.
- **Generate API**: add an optional `condition` field (NULL = encode from `prompt` as today) to the image-generation params struct. When set, skip `get_learned_condition` and use the cached condition. This keeps a single entry point and is backward compatible.
- **Unload API**: `sd_unload_text_encoder(ctx)` unregisters text-encoder tensors and frees their params blocks, releasing the ~4.9 GB. Re-registration happens lazily on the next encode via the existing `prepare_params` reload path. Cached conditioning (host RAM) is unaffected.
- **Residency prerequisite**: text encoder must be evictable. If it is registered as CPU-resident params backend (default), eviction is blocked; the unload API performs an explicit unregister + free, bypassing the eviction gate. On reload, tensors re-register and `prepare_params` repopulates them.
- **Internal seam**: `prepare_image_generation_embeds` accepts an optional precomputed condition; when provided it skips the encode and uses it directly. This is the single internal seam for condition injection.
- **CLI two-phase in-RAM batch**: (1) encode phase — read prompt file into a vector of prompts, `sd_encode_prompt` each, store handles in a vector; (2) unload phase — `sd_unload_text_encoder`; (3) generate phase — for each cached handle set `params.condition` and generate + save. No file-based handoff; conditioning stays in RAM (a few hundred MB for hundreds of prompts).
- **Prompt-file semantics**: split file on newlines, skip blank lines, one image per prompt. `--batch-count` retains per-prompt N samples with deterministic seeds.
- **ADetailer**: reuse cached conditioning per region instead of re-encoding.
- **Logging**: phase transitions logged so memory behavior is observable.
- **Memory contract**: peak during generate phase is diffusion + VAE + conditioning cache, not LLM.

## Testing Decisions

- Test external behavior only, not implementation details.
- Seam 1 (public C API): `sd_encode_prompt` returns non-null for valid prompt; handle frees without leak; `sd_generate_image` with `condition` set produces an image matching the baseline (same seed) within tolerance; `sd_unload_text_encoder` reduces resident memory and next encode succeeds.
- Seam 2 (internal): `prepare_image_generation_embeds` with a precomputed condition skips encoding and yields identical embeds to the encode path.
- Seam 3 (CLI): end-to-end batch run produces one image per prompt; peak RSS during generate phase < 6 GB; quality parity documented.
- Prior art: existing manual smoke tests in `examples/cli/README.md` and `docs/moss_image.md` examples; determinism checks via fixed seed + SHA-256 comparison (per `CONTRIBUTING.md` verification practice). No automated C++ test suite exists (`SD_BUILD_TESTS` commented out); verification is manual with reported commands/outputs.
- No committed test code (per `CONTRIBUTING.md`); keep tests local.

## Out of Scope

- New model formats or quantization schemes (reuse existing Q4_K/Q6_K).
- GPU offload of the LLM (Vulkan 780M tested, slower and unstable).
- Changes to ggml or thirdparty.
- Cross-process embedding handoff (in-RAM only).
- Overlapping encode (CPU) with diffusion (GPU) — context is serialized via ExecutionScope; not addressed here.
- Automated C++ unit tests (no test framework in repo).

## Further Notes

- Conditioning handles are host-RAM and backend-agnostic because `SDCondition` is `std::vector`-based (`sd::Tensor<T>`), so they survive text-encoder eviction and are safe to reuse across the batch.
- The text encoder must be explicitly unregistered (not just evicted) to free the 4.9 GB, because CPU-resident params are excluded from the normal eviction path.
- `conditioning_cache_size` LRU remains for repeated single-prompt calls; the new in-RAM handle vector is the explicit batch store and bypasses the LRU for the batch lifetime.