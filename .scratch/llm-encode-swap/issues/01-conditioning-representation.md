# Conditioning representation in stable-diffusion.cpp

Type: research
Status: open

## Question

How is prompt conditioning (text embeddings) represented internally in `StableDiffusionGGML` and the pipeline, and can it be extracted/serialized/exported for reuse across a separate generation call? Identify the relevant structs, where they're stored, and whether they're tied to the context lifetime. Also note whether `sd_image_generation_result_t` or any public struct carries conditioning.

## Answer

**TL;DR:** Prompt conditioning is a plain value type, `SDCondition`, produced by
`StableDiffusionGGML::get_learned_condition(const ConditionerParams&)` and consumed by
`StableDiffusionGGML::sample(...)`. All its members are host-side `sd::Tensor<T>` (i.e.
`std::vector<T>` + shape), so the data itself is copyable and context-independent, but the only
place it is kept between calls is the context-owned `ConditioningCache`; nothing exposes it
through the C API and nothing serializes it.

### 1. Structs that hold conditioning

`src/conditioning/conditioner.hpp`

| Symbol | Line | Notes |
|---|---|---|
| `struct SDCondition` | 25 | the conditioning container (below) |
| `struct ConditionerParams` | 132 | *input* to encoding + cache key: `text`, `clip_skip`, `width`, `height`, `zero_out_masked`, `const std::vector<sd::Tensor<float>>* ref_images`, `const std::vector<MiniMaxH3PresentationItem>* minimax_h3_references`, `RefImageParams ref_image_params` |
| `struct Conditioner` | 143 | abstract text-encoder interface: `virtual SDCondition get_learned_condition(int n_threads, const ConditionerParams&) = 0` (line 148), plus `runner_end()`, weight/backend plumbing |
| `struct FrozenCLIPEmbedderWithCustomWords` | 165 | CLIP/SD1/SD2/SDXL encoder; also exposes `get_learned_condition_common(...)` (line 453) used by PhotoMaker |
| other conditioners | — | `SD3CLIPEmbedder`, `FluxCLIPEmbedder`, `T5CLIPEmbedder`, `MiniT2IConditioner`, `SenseNovaU1Conditioner`, `AnimaConditioner`, `LLMEmbedder`, `MingImageEmbedder`, `LLaDAImageEmbedder`, `LTXAVEmbedder`; plus `HiDreamO1Conditioner` in `src/model/diffusion/hidream_o1.hpp`. Selected per model/version in `src/model/integrations/*.cpp` (`result.conditioner = std::make_shared<...>`), aggregated in `src/pipeline/model_builders.{h,cpp}` (`CoreRunners::conditioner`) |

`SDCondition` members (all default-constructed empty):

- `c_crossattn`, `c_vector`, `c_concat` — `sd::Tensor<float>` (the core text embeddings)
- `c_t5_ids`, `c_t5_weights` — `int32_t` / `float` token streams for T5-style token-conditioned DiTs (Anima)
- `c_input_ids`, `c_position_ids`, `c_token_types`, `c_vinput_mask` — `sd::Tensor<int32_t>` for LLM-token-based models (SenseNovaU1, HiDreamO1, Qwen-Image-2.1, MiniMax-H3)
- `c_image_embeds` — `std::vector<std::pair<int, sd::Tensor<float>>>`
- `c_ref_images`, `c_ref_audios` — `std::vector<sd::Tensor<float>>`
- `c_reference_blocks` — `std::vector<MiniMaxH3ReferenceBlock>`
- `extra_c_crossattns` — `std::vector<sd::Tensor<float>>` (Ming-Image, LLaDA, HunyuanVideo)
- `empty()` (line 49), 3-arg convenience ctor (line 44); copyable/movable by default

Field *meaning* is model-specific: for CLIP/SD1/SD2 `c_crossattn` is the last hidden states and
`c_vector` is left empty (it is only populated for SDXL); for SDXL `c_vector` is the pooled embedding concatenated with
height/width timestep embeddings to 2816 dims (`conditioner.hpp:541-563`), which is why `width`/
`height` are part of `ConditionerParams`; for T5 conditioners `c_vector` is the attention mask
(`conditioner.hpp:1616-1619`); token-id models fill only the `c_*_ids` fields (e.g.
`SenseNovaU1Conditioner::tokenize_condition`, line 1795); Ming-Image puts the second half of the
LLM hidden states in `extra_c_crossattns` (line 3376).

`src/conditioning/conditioning_cache.h`

- `class ConditioningCache` (line 10) with nested `struct Entry` (line 11):
  `{ ConditionerParams params; std::vector<sd::Tensor<float>> ref_images; std::vector<MiniMaxH3PresentationItem> references; SDCondition condition; }`.
  The ctor (line 17) *copies* request-owned `ref_images` / `minimax_h3_references` into the entry so
  the cached `params` pointers do not dangle once the request dies.
- LRU (`std::list<Entry>`, capacity default 4); `same_params` (line 45) keys on
  `text, clip_skip, width, height, zero_out_masked`, the `RefImageParams` fields, and full
  pixel-equality of ref images/reference frames.
- `get()` (line 85) returns `SDCondition` **by value** (deep copy of the vectors), logs
  `"conditioning cache hit"`, and is bypassed when `capacity_ == 0`.

Pipeline-side aggregate: `sd::pipeline::ImageGenerationEmbeds` = `{ SDCondition cond; SDCondition uncond; SDCondition img_uncond; }`
(`src/pipeline/generation.h:39`), produced by `prepare_image_generation_embeds`
(`src/pipeline/image.cpp:471`) and `prepare_video_generation_embeds` (`src/pipeline/video.cpp:1195`).

### 2. Where it is stored in `StableDiffusionGGML`

- `std::shared_ptr<Conditioner> cond_stage_model;` — `src/pipeline/diffusion_engine.h:70`
  (public, like the whole class); rebound in `build_core_runners()`
  (`src/pipeline/diffusion_engine.cpp:1087`), reset in `reset_runners()` (`:184`).
- `std::unique_ptr<ConditioningCache> conditioning_cache_;` — `src/pipeline/diffusion_engine.h:199`,
  constructed in the ctor (`diffusion_engine.cpp:99`). This is the **only persistent conditioning
  store**. Capacity comes from `sd_ctx_params_t::conditioning_cache_size`
  (`include/stable-diffusion.h:256`, wired at `diffusion_engine.cpp:925`);
  documented in `docs/caching.md` (default 4 for C API/server, 0 for `sd-cli`).
- Entry point: `SDCondition StableDiffusionGGML::get_learned_condition(const ConditionerParams&)`
  — decl `diffusion_engine.h:386`, def `diffusion_engine.cpp:1868`, body
  `conditioning_cache_->get(*cond_stage_model, n_threads, params)`.
- Per-call only: locals in `prepare_image_generation_embeds` — `cond` (`image.cpp:497`),
  `uncond` (`:510-549`), `img_uncond` (`:551-577`) — moved into `ImageGenerationEmbeds`
  (`:582-585`), then passed by `const&` to `StableDiffusionGGML::sample(...)`
  (`diffusion_engine.cpp:2298`) and into `run_condition` (`:2534`), which fills
  `DiffusionParams.context / c_concat / y / ref_latents` plus version-specific `extra`
  (`src/model/diffusion/model.hpp:197`).
- Extension-owned: `PhotoMakerExtension::id_condition` (`src/extensions/photomaker_extension.cpp:134`),
  cleared by `reset_runtime_condition()` (`:206`) each generation and returned per step from
  `before_condition()` (`:311`).

### 3. Is it tied to context lifetime?

- **Memory:** no. `sd::Tensor<T>` is a `std::vector<T> data_` plus `shape_`
  (`src/core/tensor.hpp:73-121`) — pure host memory, deep-copied on value copies, no backend
  buffer or graph handle. Each forward copies it into the compute context via
  `GGMLRunner::make_input()` (`src/core/ggml_runner.h:312` → `set_backend_tensor_data`).
- **Ownership/lifetime:** yes, operationally. Both the encoder (`cond_stage_model`) and the cache
  live in `StableDiffusionGGML`; cache entries are dropped on `reset_runners()`
  (`diffusion_engine.cpp:169`, called from model updates) and whenever LoRA state changes/fails
  (`:1833, :1843, :1854`). Cache entries are *not* cleared between generations.
- **Validity:** values are version/model-specific (tensor shapes and the semantics of `c_vector`,
  `extra_c_crossattns`, token-id fields), so an `SDCondition` is only meaningful against the same
  model/version and text-encoder stack (including LoRA/embedding state) that produced it.

### 4. Any public struct carrying conditioning?

No.

- `sd_image_generation_result_t` (`include/stable-diffusion.h:281-284`) is
  `{ sd_image_result_t* items; int count; }`, and `sd_image_result_t` (`:275-279`) is
  `{ sd_image_t image; char* generation_info; }`. `generation_info` is the
  `sdcpp.generation/v1` JSON (`src/generation_info.cpp:95`) containing prompt text, negative
  prompt, dims, seed, sample params, models, LoRA list etc. — **no embeddings**.
- `sd_video_generation_result_t` (`:286-292`) likewise: frames + audio + `generation_info`.
- `sd_ctx_params_t` (`:203-259`) exposes only `conditioning_cache_size` (`:256`) and textual-inversion
  `embeddings`/`embedding_count` (`:222`, token-embedding files for CLIP conditioning, not
  precomputed conditioning).
- No `SD_API` function takes or returns embeddings/conditioning; `generate_image` /
  `generate_video` (`src/stable-diffusion.cpp:754`, `:780`) only wrap the pipeline and marshal
  images + JSON.

### Can it be extracted / serialized / reused across a separate generation call?

- **Already reused inside a context:** `ConditioningCache` returns the cached `SDCondition` for a
  matching `ConditionerParams`, so repeated `generate_image` / `generate_video` calls with the
  same prompt/negative prompt/dims/clip_skip and ref images skip text encoding. This is the only
  reuse mechanism, and it never leaves the process.
- **Export path does not exist:** no serialization, no handle, no C API surface. All
  `SDCondition` fields are plain `std::vector<float>` / `std::vector<int32_t>` behind
  `sd::Tensor<T>`, so a serializer is mechanically trivial (shape + dtype + values per field),
  but callers must also capture model identity (`SDVersion` + text-encoder identity + LoRA /
  embedding state, since any LoRA change invalidates the cache) and, for many fields, per-field
  semantics.
- **Fields that are request byproducts, not text encodings**, and cannot be naively transferred:
  `c_concat` (assigned from the call's `concat_latent` in `image.cpp:502-503` / `:546-548`),
  `c_ref_images` / `c_ref_audios` / `c_reference_blocks` / `c_position_ids`
  (`video.cpp:1219-1227`), `c_image_embeds`.
- **Insertion points for an external encoder:** implement `Conditioner::get_learned_condition`
  (then `ConditioningCache` keeps working unchanged), or replace
  `StableDiffusionGGML::cond_stage_model`; for per-step substitution,
  `GenerationExtension::before_condition` (`src/extensions/generation_extension.h:58`) and
  `step_cache.before_condition` (`diffusion_engine.cpp:2622`).

