# How does sd-cli batch/prompt-file flow encode prompts today?

Type: research
Status: open

## Question

Trace `examples/cli/main.cpp` and `examples/common/common.cpp` for `--prompt-file`, `--batch-count`, and how each prompt is encoded (LLM forward) and passed to `generate_image`. Identify where conditioning is computed, whether it's cached, and where a split encode/generate could hook in. Report the call chain and relevant functions.

## Answer

### Short version (premise correction)

There is no prompt-file *batch* flow today. `--prompt-file` is a single-shot convenience flag that copies one whole file into the one `std::string prompt` field; it does not split lines, does not append, and cannot be repeated usefully (last occurrence wins in argv order). `--batch-count` does not produce N prompts either: it repeats *sampling* N times from the **same** conditioning, using `GenerationRequest::seed_for_batch(b)` = `(uint64_t)seed + b` masked to `INT64_MAX` (`request.cpp:231-233`). So "each prompt is encoded" is today exactly *one* forward pass of the text encoder per `generate_image()` call (plus one for the negative prompt when CFG needs it), and `batch_count` reuses that result.

### 1. Call chain (CLI → library)

```
main()                                          examples/cli/main.cpp:636
├─ ctx_params.conditioning_cache_size = 0       examples/cli/main.cpp:644   ← cache disabled before parsing
├─ parse_args()                                 examples/cli/main.cpp:287
│   └─ parse_options()  (iterates argv in order) examples/common/common.cpp:262
│       ├─ "-p/--prompt"           → gen_params.prompt            common.cpp:1104
│       ├─ "--prompt-file"         → on_prompt_file_arg            common.cpp:1712 (registered 1839)
│       │     reads whole file: prompt = istreambuf_iterator...  common.cpp:1719
│       ├─ "--negative-prompt-file" → on_negative_prompt_file_arg  common.cpp:1730 (registered 1843)
│       ├─ "-b/--batch-count"      → gen_params.batch_count        common.cpp:1232
│       └─ "--conditioning-cache-size" → ctx_params.*              common.cpp:592
│   └─ ctx/gen resolve_and_validate()                              common.cpp:302-308
│       ├─ SDGenerationParams::resolve()   strips <lora:...> from prompt, snapshot prompt_with_lora
│       │                                                        common.cpp:2660
│       └─ SDGenerationParams::validate()  batch_count > 0        common.cpp:2675
├─ SDGenerationParams::to_sd_img_gen_params_t()                   common.cpp:2803
│     params.prompt      = prompt.c_str()                         common.cpp:2861
│     params.negative_prompt = negative_prompt.c_str()            common.cpp:2862
│     params.batch_count = batch_count                            common.cpp:2874
├─ new_sd_ctx(&sd_ctx_params)                                     main.cpp:891 → stable-diffusion.cpp:648
│   └─ StableDiffusionGGML::init → build_runners → build_core_runners
│        cond_stage_model (conditioner) + diffusion_model built together
│                                                                diffusion_engine.cpp:1082-1110
└─ generate_image(sd_ctx, &img_gen_params, &generated)            main.cpp:906
    └─ generate_image (C API)  ExecutionScope around call          stable-diffusion.cpp:754-778
        └─ sd::pipeline::generate_image                            image.cpp:834
            ├─ GenerationRequest request(sd, params)               request.cpp:123
            │     prompt/negative_prompt/batch_count/seed copied   request.cpp:124-131
            ├─ prepare_image_generation_latents(...)                image.cpp:883
            ├─ prepare_image_generation_embeds(...)                image.cpp:893  ← TEXT ENCODE, ONCE
            └─ for b in [0, batch_count): sd->sample(..., embeds.cond, embeds.uncond, ...)
                                                                   image.cpp:905-963
```

Key consequences of that shape:

- `gen_params.prompt` is a single string. `--prompt-file` and `--prompt` write the same field, so both flags are mutually exclusive in practice and argv order decides which wins (`common.cpp:262-345`).
- `save_results()` numbers outputs only *within* one batch: `out_<idx>` or a `%d` pattern (`main.cpp:516-527`, `561-566`). A multi-prompt run has no per-prompt naming concept today.
- The only other `generate_image` caller in the CLI path is ADetailer: `apply_adetailer` loops over results and calls `adetail_image` (`main.cpp:604-626`), which calls `generate_image` once per detected region/mask with `generation.batch_count = 1` and the ad prompt (`src/detailer.cpp:977-1010`). Each region therefore re-encodes its prompt from scratch.

### 2. Where conditioning is computed

For image generation there is exactly **one** text-encode site:

`prepare_image_generation_embeds()` — `src/pipeline/image.cpp:471-588` (file-local `static` in `namespace sd::pipeline`).

```
ConditionerParams condition_params;                 image.cpp:479-488
    text      = request->prompt
    clip_skip = request->clip_skip
    width/height = request->width/height
    ref_images = &latents->ref_images    (only if ref_image_params.pass_to_vlm)
sd->prepare_generation_extensions(...)              image.cpp:490
sd->compute_ip_adapter_tokens(...)                  image.cpp:494

cond  = sd->get_learned_condition(condition_params)         image.cpp:497
uncond = ...negative_prompt encode (CFG path)               image.cpp:540
img_uncond = ...reuse uncond or encode negative again       image.cpp:552-577

if (cond.c_concat.empty() && pass_to_dit) cond.c_concat = latents->concat_latent   image.cpp:502-504
```

It returns `ImageGenerationEmbeds { cond, uncond, img_uncond }` (`generation.h:39-43`), and `generate_image` computes it *before* the batch loop, so all `batch_count` samples share it. `StableDiffusionGGML::sample()` takes those as `const SDCondition&` (`diffusion_engine.h:449-474`), so there is no per-image copy of the embeddings. Two model-family shortcuts avoid the extra negative encode (ideogram4, MiniT2I, Ming at `image.cpp:512-526`).

Funnel function: `StableDiffusionGGML::get_learned_condition()` (`diffusion_engine.cpp:1868-1870`):

```cpp
return conditioning_cache_->get(*cond_stage_model, n_threads, params);
```

`cond_stage_model` is a `std::shared_ptr<Conditioner>` built in `build_core_runners()` (`diffusion_engine.cpp:1087`, registered as `ModelComponent::Conditioner` / `SDBackendModule::TE` at 1105).

### 3. What "encode" actually is (LLM forward)

The concrete conditioner depends on model version; the integration files build it (`src/model/integrations/*.cpp`, e.g. `qwen_image.cpp`, `flux*.cpp`, `z_image.cpp` all do `make_shared<LLMEmbedder>`; SD1/SD2/SDXL → `FrozenCLIPEmbedderWithCustomWords`).

LLM path (`LLMEmbedder`, `src/conditioning/conditioner.hpp:1974`):

```
LLMEmbedder::get_learned_condition            conditioner.hpp:2437
 ├─ build prompt template / prompt_attn_range / min/max length per version
 ├─ tokenize()                                 conditioner.hpp:2146-2247
 └─ encode_prompt()                            conditioner.hpp:2249-2343
      ├─ sd::Tensor<int32_t> input_ids [n]     conditioner.hpp:2272
      ├─ causal attention mask (if mask)       conditioner.hpp:2273-2289
      ├─ optional position_ids (flux3)         conditioner.hpp:2291-2299
      ├─ LLM::LLMRunner::compute(n_threads, input_ids, mask, image_embeds, out_layers, ...)
      │                                        conditioner.hpp:2300 → src/model/te/llm.hpp:2195
      │                                        (LLMRunner : GGMLRunner, graph execute)
      ├─ apply_token_weights(hidden, weights)  conditioner.hpp:2314
      └─ slice off prompt_template_encode_start_idx tokens, optional zero pad
                                               conditioner.hpp:2324-2341
 └─ SDCondition result; c_crossattn = hidden; extra_c_crossattns / c_vector / c_token_types
                                               conditioner.hpp:3288-3319
```

CLIP-family path is analogous: `FrozenCLIPEmbedderWithCustomWords::get_learned_condition` (`conditioner.hpp:575`) → `tokenize` (parse_prompt_attention, BREAK, embeddings) → per-77-token chunk `CLIPTextModelRunner::compute` → concat chunks, SDXL pooled + ADM vector (`conditioner.hpp:453-573`). T5/Flux/SD3 variants: `conditioner.hpp:647` (`SD3CLIPEmbedder`), `1092` (`FluxCLIPEmbedder`), `1388/1622` (`T5CLIPEmbedder`).

Important representation fact for the swap tickets: `SDCondition` (`conditioner.hpp:25-83`) holds only `sd::Tensor<float>` / `sd::Tensor<int32_t>`, and `sd::Tensor` stores a plain `std::vector<T>` (`src/core/tensor.hpp:73-120,340`). Conditioning is therefore **host RAM, backend-agnostic, copyable, and serializable in principle** — no ggml backend handle is embedded.

### 4. Is conditioning cached?

Yes, there is a per-context LRU, but sd-cli turns it off.

- `ConditioningCache` — `src/conditioning/conditioning_cache.h:10-106`. Capacity from `sd_ctx_params_t::conditioning_cache_size` (default 4, `include/stable-diffusion.h:256`; CLI struct default 4, `common.h:126`), applied in `StableDiffusionGGML::init` (`diffusion_engine.cpp:925`).
- Lookup key is `ConditionerParams` content: `text, clip_skip, width, height, zero_out_masked`, the `RefImageParams` tuple, plus deep-compare of `ref_images` / `minimax_h3_references` tensors (`conditioning_cache.h:45-71`). It is an LRU list; entries hold `SDCondition` copies.
- `capacity_ == 0` → straight passthrough (`conditioning_cache.h:86-88`). **sd-cli hardcodes `ctx_params.conditioning_cache_size = 0` at `examples/cli/main.cpp:644`**; `--conditioning-cache-size` can re-enable it (`common.cpp:592-595`), which is then copied at `common.cpp:1067`.
- Invalidation: `reset_runners()` clears it (`diffusion_engine.cpp:169`); `apply_loras()` clears it on source-set change or failure (`1833,1843,1854`). It is per `StableDiffusionGGML` instance, never persisted.
- Note the cache would not help a batch of *distinct* prompt texts anyway; it only dedupes repeats (e.g. negative prompt encoded as both `uncond` and `img_uncond`), and since `generate_image` encodes once per call, it only helps across repeated calls in one process.

### 5. Text-encoder residency today

- `eager_load` is false by default (`include/stable-diffusion.h:243`, `diffusion_engine.cpp:1370`): params are prepared lazily on first use. When the text encoder is assigned to CPU (default in a CPU-only build, or `--backend "te=cpu"`, `common.cpp:557-561`), its params backend and compute backend are the same, so after the first encode the TE weights remain resident in RAM for the life of the context.
- Immediately after the embeds are produced, `prepare_image_generation_embeds` installs `ConditionerRunnerEndOnExit` (`image.cpp:477`, `generation.h:45-52`) which calls `cond_stage_model->runner_end()` on return. `GGMLRunner::runner_end()` (`src/core/ggml_runner.cpp:488-516`) releases the workspace/compute ctx, clears graph caches, and calls `manager->evict_compute_backend_params(...)`.
- `ModelManager::evict_compute_backend_params` (`src/model_manager.cpp:1642-1693`) only releases RAM param blocks whose `residency_mode == ResidencyMode::Disk`; the compute-backend staging copies are released. So on the target CPU-TE setup this frees the compute scratch, not the TE weights; with disk-streaming residency it can actually drop RAM pages.
- Additionally, the public C API wraps every `generate_image`/`generate_video` call in `ExecutionScope` (`stable-diffusion.cpp:761-763`), whose destructor calls `StableDiffusionGGML::end_runners()` (`diffusion_engine.h:260-264`). `ContextOperation` takes a `try_to_lock` on `execution_mutex` (`diffusion_engine.h:227-249`), so encode and generate on one context can never overlap today.

### 6. Where a split encode/generate can hook in

Ordered from narrowest to widest:

1. **`prepare_image_generation_embeds` (`image.cpp:471`)** — the precise seam. Its only request inputs are `prompt`, `negative_prompt`, `clip_skip`, `width`, `height`; plus `latents` for two things: ref images handed to the VLM (`pass_to_vlm`, `image.cpp:484-486`) and the `c_concat` fallback for DiT-ref models (`502-504, 546-548, 573-575`). Split it into (a) "build `ConditionerParams` from request" and (b) "encode → `ImageGenerationEmbeds`", then let `generate_image` accept precomputed embeds. Everything textual is latents-independent; `c_concat` must be filled later or flagged so an encode cache never stores latents.
2. **`StableDiffusionGGML::get_learned_condition` (`diffusion_engine.cpp:1868`)** — the single funnel used by both cond and uncond (and by `video.cpp:1212/1237`). A public encode API would call this inside its own `ExecutionScope`, returning conditioning handles instead of text.
3. **Residency boundaries**: `ConditionerRunnerEndOnExit` (`generation.h:45`) and `ExecutionScope::~ExecutionScope` → `end_runners()` (`diffusion_engine.h:260`). An "encode N prompts, then generate M images" loop needs the encode phase to keep its scope (skip the early `runner_end`), and needs the generate phase to not re-load TE. Both are small, local changes but they change lifetime semantics, so they belong with ticket 02.
4. **`ConditioningCache` (`conditioning_cache.h`)** — already the in-RAM store of `SDCondition` keyed by encode inputs. Two viable strategies: raise capacity ≥ prompt count and let the cache be the batch store (cheap, but capacity/eviction become the API), or have encode return explicit handles that the cache does not own (predictable, needs a new `sd_condition_t` + free function).
5. **CLI (`main.cpp:880-940`)** — needs a prompt-vector source (line-delimited file or repeated `--prompt`), an encode loop, a generate loop, and output naming beyond `_<idx>`; also the seed+b rule (`request.cpp:231-233`) must be reconciled per prompt.
6. **ADetailer (`detailer.cpp:977-1010`)** — benefits from the same seam: today it re-encodes the ad prompt per region; with precomputed embeds it would encode once and reuse per region.

### 7. Constraints / gotchas to carry forward

- `sd_img_gen_params_t` (`include/stable-diffusion.h:417-447`) has no embeddings field; there is no public encode API anywhere in `include/stable-diffusion.h`. New surface required.
- Conditioning cache is keyed on `width`/`height`, so reusing embeddings across differently-sized generations is *not* what the current cache would do; a split API must decide whether size is part of the handle identity (it matters for SDXL ADM vector: `conditioner.hpp:542-562`).
- `ref_images` can be part of the encode input (VLM path) and are deep-compared by the cache; prompt batches that share reference images can share an encode, ones that don't cannot.
- `request.prompt` is copied from the C string (`request.cpp:124`), so a split API must own/reference prompt text for the lifetime of the handle or hash it.
- `--prompt-file` content is taken verbatim (including trailing newline) and only `<lora:...>` tags are stripped, in `SDGenerationParams::resolve` (`common.cpp:2623-2666`).
- Video generation has the same pattern (`src/pipeline/video.cpp:1200,1212,1237`) — a library-level split should be designed to serve both `generate_image` and `generate_video` or explicitly scope to image-only.

### File:line index

| Concern | Location |
| --- | --- |
| CLI one-shot params, cache=0 | `examples/cli/main.cpp:642-648, 880-940` |
| `--prompt-file` read | `examples/common/common.cpp:1712-1728` (registered `1839`) |
| `-p/--prompt`, `-b/--batch-count` | `examples/common/common.cpp:1104-1110, 1232-1235` |
| batch clamp/validate | `examples/common/common.cpp:2635, 2675-2678` |
| prompt → `sd_img_gen_params_t` | `examples/common/common.cpp:2861-2874` |
| C API entry + ExecutionScope | `src/stable-diffusion.cpp:754-778` |
| pipeline `generate_image` | `src/pipeline/image.cpp:834-1150` |
| embeds computed once | `src/pipeline/image.cpp:471-588, 893-902` |
| batch sampling loop (seed+b) | `src/pipeline/image.cpp:905-963`; `src/pipeline/request.cpp:231-233` |
| conditioning funnel | `src/pipeline/diffusion_engine.cpp:1868-1870` |
| `ConditioningCache` | `src/conditioning/conditioning_cache.h:10-106` |
| `ConditionerParams` / `SDCondition` / `Conditioner` | `src/conditioning/conditioner.hpp:25-83, 132-161` |
| LLM forward | `src/conditioning/conditioner.hpp:2437, 2249-2343, 2300, 3288-3319` |
| runner teardown / eviction | `src/core/ggml_runner.cpp:488-516`; `src/model_manager.cpp:1642-1693` |
| conditioner early `runner_end` | `src/pipeline/generation.h:45-52`; call `src/pipeline/image.cpp:477` |
| ADetailer re-encode per region | `src/detailer.cpp:977-1010`; `examples/cli/main.cpp:604-626` |
