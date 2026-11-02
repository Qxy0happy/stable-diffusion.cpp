# Can ModelManager evict TE params and lazily reload within a live context?

Type: research
Status: open

## Question

Does `ModelManager` support evicting text-encoder (TE) params and lazily reloading them later within the same `sd_ctx_t`, without destroying the context? Investigate `load_tensors_to_params_backend`, `prepare_params`, residency mode, and whether `SDContextParams::conditioning_cache_size` / eager_load flags allow partial teardown. Report which modules can be demoted to disk/CPU and re-staged.

## Answer

**Verdict: ModelManager can evict and lazily reload TE weights within a live
`sd_ctx_t`, but only when TE was registered in `ResidencyMode::Disk`
(`params_backend` assignment `te=disk` at context creation), or indirectly when
the VRAM-pressure path demotes GPU-resident `ParamBackend` states. There is no
runtime API to evict a specific module on demand, and CPU-resident
`ParamBackend` weights (e.g. an LLM kept in RAM) can never be evicted today.**

Everything needed for the *reload* half already exists and is exercised on every
graph; the missing piece is a policy/API to *evict* a chosen module.

### 1. Lazy (re)load is the default and is repeatable

- Every graph execution calls `ModelManager::prepare_params` for the segment's
  params (`src/core/ggml_runner.cpp:994` → `src/core/segment_weight_pipeline.cpp:104-119`),
  so loading is lazy on first use unless `eager_load` is set.
- `prepare_params` (`src/model_manager.cpp:1578-1613`) →
  `load_tensors_to_params_backend` (`427-500`) → `mmap_params` /
  `alloc_params_buffers` / `load_tensors` → `stage_tensors_to_compute_backend`
  (`502-629`), then pins the states and applies pending LoRAs.
- `load_tensors_to_params_backend` skips any state with
  `loaded_to_params_backend == true` (`440-443`), so calling it again after an
  eviction simply reloads. `free_params_storage_block` is the only place that
  clears the flag (and nulls `tensor->buffer/data/extra`) at
  `src/model_manager.cpp:1241-1251`.
- Reload data comes from `TensorState::sources` via the `ModelLoader`; the loader
  reopens the file on each load (`src/model_loader.cpp:1184-1191`), so no open
  handle is required. `sources`, `metadata_validated` and the runner's
  `params_ctx` tensor metadata survive eviction — only the buffer data is freed.
- Reload therefore works inside the same context as long as the owning runner
  objects (and `ModelManager`) are alive. Destroying the runner
  (`reset_runners` → `cond_stage_model.reset()`, `diffusion_engine.cpp:168-207`)
  invalidates the tensor pointers and is a full reload, not an eviction.

### 2. Residency mode is fixed at registration

- `ModelManager::ResidencyMode { Disk, ParamBackend }`, default `ParamBackend`
  (`src/model_manager.h:19-22`, `45`; set at `src/model_manager.cpp:262`).
- Chosen once, at runner registration:
  `register_runner_params` (`src/pipeline/diffusion_engine.cpp:316-372`) maps
  `backend_manager.params_backend_is_disk(module)` → `Disk`
  (`332-333`). `disk` is a params-backend-only token
  (`src/core/ggml_extend_backend.cpp:953-978`, `docs/backend.md:369-391`); it does
  not create a separate backend — params use the module's runtime backend.
- Consequences of `Disk`:
  - `params_backend == compute_backend`, so no compute staging is created
    (`stage_tensors_to_compute_backend` skips `compute == params`, `518`).
  - mmap is disabled (`can_mmap_storage` requires `ParamBackend`,
    `src/model_manager.cpp:858-868`), so a reload is a real read + alloc.
  - Its params blocks become releasable: `release_params_storage_blocks` only
    frees blocks whose every state is unpinned, unstaged **and** `Disk`
    (`src/model_manager.cpp:1255-1303`, condition `1277-1279`).
- Consequences of `ParamBackend`: weights stay resident; may be mmap-backed
  (`806-868`); **not** releasable via `evict_compute_backend_params`.

### 3. Existing eviction paths

**a) Per-runner end (the practical, request-scoped release).**
`GGMLRunner::runner_end` (`src/core/ggml_runner.cpp:488-516`) calls
`ModelManager::evict_compute_backend_params` for that runner's params.
`evict_compute_backend_params` (`src/model_manager.cpp:1642-1695`):

- frees compute staging blocks whose states are all unpinned (`1652-1673`);
- frees params storage blocks whose states are all unpinned, unstaged and
  `Disk` (`1675-1694`).

`runner_end` is reached automatically at the end of every C API request through
`ExecutionScope::~ExecutionScope` → `end_runners()` → each conditioner/runner's
`runner_end()` (`src/pipeline/diffusion_engine.h:251-265`,
`src/stable-diffusion.cpp:761,787`, `src/pipeline/diffusion_engine.cpp:142-166`,
`src/conditioning/conditioner.hpp:798-806, 1884-1885`). So a `te=disk` TE is
loaded for the encode and freed when the request ends; the next request reloads
it from disk. `end_runners()` is a public method of `StableDiffusionGGML`
(`diffusion_engine.h:217`), i.e. callable from new C API code.

The same call is the "demote to CPU" mechanism for `params_backend=cpu` modules:
the `ParamBackend` host weights stay in RAM, while their GPU compute staging
blocks are freed and re-staged by `prepare_params` on next use (this is what
`--offload-to-cpu` / `te=cpu` relies on). Re-targeting the compute backend is
done by `assign_compute_backend` (`1391-1438`), which only accepts unpinned,
unstaged states.

**b) Capacity pressure (GPU only).**
`ensure_compute_backend_capacity` (`src/model_manager.cpp:1863-2048`):

- returns immediately for CPU compute (`1878-1880`);
- releases staged GPU copies and `Disk`-mode params blocks
  (`add_evictable_state`, `1910-1924`);
- then *demotes* eligible `ParamBackend` states to `Disk` and releases them
  (`1985-2009`, `can_demote_to_disk` at `1986-1995`).

Demotion requires: no `view_src`, mode `ParamBackend`, `params_backend ==
compute_backend`, backend not CPU, `pin_count == 0`, not staged, already
loaded, not protected/required, not in an mmap block, `sources` non-empty, no
LoRA applied (`applied_lora_epoch == UINT64_MAX`), no split buffer.
Demotion is effectively permanent for states whose block was freed:
`revert_demotions` only restores `ParamBackend` while
`loaded_to_params_backend` is still true (`1978-1984`), which
`free_params_storage_block` has already cleared.

**c) Unregister + re-register (in-context, no model reload).**
`unregister_param_tensors(component)` (`src/model_manager.cpp:288-295`) →
`unregister_tensor_states` (`307-398`) rejects pinned/staged states, then frees
every params/staging block containing the component's states and drops the
states. Re-registering the same `ggml_tensor*` pointers
(`register_param_tensors`, `218-286`) and letting the next graph call
`prepare_params` reloads from disk. Caveats: block-granular (may also free a
sibling component that shared the block), and no production caller does this for
TE today — only `LoraModel` unregisters its own params. `GGMLRunner::evict_compute_backend_param_tensors`
(`src/core/ggml_runner.cpp:120-128`) exists as an unused hook into the same
eviction call.

### 4. What cannot be evicted

- **CPU-resident `ParamBackend` weights** (`params_backend == compute_backend ==
  CPU`, the LLM/TE-on-CPU case): `can_demote_to_disk` requires
  `!sd_backend_is_cpu`, the capacity path returns early for CPU, and
  `evict_compute_backend_params` needs `Disk`. Such weights stay in RAM until
  the context is destroyed or the component is unregistered.
- mmap-backed blocks (excluded from demotion `1970-1975`; mmap only exists in
  `ParamBackend`).
- Row-split params (`split_buffer_type != nullptr`) and LoRA-merged params.
- Parts of a block: releases are all-or-nothing per allocation block, and blocks
  are grouped by `(buffer_type, residency_mode)`
  (`alloc_params_buffers`, `870-967`). TE in `Disk` mode is grouped separately
  from `ParamBackend` siblings, but other `disk` modules can share its blocks and
  then block its release.
- `common_ignore_tensors` is unrelated to eviction: it is a static set of
  prefixes skipped at registration/load (`should_ignore`, `745-752`, populated in
  `diffusion_engine.cpp:1336-1364`), decided at init.

### 5. Which modules can be demoted / re-staged

Registration-time `Disk` is available to every module registered through
`register_runner_params` (mode from `params_backend_is_disk`, `diffusion_engine.cpp:332-333`)
plus the upscaler/detector/LoRA paths:

| Module (ModelComponent) | Registration | `disk` at registration | Runtime demotion |
|---|---|---|---|
| Conditioner / TE (`SDBackendModule::TE`) | `diffusion_engine.cpp:1105` | yes | yes, only if GPU-resident |
| Diffusion, HighNoiseDiffusion | `1106-1107` | yes | yes, if GPU-resident |
| CLIPVision, IPAdapter, AudioEncoder | `1108-1110` | yes | yes, if GPU-resident |
| VAE, PreviewVAE, AudioVAE | `1131-1133` | yes | yes, if GPU-resident |
| ControlNet | `1152-1165` | yes | yes, if GPU-resident |
| Extensions (PhotoMaker/PuLID, `PHOTOMAKER`) | `1184` | yes | yes, if GPU-resident |
| LoRA | `1633-1639` | yes | params follow compute; own unregister on teardown |
| Upscaler | `src/upscaler.cpp:139-143` | yes | yes, if GPU-resident |
| Detector | `src/detailer.cpp:713-718` | yes | yes, if GPU-resident |
| LatentUpsampler | `src/pipeline/video.cpp:1406-1410` | **no** (hardcoded `ParamBackend`) | yes, if GPU-resident |
| convert (in-memory) | `src/convert.cpp:498-500` | no (hardcoded, CPU) | no |

"Re-staged" after eviction: `prepare_params` re-runs `load_tensors_to_params_backend`
plus `stage_tensors_to_compute_backend` on next use; GPU staging can also be
prefetched ahead of the segment, but prefetch requires the params to be loaded
already (`populate_prefetch_block`, `src/model_manager_prefetch.cpp:71-78`), so
prefetch is staging, not disk reload.

### 6. `conditioning_cache_size` and `eager_load` do not enable partial teardown

- `conditioning_cache_size` only sizes `ConditioningCache`
  (`src/conditioning/conditioning_cache.h:34-105`, wired at
  `diffusion_engine.cpp:906-925`, default 4, 0 disables). Entries are
  `SDCondition` host tensors (`sd::Tensor<float>` owning host memory, see
  `make_sd_tensor_from_ggml`, `src/core/tensor_ggml.hpp:50-83`), so cached
  conditioning remains valid after TE weights are evicted. Clearing happens on
  `reset_runners` and `apply_loras` (`diffusion_engine.cpp:169, 1833, 1843, 1854`).
- `eager_load` only preloads all params at init
  (`diffusion_engine.cpp:1370-1378` → `load_all_params_eagerly`,
  `model_manager.cpp:400-409`); it is force-disabled under graph-cut layer
  split (`diffusion_engine.cpp:998-1000`). It does not change evictability, and
  eager loading all states can leave extra blocks that a per-runner eviction
  will not match.

### 7. Concrete implications for the LLM encode swap

- To free a 4.9 GB CPU LLM between encode and generate **inside one live
  context**, the context must be created with a `params_backend` assignment such
  as `*=cpu,te=disk` (or `te=disk`). Then run the encode batch and call
  `end_runners()` once at the end of the batch (or make the new encode API its
  own `ExecutionScope`): `runner_end` will release the `Disk`-mode TE buffers.
  The next encode reloads from the model file.
- Do **not** rely on automatic release during the batch: TE chunk loops pass
  `auto_runner_end = false` (`conditioner.hpp:1033-1036`, `1951-1957`) precisely
  to keep weights resident across chunks; CPU has no VRAM-pressure path, so
  release must be explicit. Conversely `compute_input_embeds` ends the runner
  after each call (`src/model/te/llm.hpp:2658-2669`) — using it per prompt would
  evict/reload the LLM per prompt.
- `Disk` mode disables mmap, so each batch pays a real read + allocation of the
  TE file (a several-GB read for the 9B LLM). That is acceptable once per batch
  but rules out per-prompt teardown.
- If requiring `te=disk` at init is undesirable, the smallest new capability is a
  `ModelManager::demote_component_to_disk(ModelComponent)` that (i) sets
  `residency_mode = Disk` for that component's states plus siblings in shared
  blocks, and (ii) calls `release_params_storage_blocks(false, &target)`. It must
  drop `can_demote_to_disk`'s `!sd_backend_is_cpu` restriction for this
  explicit path (host buffers are freed by `free_params_storage_block` just the
  same), and must skip mmap-backed / split / LoRA-applied states. Reload then
  needs no new code.
- Encode and generate cannot overlap in one context: `ContextOperation`
  `try_lock`s `execution_mutex` and fails if another operation is executing
  (`diffusion_engine.h:227-249`).
- Keep runner objects alive across the swap; only buffer data is reclaimed, so
  peak RAM drops by the TE weight bytes without a re-init of the TE graph code.

### 8. Key code paths (for the implementing ticket)

| Concern | Location |
|---|---|
| Residency mode enum / default | `src/model_manager.h:19-22, 45` |
| Mode chosen from `params_backend` | `src/pipeline/diffusion_engine.cpp:332-333`; `src/core/ggml_extend_backend.cpp:953-978` |
| Lazy load / reload | `src/model_manager.cpp:427-500, 806-868, 969-1118, 1578-1613` |
| Compute staging | `src/model_manager.cpp:502-629`; skip when `compute == params` (`518`) |
| Params block release | `src/model_manager.cpp:1234-1303` (needs `Disk`, `1277-1279`) |
| Explicit eviction | `src/model_manager.cpp:1642-1695` |
| Pressure eviction + demotion | `src/model_manager.cpp:1863-2048` (demote `1985-2009`) |
| Unregister / re-register | `src/model_manager.cpp:218-398` |
| Per-runner end | `src/core/ggml_runner.cpp:488-516`; `src/conditioning/conditioner.hpp:798-806, 1884-1885` |
| Request scope → `end_runners` | `src/pipeline/diffusion_engine.h:251-265`; `src/stable-diffusion.cpp:754-812` |
| Conditioning cache | `src/conditioning/conditioning_cache.h:34-105`; host-copy at `src/core/tensor_ggml.hpp:50-83` |
| Eager load | `src/pipeline/diffusion_engine.cpp:1370-1378`; `src/model_manager.cpp:400-409` |
| Docs for `disk` | `docs/backend.md:33-37, 369-391, 417, 438` |

