# Repository Guidelines

This document is for AI coding agents working in `stable-diffusion.cpp`. It covers
repository routing, build/test workflow, architecture, conventions, and pitfalls.
General contribution rules (PR scope, commit style, security, AI policy) live in
`CONTRIBUTING.md`.

## Agent Operating Rules

Before analyzing or modifying the repository:

1. Read this file.
2. Confirm the current tree with `rg`/directory listing before relying on a path.
3. Start from `src/` and relevant `docs/` for runtime behavior.
4. Read relevant code before editing.
5. Prefer the smallest change fitting existing architecture.
6. Run the narrowest useful build/inspection command available.

Agents must not:

- Run `git push`, create PRs, or submit issue/PR comments on the user's behalf.
- Create commits unless explicitly requested.
- Modify `ggml/`, `thirdparty/`, or `examples/server/frontend/` unless explicitly
  requested and necessary.
- Read large model files or tokenizer vocabulary files (`src/tokenizers/vocab/`).
- Rewrite unrelated code for style-only reasons.
- Add secrets, model weights, generated binaries, local absolute paths, or
  machine-specific output.

When a change is large, architectural, or likely to affect public behavior, pause
and present a short plan before editing.

## Project Overview

`stable-diffusion.cpp` is a C/C++17 diffusion inference library built on a
patched `ggml` submodule. It exposes three surfaces:

1. C library — `include/stable-diffusion.h` (`sd_ctx_t` opaque contexts).
2. CLI — `examples/cli/main.cpp` → `sd-cli`.
3. HTTP server + embedded web UI — `examples/server/main.cpp` → `sd-server`.

Supports SD 1.x/2.x, SDXL, SD3, Flux, Flux2, Z-Image, Qwen, Wan, LTX, Hunyuan,
and many more model families (see `src/model.h` `SDVersion`, append-only).

## Architecture & Data Flow

Layering (top → bottom):

```
include/stable-diffusion.h / src/stable-diffusion.cpp   C ABI
        ↓
src/pipeline/diffusion_engine.{h,cpp}   StableDiffusionGGML context/engine
        ↓
src/pipeline/{image,video,request,model_builders}.cpp   pipelines + sample plan
        ↓
src/conditioning + src/model/*   conditioners, diffusion runners, VAE, adapters
src/model/adapter, src/extensions   LoRA / IP-Adapter / PuLID / PhotoMaker
        ↓
src/core   sd::Tensor, ggml integration, backend manager, graph runner, RNG
        ↓
ggml/ (submodule)   CPU/CUDA/Vulkan/Metal/… kernels
        ↓ (loading)
src/model_loader* → src/model_manager*   file parse, version detect, residency
```

Key ownership: `StableDiffusionGGML` owns backend manager, all runners, RNGs,
`ModelManager`, LoRA/extensions, and the denoiser. `sd_ctx_t` wraps it.

### Context creation flow

`new_sd_ctx()` (`src/stable-diffusion.cpp`) → `StableDiffusionGGML::init()`
(`src/pipeline/diffusion_engine.cpp`):

1. Validate tensor types, scales, cache size, flash-attn, direct-conv.
2. Copy caller strings into `ModelConfig`.
3. Init `SDBackendManager` from `--backend`/`--params-backend`/`--split-mode`.
4. `ModelManager` + `ModelLoader` parse files, convert tensor names, detect
   `SDVersion` via `ModelLoader::get_sd_version()`.
5. `build_core_runners()` → `ModelDefinition::build()` creates Conditioner,
   diffusion runner (optional high-noise), CLIP vision, IP-Adapter, audio
   encoder.
6. `build_vae_runners()` selects VAE / TAE / audio VAE.
7. ControlNet, LoRA, extensions, prefetch/residency prepared.

### Image generation flow

`generate_image()` (`src/stable-diffusion.cpp`) → `sd::pipeline::generate_image()`
(`src/pipeline/image.cpp`):

1. `GenerationRequest` resolves size, seed, strength, guidance, model list,
   reference inputs.
2. `ImagePreprocessor` applies preprocessing rules.
3. Apply LoRAs + circular-axis settings.
4. `SamplePlan` resolves sampler, scheduler, sigmas.
5. Prepare initial latents (noise for txt2img; VAE encode for img2img) and
   conditioning embeddings.
6. Per-batch loop calls `StableDiffusionGGML::sample()` (denoiser, guidance,
   caches, preview/progress callbacks).
7. Decode latents via VAE; optional hires/upscale/ADetailer; build
   `sdcpp.generation/v1` JSON. Results returned as owned
   `sd_image_generation_result_t`.

Video generation follows the same shape via `src/pipeline/video.cpp` and
`sd_vid_gen_params_t`.

### Model integration flow

Canonical steps (`docs/model_config.md`):

1. Add `SDVersion` without renumbering existing values (`src/model.h`).
2. Implement network + `*Config::detect_from_weights()`.
3. Add a `ModelDefinition` in `src/model/integrations/`.
4. Register the family explicitly in `src/model_registry.cpp`.

## Key Directories

- `src/core/` — `sd::Tensor`, ggml integration, backend manager, graph runner/
  cutting, RNG, parallel executor, PE cache, probes.
- `src/model/` — model families and components (diffusion, te, vae, adapter).
- `src/model/integrations/` — one `ModelDefinition` per `SDVersion`.
- `src/model_io/` — GGUF, safetensors, torch, pickle, streaming writer.
- `src/runtime/` — schedulers/sigmas, denoisers, guidance, image preprocessing,
  sample caches (ucache, easycache, dbcache, taylorseer, cache-dit, spectrum).
- `src/tokenizers/` — tokenizer impls; `vocab/` is large data, do not read.
- `src/conditioning/` — conditioning/prompt implementation.
- `src/extensions/` — PhotoMaker, PuLID.
- `src/pipeline/` — engine, request/plan, image & video pipelines, builders.
- top-level `src/*.cpp/.h/.hpp` — public impl, model loading, conversion,
  versioning, managers, detailer, upscaler.
- `include/` — C API (`stable-diffusion.h`); treat as stable.
- `examples/cli/` — `sd-cli`; `examples/common/` — shared example code (not a
  library; compiled into both cli and server); `examples/server/` — `sd-server`
  + embedded frontend.
- `docs/` — model/build/behavior docs.
- `cmake/` — CMake support modules.
- `thirdparty/`, `ggml/` — vendored deps/submodules; do not modify.
- `models/`, `build*/` — local/generated; ignore.

## Development Commands

Submodule init first:

```bash
git submodule update --init --recursive
```

Build (canonical, `docs/build.md`):

```bash
cmake -S . -B build            # CPU; add -DSD_CUDA=ON etc.
cmake --build build --config Release
```

Outputs go to `${CMAKE_BINARY_DIR}/bin` (multi-config generators may use
`bin/Release/`). Key options: `SD_CUDA`, `SD_HIPBLAS`, `SD_METAL`, `SD_VULKAN`,
`SD_OPENCL`, `SD_SYCL`, `SD_MUSA`, `SD_WEBP`/`SD_WEBM`, `SD_BUILD_SHARED_LIBS`,
`SD_USE_SYSTEM_GGML`, `SD_GGML_SOURCE_DIR`. See `CMakeLists.txt` for full list.

Source glob: `file(GLOB … CONFIGURE_DEPENDS)` covers `src/*`, `src/conditioning`,
`src/core{,/cpu,/probe}`, `src/extensions`, `src/model/*/*`, `src/pipeline`,
`src/runtime`, `src/model_io`, `src/tokenizers{,/vocab}`. New `.cpp/.h/.hpp`
under an existing globbed dir needs no CMake edit; a new top-level `src/<dir>/`
needs the glob extended. `examples/common/` has no CMakeLists — its sources are
listed in both `examples/cli/CMakeLists.txt` and `examples/server/CMakeLists.txt`.

Frontend (when `SD_SERVER_BUILD_FRONTEND=ON`):

```bash
pnpm --ignore-workspace -C examples/server/frontend install
pnpm --ignore-workspace -C examples/server/frontend run build:header
```

Local-only: `Taskfile.yml` is untracked (`.git/info/exclude`); do not document
as repo-wide. CI is `.github/workflows/build.yml` (build-only matrix).

Style tooling (not CI-gated):

```bash
scripts/format-code.sh   # or .ps1
```

`.clang-format`: Chromium base, 4-space, no tabs, `ColumnLimit: 0`, aligned
consecutive assignments. `.clang-tidy`: a few `modernize-*` checks.

## Code Conventions & Common Patterns

- C11 + C++17 required. `CMAKE_EXPORT_COMPILE_COMMANDS=ON` →
  `compile_commands.json` for clangd.
- MSVC global flags: `_CRT_SECURE_NO_WARNINGS`,
  `_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING`, `/utf-8`, `/MP` (gated).
  Library additionally `/bigobj`, `/EHsc-` (ggml backends throw through C API).
- Public API in `include/` is stable; avoid breaking compatibility unless asked.
- Internal tensor-returning interfaces use an empty `sd::Tensor` for null/absent/
  failure. Do not add `std::optional<sd::Tensor<...>>` unless a distinct semantic
  state is truly required.
- `sd::Tensor` shape order is NOT PyTorch/NumPy-style: `shape()[0]` is the
  lowest/most-contiguous dimension; higher indices are higher dimensions.
  Broadcasting aligns dimensions from low to high; append implicit `1`s at the
  high end for fewer-dim tensors.
- `ggml_n_dims` can drop trailing singleton high dimensions; do not assume a
  logical trailing `1` is counted.
- Comments rare and useful: only for non-obvious invariants, constraints,
  compatibility, pitfalls. No task-specific or "what the code does" comments.
- UTF-8 with LF for text files. Do not assume terminal encoding on Windows.
- Naming/header rules per `CONTRIBUTING.md` (47–67).

## Important Files

- `include/stable-diffusion.h` — public C API.
- `src/stable-diffusion.cpp` — C API impl, params init, enum↔string tables.
- `src/model.h` — `SDVersion`, `ModelTraits`, version predicates.
- `src/model_definition.h` / `src/model_registry.cpp` — model families.
- `src/model_loader.{h,cpp}` — format parsing, version detection, mmap.
- `src/model_manager.{h,cpp}` — residency, lazy load, LoRA application.
- `src/name_conversion.{h,cpp}` — weight-name mapping.
- `src/pipeline/diffusion_engine.{h,cpp}` — context, sampling loop.
- `src/pipeline/{image,video,request,model_builders}.cpp` — pipelines.
- `src/core/` — tensor/ggml/backend foundation.
- `src/version.cpp` — version/commit from CMake compile defs.
- `CMakeLists.txt` — build config, options, globs, version stamping.
- `CONTRIBUTING.md` — contribution rules.
- `docs/model_config.md` — adding a new model family.

## Runtime/Tooling Preferences

- CMake ≥ 3.12, C11, C++17. Prefer `cmake -S . -B build` + `cmake --build`.
- Git submodules required (`ggml`, frontend, `thirdparty/{libwebp,libwebm,cutlass}`).
- Frontend uses pnpm/Vite+ (Vue 3 + TS); `vp check`/`vp test` (no tests today).
- Format with `.clang-format` via `scripts/format-code.sh|.ps1`.
- No node-based test runner in CI; `vp test` discovers zero tests currently.

## Testing & QA

There is **no automated C++ test suite**: `SD_BUILD_TESTS` is commented out in
`CMakeLists.txt`, no `enable_testing()`/`add_test()`, no CTest, no coverage
tooling. CI (`.github/workflows/build.yml`) validates **compilation only**
across its matrix. `CONTRIBUTING.md` forbids committing test code/scripts; keep
them local and report verification evidence in PRs.

Manual verification (document commands + key outputs in PRs):

```bash
# build
cmake -S . -B build && cmake --build build --config Release

# smoke generation (deterministic)
./build/bin/sd-cli -m models/...safetensors -p "a lovely cat" --seed 42 --steps 8

# inspect output metadata
./build/bin/sd-cli -M metadata --image ./output.png --metadata-format json

# version
./build/bin/sd-cli --version

# server
./build/bin/sd-server --model-dir ./models   # then hit relevant endpoint
```

Determinism checks are the documented ad-hoc regression technique: fixed
seed/settings, compare SHA-256 or byte-identical output. Backend probes use
deterministic F32 inputs for A/B comparison (`docs/backend_probe.md`), but that
is runtime auto-tuning, not a test suite.

Do not claim CI runs tests. Do not add committed tests. If a permanent
regression test is truly needed, confirm with the user first given
`CONTRIBUTING.md` policy.