# T8: Branch generate_image on provided condition

Type: task
Status: resolved
Blocked by: 01, 07

## Task

In `sd::pipeline::generate_image` (image.cpp:834) and `prepare_image_generation_embeds` (image.cpp:471), when `params.condition != NULL`, skip `get_learned_condition` and use the cached SDCondition (cond/uncond/img_uncond). ADetailer (detailer.cpp:977) should reuse cached cond instead of re-encoding per region.

## Acceptance

- Generate from condition matches baseline image (same seed) within tolerance.

## Answer

Landed in `f1c0bde0` (merged in `2f00a13f`, hardening in `b71effb5`).

- `prepare_image_generation_embeds(..., const ImageGenerationEmbeds* precomputed = nullptr)`: when non-null it copies the embeds, skips every `get_learned_condition` call and every model-specific uncond derivation (those were already materialised at encode time), and fills the latents-dependent `c_concat` for `cond`/`uncond`/`img_uncond` exactly like the baseline fallbacks. The empty-check is kept because the handle is always encoded with empty latents.
- The precomputed branch also recomputes IP-Adapter tokens (`compute_ip_adapter_tokens`), which in a batch otherwise belong to the last encoded prompt.
- `generate_image` passes `sd_img_gen_params->condition ? &...->embeds : nullptr`; with `condition == NULL` behaviour is unchanged.
- ADetailer reuses `inpaint_params->condition` only when the per-region resolved prompt equals the base prompt and, for size-dependent conditioners (SDXL/ADM vector), the region dimensions match the encode dimensions stored on the handle.

Verified (MOSS-Image-SFT Q6_K + Qwen3.5-9B TE, `--max-vram 4`, 512x512, 8 steps, seed 42):
- single-stage `-p "a red fox in a snowy forest, oil painting"` → `baseline_gc.png`
- batch mode (`--prompt-file` with the same single line) → `batch1.png`
- both SHA-256 `4e679610f210f7f357aff2d1995e0924e9134148984519588929be2a16bf0995` — byte-identical, i.e. exact parity, not just within tolerance.
- `out_0.png` from the 100-prompt batch matches the same hash.
