# T8: Branch generate_image on provided condition

Type: task
Status: open
Blocked by: 01, 07

## Task

In `sd::pipeline::generate_image` (image.cpp:834) and `prepare_image_generation_embeds` (image.cpp:471), when `params.condition != NULL`, skip `get_learned_condition` and use the cached SDCondition (cond/uncond/img_uncond). ADetailer (detailer.cpp:977) should reuse cached cond instead of re-encoding per region.

## Acceptance

- Generate from condition matches baseline image (same seed) within tolerance.
