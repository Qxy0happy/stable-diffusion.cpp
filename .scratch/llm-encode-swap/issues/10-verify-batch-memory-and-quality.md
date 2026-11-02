# T10: Verify batch memory + quality

Type: task
Status: open
Blocked by: 09

## Task

Run 100-prompt batch on the Q6_K build; profile peak RSS before/after TE unload; compare output quality to baseline single-stage (SSIM/PSNR or visual). Report commands and numbers.

## Acceptance

- Peak RSS < 6GB during generate phase; quality parity documented.
