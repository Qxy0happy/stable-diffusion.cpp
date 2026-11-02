# T10: Verify batch memory + quality

Type: task
Status: resolved
Blocked by: 09

## Task

Run 100-prompt batch on the Q6_K build; profile peak RSS before/after TE unload; compare output quality to baseline single-stage (SSIM/PSNR or visual). Report commands and numbers.

## Acceptance

- Peak RSS < 6GB during generate phase; quality parity documented.

## Answer

Verified on the integration branch `spec/llm-encode-swap` @ `d10669a0` with a CUDA (clang-cl, Ninja, `SD_CUDA=ON`) `sd-cli`.

Setup: `D:/Cache/Download/MOSS-Image-SFT` — `MOSS-Image-SFT-Q6K.gguf` (4258 MB DiT), `Qwen3.5-9B-UD-Q4_K_XL.gguf` (TE, `--backend te=cpu`), `vae/diffusion_pytorch_model.safetensors`, `tokenizer/tokenizer.json`. Flags: `--cfg-scale 3.0 --seed 42 --steps 8 -H 512 -W 512 --max-vram 4` (`--max-vram 4` is required because the desktop already holds ~3.6 GB of the 8 GB VRAM; without it the workspace capacity check fails at 512 and 768). RSS sampled every 200 ms with a PowerShell wrapper that tags phases from the CLI logs.

| run | prompts | wall | peak RSS | notes |
| --- | --- | --- | --- | --- |
| single-stage baseline (`-p`) | 1 | 25 s | 9.82 GB | LLM + DiT resident together |
| two-phase batch (`--prompt-file`) | 1 | 27 s | 5.2 GB | byte-identical image |
| two-phase batch (`--prompt-file`) | 100 | 1004 s | 5.21 GB encode / 5.16 GB generate | 100/100 images, 0 errors |

Phase profile of the 100-prompt run: encode 100 prompts in 101.0 s (LLM on CPU), `text encoder unloaded` → RSS falls to 1.03 GB, generate phase peaks at 5.16 GB (DiT + VAE + conditioning cache), i.e. the spec's memory contract holds: generate peak is 5.16 GB < 6 GB and the LLM is no longer resident.

Quality parity: the 1-prompt batch image and `out_0.png` of the 100-prompt batch share SHA-256 `4e679610f210f7f357aff2d1995e0924e9134148984519588929be2a16bf0995` with the single-stage baseline — exact, byte-for-byte equality at seed 42.

Commands and artefacts: `D:/Cache/Agents/sdcpp-smoke/t10/{run.ps1,baseline_gc.args,batch1.args,batch100.args,*.rss.csv,prompts100.txt,out_*.png}`.

## Comments

**显存口径更正（`--backend te=cuda0` 实测）。** 上面的数字全部是 RSS。显存侧用 GPU Adapter Memory 计数器（dedicated / shared 分开）重测 1-prompt 批处理，桌面基线 dedicated 3461 MiB / shared 201 MiB：

| 阶段 | 显存峰值（dedicated 增量） | shared 增量 |
| --- | --- | --- |
| encode（TE 在 `cuda0`） | +3685 MiB | +5002 MiB |
| generate（DiT） | +4290 MiB | +4649 MiB |

结论：`nvidia-smi memory.used` 在 WDDM 上等于 dedicated + shared，之前把 +4202 MiB 全算作显存是错的。TE 的 4883 MB 参数落在 `CUDA_Host`（pinned host → shared），unload 释放的是这部分 RAM；dedicated VRAM 里的分段 staging 由 encode 结束时的 `runner_end` 释放，不经过 unload。`sd_unload_text_encoder` 现在会打印 `released X MB VRAM, Y MB RAM`，在该配置下实测为 `0.00 MB VRAM, 4883.55 MB RAM`。

dedicated 峰值之所以停在 4.3 GB 以内，是 `--max-vram 4`（自动图切分）在起作用；它在本改动之前就存在。因此本 ticket 的「< 6 GB」只能作为 RAM 验收，显存结论必须按上表重述。
