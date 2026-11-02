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
