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

**显存口径更正 + 空卡实测（1024²，MOSS-Image-SFT Q6K + Qwen3.5-9B TE）。** 上面的数字全部是 RSS。显存用 GPU Adapter Memory 计数器的 dedicated / shared 分开采样；桌面 GPU 占用清空后（dedicated 基线 ~390 MiB，可用 ~7.9 GB）：

| 配置 | TE 参数位置 | DiT 参数位置 | encode 峰值 dedicated | generate 峰值 dedicated | peak shared | 单图耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 单阶段，默认 params | CUDA_Host | CUDA_Host | — | +5444 MiB | +9246 MiB | 44 s |
| 两阶段，默认 params | CUDA_Host（unload 释放 4883 MB RAM） | CUDA_Host | +4018 MiB | +5482 MiB | +4961 → +4346 MiB | 43 s |
| 两阶段，`--params-backend *=cuda0` | VRAM 4883 MB（unload 释放 4883 MB VRAM） | VRAM 4258 MB | +4973 MiB | +6941 MiB | +99 MiB | 42 s |
| 单阶段，`--params-backend *=cuda0` | VRAM 4883 MB（常驻） | 未进 VRAM，退化为分段 staging | — | +7053 MiB | +294 MiB | 49 s |

结论：

- 显存释放是真实且可复现的，但**前提是 TE 参数在 device buffer**（`--params-backend te=cuda0`）：日志实测 `released 4883.55 MB VRAM, 0.00 MB RAM`。默认 params 走 `CUDA_Host` 时同一行是 `0.00 MB VRAM, 4883.55 MB RAM`，`nvidia-smi memory.used` 的下降来自 pinned host（WDDM 把 shared 计入），不是显存。
- 顺序驻留的收益不是「峰值更低」，而是**DiT 参数得以整体驻留 VRAM**：TE 4883 + DiT 4258 = 9141 MB 超过 8188 MB 卡容量，单阶段时管理器无法让 DiT 进 VRAM（日志里没有 4258 MB 的 VRAM 装载），只能逐段 staging；两阶段卸载后 DiT 完整常驻。对应 1024² 单图 49 s → 42 s，encode 3.4 s → 1.9 s。
- 四次运行的输出 SHA-256 全部相同（`2f6a45ec412c2d3b1371db5aa5faafae9df42cca9610efd324082bfa4805c9af`），说明 backend/params 放置不改变结果。
- 1024² 现在可跑的前提是桌面不再占用显存；此前 768² 就失败是因为桌面占了 3.4 GB dedicated。
- 因此 T10 的「< 6 GB」只能作为 RAM 验收；显存侧应以 `--params-backend te=cuda0` 配置下的「load 时 4883 MB VRAM、generate 峰值 6.9 GB」为准。
