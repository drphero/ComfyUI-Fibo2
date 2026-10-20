# Implementation and validation

Fibo 2.0 Edit Turbo was converted to separate, standalone ComfyUI checkpoints and tested on 20 October 2026. The model reference is [Bria's Fibo 2 implementation](https://github.com/huggingface/diffusers/pull/14991), pinned at `ab96e2b58c89adb9df418af74d81cc8b1e1be074`.

The installed layout is:

| Folder | Checkpoint | Size in bytes | Loader |
| --- | --- | ---: | --- |
| `models/diffusion_models` | `fibo2_edit_turbo_bf16.safetensors` | 16,254,770,698 | Fibo2 Diffusion Model Loader |
| `models/text_encoders` | `qwen3_vl_4b_fibo2_bf16.safetensors` | 8,887,924,135 | Fibo2 Text Encoder Loader |
| `models/vae` | `fibo2_vae.safetensors` | 336,211,452 | Stock Load VAE |

The transformer shards are merged, with architecture configuration embedded in checkpoint metadata. The text encoder's `model.language_model.*` and `model.visual.*` keys are converted to the native `model.*` and `visual.*` layout. Tokenizer JSON, fonts, and font licenses are embedded as byte tensors. The VAE keys and attention weight shapes are converted offline using ComfyUI's own conversion helper. No original release-directory files are read during inference. No core files are edited, no core loader functions are monkey-patched, and no Diffusers installation is needed. The two custom loaders use standard `folder_paths` resolvers and output ordinary `MODEL` and `CLIP` objects.

The runtime implementation uses `BaseModel`, `CoreModelPatcher`, ComfyUI operations, Comfy Kitchen fused RMSNorm/RoPE, per-module `ComfyAttention`, `AttentionTensorContainer`, and block prefetch queues. ComfyUI owns device placement and offloading. The FLUX flow prediction interface integrates the velocity with standard Euler sampling.

ComfyUI's normalized FLUX.2 VAE latents have 128 channels at 1/16 resolution. They already contain Fibo's 32 channels packed into 2×2 patches in `(channel, dy, dx)` order, so flattening their spatial dimensions gives the required image tokens. Predictions return in the same native latent layout. Gist uses planes 1–512, the target uses plane 513, and references use planes 514 onward.

The text encoder is ComfyUI's native Qwen3-VL-4B, including vision and DeepStack. Its six conditioning bundles concatenate normalized hidden states from `(9,20,31)`, `(5,16,27)`, `(7,18,29)`, `(9,20,31)`, `(11,22,33)`, and `(13,24,35)`. They travel as a standard 3D conditioning tensor and are unpacked inside the transformer. Tokenization uses embedded assets when present, otherwise the local assets bundled with the custom-node package.

The scheduler maps `1, 0.75, 0.5, 0.25` through Fibo's exponential time shift and appends zero. At 1024×1024 the sigmas are approximately `1, 0.90453077, 0.75951092, 0.51284410, 0`.

The text loader also accepts complete Hugging Face-style Qwen3-VL-4B single-file checkpoints directly. It normalizes parameter names, preserves the quantization markers/scales, and passes detected quantization metadata to the existing SDClipModel mixed-precision operations. INT8 ConvRot weights and embeddings are not reinterpreted as ordinary dense weights.

Checks completed:

| Check | Result |
| --- | --- |
| Transformer export integrity | All 783 tensors matched original values, shapes, and dtypes bit for bit |
| Text-encoder export integrity | Every neural-network tensor and all five embedded assets matched the originals bit for bit |
| Embedded tokenizer | Token IDs matched the original tokenizer, including multimodal markers and Unicode |
| VAE export integrity | Every tensor matched the native ComfyUI layout and its values bit for bit |
| Small-model transformer math against pinned reference | Passed with zero, one, and two references, including differently padded prompts; maximum float32 error below 0.000001 |
| Schedule, prompt formatting, mask preprocessing, source-path containment | Passed |
| Updated GUI workflow → API conversion → native queue | Passed with separate model/text loaders and stock VAE loader |
| Converted checkpoints, four Euler steps, 1024×704 | Successful red-jacket edit in 11.82 seconds after restart; output pixel-identical to the earlier release-directory run |
| Converted checkpoints, two references / masked VGL / text rendering at 512×352 | All three runs succeeded and were pixel-identical to the previous implementation at the same seeds and prompts |
| Downloaded qwen3vl_4b_bf16.safetensors plus standard flux2-vae.safetensors | Four-step 1024×704 edit succeeded, pixel-identical to the earlier BF16 output |
| Downloaded qwen3vl_4b_int8_convrot.safetensors plus standard FLUX.2 VAE | Four-step 1024×704 edit succeeded; visually matching red jacket, mean absolute pixel difference 0.145 on a 0–255 scale relative to BF16 |
| Native INT8 storage | ComfyUI selected MixedPrecisionOps and reported 4632 MB staged encoder weights, versus 8463 MB for BF16; no missing-weight warnings |
| Encoder-loading regression test | Both downloads and the older converted file passed key mapping, dtype, quantization marker/scale retention, tokenizer, and Unicode font-rendering checks |
| Core modifications / Diffusers dependency | Git working tree clean; Diffusers not installed or imported |

The preceding implementation was also checked at 1024×1024. The converted files preserve every tensor; the checkpoint-layout change requires no change to denoising math.

Timings are observations from this instance, not benchmark guarantees. Cache state and reference count affect runtime and memory use. Float32 small-model parity does not establish pixel-identical output against the original Diffusers runtime. CPU-only execution, other accelerators, quantization formats beyond the tested INT8 ConvRot checkpoint, LoRA, and older ComfyUI releases have not been validated end to end. Staged weight sizes are encoder storage observations, not peak-VRAM measurements. The pixel-difference measurement is one edit, not a broad quality benchmark.

Historical image-edit checks used a source crop from Bria's downloaded example. No source photograph or generated sample is included in the public release folders. The released workflows require users to select their own image. Release 0.3.1 validation is recorded below.

## Release 0.3.1 — ComfyUI-Fibo2

Release preparation completed 2026-10-20. The project folder, package metadata, displayed node labels, example titles and installed saved workflow use ComfyUI-Fibo2 / Fibo2. Node IDs already use `Fibo2*`; checkpoint format version 1 remains backward compatible. Original installed weight files were not replaced.

| Release check | Result |
| --- | --- |
| Rebuilt diffusion checkpoint | 783 tensors; each name, shape, dtype and raw tensor byte sequence matches the original shards |
| Source integrity / provenance | Both source-shard SHA-256 hashes match their downloaded Hugging Face LFS hashes; weights revision `e976703bda26af652cb93c8856cd4fa21005567d` |
| License documentation | Upstream original/latest cards retained separately; licensing reference `177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7`; Bria model license kept separate from GPL code and third-party asset licenses |
| Converter regression tests | Five tests passed: default all-component export, component-only exports without unrelated sources, metadata/key/value preservation, no-overwrite behavior, unknown/mixed revisions, duplicate shards and source-path containment |
| Reference math | Zero/one/two reference cases and padded masks passed; maximum float32 absolute error 8.35e-7; schedule, prompts and mask preprocessing passed |
| Encoder loader regression | BF16, INT8 ConvRot and prior converted encoder passed key mapping, dtype/scales/rotation retention, tokenizer and font checks |
| Renamed node registration | All four new display labels registered from ComfyUI-Fibo2; no duplicate older project folder |
| GUI/API examples | Updated GUI graph converted successfully and agrees with the standalone API graph after private test inputs are supplied |
| Rebuilt checkpoint + downloaded BF16 + standard FLUX.2 VAE | Four-step 1024×704 edit succeeded; pixels identical to the prior BF16 result |
| Rebuilt checkpoint + downloaded INT8 ConvRot + standard FLUX.2 VAE | Four-step 1024×704 edit succeeded; native MixedPrecisionOps selected, with no missing-weight warnings |
| BF16/INT8 comparison | One edit: mean absolute pixel difference 0.145329 on a 0–255 scale; maximum 31. This is not a broad quality benchmark |
| Runtime dependencies / ComfyUI core | Loader/reference/export checks imported no Diffusers; ComfyUI core Git working tree remains clean |

Public source/Hugging Face folders contain no test photographs or generated image examples. Private execution histories and local fixture paths are excluded. Source assets retain their upstream license notices and SHA-256 provenance; release checksums cover all distributed files. Final account namespaces and cross-repository URLs are deferred to publication as documented in RELEASING.md.
