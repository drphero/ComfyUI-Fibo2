# ComfyUI-Fibo2

Native ComfyUI support for [Bria AI's Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo): four-step image editing, masks, multiple references and text rendering. Uses ComfyUI operations, sampling, model patching and memory management. No Diffusers runtime or ComfyUI core modifications are required.

This is an unofficial integration. Bria AI created the model; this project does not imply endorsement by Bria, Hugging Face, Qwen or Comfy Org.

## Installation

1. Extract or copy this repository as `ComfyUI/custom_nodes/ComfyUI-Fibo2`. Use ComfyUI's Python environment; its existing dependencies suffice.
2. Place the model files in the folders below. Download the converted diffusion model from [theunlikely/Fibo-2.0-edit-turbo-ComfyUI](https://huggingface.co/theunlikely/Fibo-2.0-edit-turbo-ComfyUI). Alternatively, convert your authorized local Bria download using the command below.
3. Restart ComfyUI. Drag [workflows/fibo2-edit-turbo.json](workflows/fibo2-edit-turbo.json) onto the canvas, upload your own source image in **Load Image**, and write your edit instruction.

```text
ComfyUI/
├── custom_nodes/ComfyUI-Fibo2/
└── models/
    ├── diffusion_models/fibo2_edit_turbo_bf16.safetensors
    ├── text_encoders/qwen3vl_4b_bf16.safetensors
    └── vae/flux2-vae.safetensors
```

| Component | Download / requirement |
| --- | --- |
| Fibo diffusion model | [Converted BF16 checkpoint](https://huggingface.co/theunlikely/Fibo-2.0-edit-turbo-ComfyUI), or the offline converter below. Original: [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo). |
| Text encoder | [Full Qwen3-VL-4B BF16](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors). Complete Qwen3-VL-4B INT8 ConvRot files are also supported; the tested filename is `qwen3vl_4b_int8_convrot.safetensors`. Language model, vision tower and DeepStack must all be present. |
| VAE | Standard [FLUX.2 VAE](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/vae/flux2-vae.safetensors), using the stock **Load VAE** node. |

Follow each provider's license and access terms. Encoder and VAE weights are distributed separately. Other Qwen sizes, pruned/text-only exports, FP8, GGUF and unrelated INT8 formats have not been validated. The original Qwen3-VL-4B-Instruct neural tensors were verified identical to Bria's encoder tensors; Fibo-specific prompt processing and hidden-state selection are handled by this integration.

Tested with ComfyUI **0.42.0**, commit `fd734e95`, frontend **1.55.18**, Python **3.12**, PyTorch **2.14.1+cu132**, and RTX 4090 24 GB. Requires ComfyUI's native Qwen3-VL support, Comfy Kitchen and v3 node API. This is the tested configuration, not a claim that every older release works. See [VALIDATION.md](VALIDATION.md).

## Workflow

| Node | Purpose |
| --- | --- |
| Fibo2 Diffusion Model Loader | Select converted weights from `models/diffusion_models`. |
| Fibo2 Text Encoder Loader | Select full BF16 or INT8 ConvRot Qwen3-VL-4B from `models/text_encoders`. Leave dtype at `default`; quantization scales/rotations are detected automatically. |
| Fibo2 Edit Conditioning | Build image/text conditioning, with optional mask and growing reference-image sockets. |
| Fibo2 Scheduler | Supply Fibo's resolution-dependent schedule. |

The example uses stock `Load VAE`, `Empty Flux 2 Latent`, `BasicGuider`, `RandomNoise`, `KSamplerSelect`, `SamplerCustomAdvanced`, `VAEDecode` and `SaveImage`. Use **four Euler steps**, **BasicGuider (CFG 1)** and dimensions divisible by 16. The two size nodes feed both conditioning and the output latent. Stock diffusion/CLIP loaders do not recognize this integration's architecture; use the two custom loaders.

Prompts accept direct instructions or VGL JSON containing `edit_instruction` and a description of the desired result. `text_render` entries enable text-image conditioning. Connect additional growing reference sockets and refer to inputs as `<image_1>`, `<image_2>`, etc. Image batches supply multiple references. Bria recommends up to four references.

For masks, white marks the region to regenerate. That region is greyed before reference encoding, matching the reference pipeline. Describe its desired contents; an instruction naming only the now-hidden object may leave grey areas. Use stock `ImageCompositeMasked` after decoding for exact preservation outside the mask.

No example photograph is bundled. Select your own image and adapt the prompt. For API use, export the supplied workflow in API format from ComfyUI, set the input image and edit prompt, and submit the exported graph under the native API's `prompt` key.

## Offline conversion

After installing this package, run from your ComfyUI directory using its Python environment:

```bash
python custom_nodes/ComfyUI-Fibo2/convert.py /path/to/Fibo-2.0-edit-turbo --models models --components diffusion_model
```

Omit `--components` to retain the original behavior: convert diffusion model, text encoder and VAE. Select any subset with `--components diffusion_model text_encoder vae`. Existing destination files are never overwritten; original source files remain untouched. Conversion reads local files only and uses ComfyUI's own VAE conversion helper, without importing Diffusers.

Diffusion conversion merges shards and embeds architecture/provenance metadata. Tensor names, values, shapes and dtypes are preserved; no training or quantization takes place. Optional encoder conversion maps Qwen keys and embeds unchanged tokenizer/fonts/notices. Optional VAE conversion maps keys and reshapes attention weights for ComfyUI. Older converted encoder/VAE files remain supported. Inference needs no original Hugging Face directory or model-folder sidecars.

When local Hugging Face download metadata is present and consistent, the converter records the component source revision. Otherwise it records `unknown`; it does not infer a revision from the current remote repository. See [CHANGELOG.md](CHANGELOG.md) for code changes and the [companion model release](https://huggingface.co/theunlikely/Fibo-2.0-edit-turbo-ComfyUI) for checkpoint provenance.

## Verification

Run from ComfyUI after installing this project in `custom_nodes/ComfyUI-Fibo2`:

```bash
python custom_nodes/ComfyUI-Fibo2/tests/test_converter.py --comfyui .
python custom_nodes/ComfyUI-Fibo2/tests/verify_conversion.py --source /path/to/Fibo-2.0-edit-turbo --models models --components diffusion_model
python custom_nodes/ComfyUI-Fibo2/tests/verify_encoder_loading.py --comfyui .
```

The first check uses tiny synthetic weights. The second compares every selected component against its source; omit `--components` to verify all three converted files. The encoder check requires the two downloaded encoders and older converted encoder named in its script; it inspects real loader inputs while intercepting network construction.

For small-model math parity, download `transformer_bria_fibo2.py` from [pinned commit ab96e2b](https://github.com/muhammadali-afridi/diffusers/blob/ab96e2b58c89adb9df418af74d81cc8b1e1be074/src/diffusers/models/transformers/transformer_bria_fibo2.py) to a reference directory, then run:

```bash
python custom_nodes/ComfyUI-Fibo2/tests/verify_reference.py --comfyui . --reference /path/to/reference-directory
```

That test verifies the reference hash and evaluates extracted math without importing Diffusers.

## Licenses and attribution

Integration code: **GPL-3.0-or-later**, with upstream notices and bundled-component exceptions in [THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE). No model weights are included in this code repository.

Bria's weights retain **CC BY-NC 4.0**, as stated in its [model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo); commercial use requires a separate [agreement with Bria](https://platform.bria.ai/licensing/pricing?model=fibo-2). The code's GPL license does not relicense model weights, tokenizer, fonts or externally downloaded components.
