# Changelog

## 0.3.1 — 2026-10-20

- Rename the project, installed folder, package metadata, node labels and workflow titles to ComfyUI-Fibo2 / Fibo2. Existing `Fibo2*` node IDs and checkpoint format version 1 remain valid.
- Prepare portable installation/download instructions, source attribution, bundled licenses and asset provenance. Workflows require users to select their own input instead of referencing a local test photograph.
- Add converter `--components` selection; omission still converts all three components. Record converter version, component source revision when locally available, and conversion changes. Diffusion exports identify Bria, CC BY-NC 4.0 and commercial-licensing links.
- Add tiny converter regression tests and component-selectable export verification. Rebuild the separate BF16 diffusion release with unchanged neural tensors and enriched metadata.

## 0.3.0 — 2026-10-20

- Accept full Qwen3-VL-4B BF16 and ComfyUI INT8 ConvRot files directly, including native embedding rotation and retained quantization descriptors/scales.
- Bundle unchanged tokenizer and fallback fonts for offline inference without encoder sidecars.
- Validate BF16/INT8 edits with the standard FLUX.2 VAE.

## 0.2.0 — 2026-10-20

- Convert to separate diffusion, encoder and VAE checkpoints in standard ComfyUI folders. Preserve learned values/dtypes; map encoder/VAE keys and VAE attention shapes; embed architecture metadata and encoder assets.
- Replace the release-directory loader with independent diffusion/text loaders and stock VAE loading.

## 0.1.0 — 2026-10-20

- Adapt Bria/Hugging Face's Apache-2.0 reference into ComfyUI operations, optimized attention, model patching and block prefetch.
- Add prompt/vision/hidden-state conditioning, text-image rendering, mask preprocessing, multiple references and resolution-dependent scheduling with standard ComfyUI sampling.
- No Diffusers runtime, ComfyUI core edits, retraining or model quantization.
