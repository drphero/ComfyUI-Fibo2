# Licenses and sources

The root [LICENSE](LICENSE) applies to this integration as GPL-3.0-or-later. It does not replace these upstream licenses or apply to separately downloaded weights. Preserve this file, [NOTICE](NOTICE) and the referenced license texts in redistributions.

| Component | Source and modifications | License / notice |
| --- | --- | --- |
| Upstream portions of `model.py`, `prompt.py`, `text_images.py` | Bria AI / HuggingFace Team, [Diffusers PR #14991](https://github.com/huggingface/diffusers/pull/14991), commit `ab96e2b58c89adb9df418af74d81cc8b1e1be074`. Modifications identified in file headers and CHANGELOG. | [Apache-2.0](licenses/Apache-2.0.txt); copyright 2026 Bria AI and The HuggingFace Team. Notices retained within the combined GPL-3.0-or-later integration. |
| `assets/tokenizer.json` | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct); bundled unchanged from Bria's processor export. | [Apache-2.0](licenses/Apache-2.0.txt). |
| Template portions of `examples/*.json` | [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates), `templates/image_flux2_klein_image_edit_4b_distilled.json`; sampling graph adapted for Fibo. | [MIT](licenses/Comfy-Org-workflow-templates-MIT.txt), copyright (c) 2023-present Comfy Org. |
| `assets/Aileron-Regular.otf` | Aileron Regular 1.102, Sora Sagano (dot colon); unchanged. | [Supplied notice](assets/LICENSE-Aileron.txt), [CC0-1.0](licenses/CC0-1.0.txt). |
| `assets/DejaVuSans.ttf` | DejaVu Fonts; unchanged. Bitstream copyright 2003; DejaVu changes public domain. | [Supplied Bitstream Vera notice](assets/LICENSE-DejaVu.txt). Its `debian/*` GPL-2+ paragraph concerns Debian packaging, which is not bundled. |

[assets/SOURCES.json](assets/SOURCES.json) records bundled asset hashes and origins. No upstream photographs or implementation source trees are bundled. ComfyUI is an external GPL-3.0 dependency; PyTorch, Pillow and other dependencies retain their respective licenses.

## Separately obtained weights

- **Fibo 2.0 Edit Turbo:** Bria AI's [model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) states CC BY-NC 4.0. Non-commercial use under that license; commercial use requires a [separate agreement with Bria](https://platform.bria.ai/licensing/pricing?model=fibo-2). Upstream access is gated. The companion model package preserves upstream license metadata and records conversion changes.
- **Qwen3-VL-4B-Instruct:** the [official card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) identifies Apache-2.0. Quantized distributions may include distributor notices.
- **FLUX.2 VAE:** obtain it from the [linked provider](https://huggingface.co/Comfy-Org/flux2-dev) under its applicable terms. This project neither redistributes nor assigns a license to it.
