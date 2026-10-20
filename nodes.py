import json
import math

import numpy as np
from PIL import Image
import torch

import comfy.model_management as mm
import comfy.model_patcher
import comfy.sd
import comfy.storage
import comfy.supported_models_base
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .model import FiboConfig
from .text_encoder import FiboTokenizer, FiboTEModel


def load_checkpoint(path, component):
    state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    if metadata.get("fibo2_native_format") != "1" or metadata.get("fibo2_component") != component:
        raise ValueError(f"Select a converted Fibo 2 {component} checkpoint; use this package's convert.py")
    return state, metadata


class Fibo2DiffusionModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Fibo2DiffusionModelLoader", display_name="Fibo2 Diffusion Model Loader", category="loaders/fibo2",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                    io.Combo.Input("dtype", options=["default", "bfloat16", "float32"], default="default")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model_name, dtype):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        state, metadata = load_checkpoint(path, "diffusion_model")
        config = json.loads(metadata["fibo2_config"])
        parameters = comfy.utils.calculate_parameters(state)
        device = mm.get_torch_device()
        choices = {"default": None, "bfloat16": torch.bfloat16, "float32": torch.float32}
        requested_dtype = choices[dtype]
        weight_dtype = requested_dtype or mm.unet_dtype(model_params=parameters, supported_dtypes=[torch.bfloat16, torch.float32], weight_dtype=torch.bfloat16)
        cfg = FiboConfig(config)
        cfg.set_inference_dtype(weight_dtype, mm.unet_manual_cast(weight_dtype, device, cfg.supported_inference_dtypes))
        model = cfg.get_model(state)
        patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=mm.unet_offload_device(), fast_disk=comfy.storage.state_dict_fast_disk(state))
        missing, unexpected = model.diffusion_model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
        if missing or unexpected:
            raise ValueError(f"Fibo transformer weights do not match: missing={missing}, unexpected={unexpected}")
        return io.NodeOutput(patcher)


class Fibo2TextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Fibo2TextEncoderLoader", display_name="Fibo2 Text Encoder Loader", category="loaders/fibo2",
            inputs=[io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                    io.Combo.Input("dtype", options=["default", "bfloat16", "float32"], default="default")],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name, dtype):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        state, _ = comfy.utils.convert_old_quants(state, model_prefix="", metadata=metadata)
        assets = {key.removeprefix("_fibo2_assets."): state.pop(key) for key in list(state) if key.startswith("_fibo2_assets.")}
        state = comfy.utils.state_dict_prefix_replace(state, {
            "model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head.",
        })
        required = ("model.layers.0.input_layernorm.weight", "model.layers.35.input_layernorm.weight",
                    "visual.patch_embed.proj.weight", "visual.deepstack_merger_list.2.linear_fc2.weight")
        if not all(key in state for key in required) or state[required[0]].shape[0] != 2560:
            raise ValueError("Fibo requires a full Qwen3-VL-4B checkpoint including the vision tower and DeepStack")
        target = comfy.supported_models_base.ClipTarget(FiboTokenizer, FiboTEModel)
        weight_dtype = {"default": torch.bfloat16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        model_options = {"dtype": weight_dtype}
        quantization = comfy.utils.detect_layer_quantization(state, "")
        if quantization is not None:
            model_options["quantization_metadata"] = quantization
        clip = comfy.sd.CLIP(target, tokenizer_data=assets, parameters=comfy.utils.calculate_parameters(state), state_dict=[state], model_options=model_options)
        return io.NodeOutput(clip)


def image_size(aspect, area):
    sizes = [(w, h) for w in range(64, 2 * area // 64, 16) for h in range(64, 2 * area // w, 16)
             if 0.9 * area <= w * h <= 1.1 * area]
    return min(sizes, key=lambda size: (round(1 - min(size[0] / size[1] / aspect, aspect / (size[0] / size[1])), 9), abs(size[0] * size[1] - area), -size[0]))


def crop_reference(image, width, height, mask=None):
    image = Image.fromarray((image[..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
    if mask is not None:
        mask = Image.fromarray((mask.cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        mask = mask.point(lambda value: 255 if value >= 128 else 0)
        image = Image.composite(Image.new("RGB", image.size, (128, 128, 128)), image, mask)
    scale = max(height / image.height, width / image.width)
    rw, rh = math.ceil(image.width * scale), math.ceil(image.height * scale)
    image = image.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


class Fibo2EditConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Fibo2EditConditioning", display_name="Fibo2 Edit Conditioning", category="conditioning/fibo2",
            inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Int.Input("width", default=1024, min=16, max=16384, step=16),
                    io.Int.Input("height", default=1024, min=16, max=16384, step=16),
                    io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=1)),
                    io.Mask.Input("mask", optional=True),
                    io.Boolean.Input("render_text_images", default=True, advanced=True)],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, images, mask=None, render_text_images=True):
        references = []
        for name in sorted(images, key=lambda name: int(name.removeprefix("image"))):
            batch = images[name]
            if batch is not None:
                references.extend(batch)
        vae_latents, vision_images = [], []
        for i, image in enumerate(references):
            w, h = (width, height) if len(references) == 1 else image_size(image.shape[1] / image.shape[0], width * height)
            crop = crop_reference(image, w, h, mask[0] if mask is not None and i == 0 else None)
            pixels = torch.from_numpy(np.array(crop).astype(np.float32) / 255.0).unsqueeze(0)
            vae_latents.append(vae.encode(pixels))
            thumb = crop.copy()
            thumb.thumbnail((768, 768))
            vision_images.append(thumb)
        tokens = clip.tokenize(prompt, images=vision_images, render_text_images=render_text_images)
        conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"reference_latents": vae_latents})
        return io.NodeOutput(conditioning)


class Fibo2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Fibo2Scheduler", display_name="Fibo2 Scheduler", category="sampling/custom_sampling/schedulers",
            inputs=[io.Latent.Input("latent"), io.Int.Input("steps", default=4, min=1, max=10000)],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, latent, steps):
        h, w = latent["samples"].shape[-2:]
        mu = 0.5 + (h * w - 256) * (1.15 - 0.5) / (4096 - 256)
        shift = math.exp(mu)
        sigmas = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
        sigmas = shift / (shift + (1.0 / sigmas - 1.0))
        return io.NodeOutput(torch.cat([sigmas.float(), torch.zeros(1)]))
