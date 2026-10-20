import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torchvision.transforms.v2 import functional as TF

from comfy import sd1_clip
from comfy.text_encoders.qwen3vl import Qwen3VL

from .prompt import Prompt
from .text_images import draw_text_image, load_font


BUNDLES = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAPS = sorted({i for bundle in BUNDLES for i in bundle})


def vision_pixels(image):
    # Qwen2VLImageProcessor, with the release's patch size and normalization.
    pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
    h, w = pixels.shape[-2:]
    rh, rw = round(h / 32) * 32, round(w / 32) * 32
    if rh * rw > 1605632:
        scale = math.sqrt(h * w / 1605632)
        rh, rw = max(32, math.floor(h / scale / 32) * 32), max(32, math.floor(w / scale / 32) * 32)
    elif rh * rw < 3136:
        scale = math.sqrt(3136 / (h * w))
        rh, rw = math.ceil(h * scale / 32) * 32, math.ceil(w * scale / 32) * 32
    pixels = TF.resize(pixels, [rh, rw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    pixels = (pixels.float() - 127.5) / 127.5
    gh, gw = rh // 16, rw // 16
    patches = pixels.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16).permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(gh * gw, 1536)
    return patches, torch.tensor([[1, gh, gw]], dtype=torch.long)


class FiboTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.assets = tokenizer_data.copy()
        for name in ("tokenizer.json", "Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"):
            if name not in self.assets:
                data = (Path(__file__).parent / "assets" / name).read_bytes()
                self.assets[name] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self.tokenizer = Tokenizer.from_str(self.assets["tokenizer.json"].numpy().tobytes().decode("utf-8"))
        self.fonts = [load_font(self.assets[name].numpy().tobytes()) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")]

    def tokenize_with_weights(self, text, return_word_ids=False, images=(), render_text_images=True, **kwargs):
        caption, _ = Prompt._parse_prompt(text)
        text_images = []
        if render_text_images and isinstance(caption, dict):
            text_images = [draw_text_image(entry["text"], self.fonts) for entry in caption.get("text_render", [])
                           if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]]
        if images:
            text = Prompt._get_edit_prompt(text, len(images), bool(text_images))
        all_images = list(images) + text_images
        text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        result = []
        image_iter = iter(all_images)
        for token in tokens:
            if token == 151655:
                image = next(image_iter)
                pixels, grid = vision_pixels(image)
                token = {"type": "image", "data": (pixels, grid)}
            result.append((token, 1.0, 0) if return_word_ids else (token, 1.0))
        return {"qwen3vl_4b": [result]}

    def state_dict(self):
        return {"_fibo2_assets." + name: data for name, data in self.assets.items()}


class FiboQwenVL(Qwen3VL):
    model_type = "qwen3vl_4b"

    def preprocess_embed(self, embed, device):
        pixels, grid = embed["data"]
        merged, deepstack = self.visual(pixels.to(device=device, dtype=torch.float32), grid)
        return merged, {"grid": grid, "deepstack": deepstack}


class FiboClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, layer=TAPS, layer_idx=None, textmodel_json_config={}, dtype=dtype,
                         special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=FiboQwenVL,
                         enable_attention_masks=True, return_attention_masks=True, model_options=model_options)


class FiboTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=FiboClipModel, model_options=model_options)
        self.feature_dtype = dtype

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        # Native Llama's list taps and HF hidden_states both index the input to block k.
        normalized = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-6)
        normalized = normalized.to(self.feature_dtype)
        by_layer = {layer: normalized[:, i] for i, layer in enumerate(TAPS)}
        bundles = [torch.cat([by_layer[layer] for layer in bundle], dim=-1) for bundle in BUNDLES]
        out = torch.stack(bundles, dim=2).flatten(2)
        if bool(extra["attention_mask"].all()):
            extra.pop("attention_mask")
        return out, pooled, extra
