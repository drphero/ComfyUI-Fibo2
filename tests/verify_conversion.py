"""Verify that the on-disk conversion preserves every weight and tokenizer asset."""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--comfyui", type=Path, default=Path.cwd())
parser.add_argument("--components", nargs="+", choices=["diffusion_model", "text_encoder", "vae"], default=["diffusion_model", "text_encoder", "vae"])
args = parser.parse_args()
sys.path.insert(0, str(args.comfyui))
from comfy.diffusers_convert import convert_vae_state_dict

torch.set_num_threads(8)


def check_tensor(actual, expected, key):
    assert actual.shape == expected.shape, key
    assert actual.dtype == expected.dtype, key
    assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key


source = args.source
if "diffusion_model" in args.components:
    index = json.loads((source / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    with safe_open(str(args.models / "diffusion_models/fibo2_edit_turbo_bf16.safetensors"), framework="pt") as converted:
        assert converted.metadata()["fibo2_component"] == "diffusion_model"
        assert converted.metadata()["fibo2_native_format"] == "1"
        original_config = json.loads((source / "transformer/config.json").read_text())
        expected_config = {key: value for key, value in original_config.items() if not key.startswith("_") and key != "axes_lens"}
        assert json.loads(converted.metadata()["fibo2_config"]) == expected_config
        assert set(converted.keys()) == set(index["weight_map"])
        for shard in sorted(set(index["weight_map"].values())):
            with safe_open(str(source / "transformer" / shard), framework="pt") as original:
                for key in original.keys():
                    check_tensor(converted.get_tensor(key), original.get_tensor(key), key)
    print("Transformer: every tensor matched bit for bit", flush=True)

if "text_encoder" in args.components:
    assets = {}
    with safe_open(str(args.models / "text_encoders/qwen3_vl_4b_fibo2_bf16.safetensors"), framework="pt") as converted:
        with safe_open(str(source / "text_encoder/model.safetensors"), framework="pt") as original:
            expected_keys = set()
            for key in original.keys():
                native_key = key.replace("model.language_model.", "model.", 1).replace("model.visual.", "visual.", 1)
                if native_key.startswith("lm_head."):
                    native_key = "model." + native_key
                check_tensor(converted.get_tensor(native_key), original.get_tensor(key), key)
                expected_keys.add(native_key)
            assert set(converted.keys()) - expected_keys == {
                "_fibo2_assets." + name for name in ["tokenizer.json", "Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"]
            }
        for key in converted.keys():
            if key.startswith("_fibo2_assets."):
                name = key.removeprefix("_fibo2_assets.")
                data = converted.get_tensor(key).numpy().tobytes()
                relative = "processor/tokenizer.json" if name == "tokenizer.json" else "fonts/" + name
                assert data == (source / relative).read_bytes(), name
                assets[name] = data
    print("Text encoder: every tensor and embedded asset matched bit for bit", flush=True)

    native_tokenizer = Tokenizer.from_str(assets["tokenizer.json"].decode("utf-8"))
    original_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(source / "processor"), local_files_only=True)
    for prompt in ["Make the jacket red", '{"edit_instruction":"Add FIBO","text_render":[{"text":"FIBO"}]}',
                   '<|im_start|>user\n{"image":"<|vision_start|><|image_pad|><|vision_end|>","edit_instruction":"Grün — 東京"}<|im_end|>\n<|im_start|>assistant\n']:
        assert native_tokenizer.encode(prompt, add_special_tokens=False).ids == original_tokenizer.encode(prompt, add_special_tokens=False)
    print("Embedded tokenizer: matched original token IDs", flush=True)

if "vae" in args.components:
    with safe_open(str(source / "vae/diffusion_pytorch_model.safetensors"), framework="pt") as original:
        expected = convert_vae_state_dict({key: original.get_tensor(key) for key in original.keys()})
    with safe_open(str(args.models / "vae/fibo2_vae.safetensors"), framework="pt") as converted:
        assert set(converted.keys()) == set(expected)
        assert "decoder.up.3.block.0.norm1.weight" in converted.keys()
        assert "bn.running_mean" in converted.keys()
        for key, tensor in expected.items():
            check_tensor(converted.get_tensor(key), tensor, key)
    print("VAE: every tensor matched the native ComfyUI layout bit for bit", flush=True)

assert "diffusers" not in sys.modules
print("CONVERSION VERIFIED; no Diffusers imported", flush=True)
