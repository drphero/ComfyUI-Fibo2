"""Check standard BF16, INT8 ConvRot, and converted Fibo text-loader inputs."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comfyui", type=Path, default=Path.cwd())
args = parser.parse_args()
sys.path.insert(0, str(args.comfyui))
package = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fibo_encoder_test", package / "__init__.py", submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from fibo_encoder_test.nodes import Fibo2TextEncoderLoader
from fibo_encoder_test.text_encoder import FiboTokenizer
from fibo_encoder_test.text_images import draw_text_image


baseline = FiboTokenizer()
sample_prompt = '{"edit_instruction":"Make it green","text_render":[{"text":"Grün — 東京"}]}'
expected_ids = baseline.tokenizer.encode(sample_prompt, add_special_tokens=False).ids
expected_text_image = draw_text_image("Grün — 東京", baseline.fonts).tobytes()

# Intercept model construction to inspect exactly what reaches ComfyUI, without
# allocating another 4B network alongside the live ComfyUI service.
for filename, quantized in [
    ("qwen3vl_4b_bf16.safetensors", False),
    ("qwen3vl_4b_int8_convrot.safetensors", True),
    ("qwen3_vl_4b_fibo2_bf16.safetensors", False),
]:
    with patch("comfy.sd.CLIP") as construct:
        Fibo2TextEncoderLoader.execute(filename, "default")
        kwargs = construct.call_args.kwargs
        state = kwargs["state_dict"][0]
        options = kwargs["model_options"]
        assert "visual.deepstack_merger_list.2.linear_fc2.weight" in state
        assert not any(key.startswith("model.language_model.") for key in state)
        weight = state["model.layers.0.self_attn.q_proj.weight"]
        embedding = state["model.embed_tokens.weight"]
        if quantized:
            assert weight.dtype == torch.int8
            assert embedding.dtype == torch.int8
            assert options["quantization_metadata"] == {"mixed_ops": True}
            for key in ["model.layers.0.self_attn.q_proj", "model.embed_tokens"]:
                descriptor = json.loads(state[key + ".comfy_quant"].numpy().tobytes())
                assert descriptor["format"] == "int8_tensorwise"
                assert descriptor["convrot"] is True
                assert descriptor["convrot_groupsize"] == 256
                assert key + ".weight_scale" in state
        else:
            assert weight.dtype == torch.bfloat16
            assert "quantization_metadata" not in options
        tokenizer = FiboTokenizer(tokenizer_data=kwargs["tokenizer_data"])
        assert tokenizer.tokenizer.encode(sample_prompt, add_special_tokens=False).ids == expected_ids
        assert draw_text_image("Grün — 東京", tokenizer.fonts).tobytes() == expected_text_image
        assert set(tokenizer.state_dict()) == {"_fibo2_assets." + name for name in baseline.assets}
        if filename == "qwen3_vl_4b_fibo2_bf16.safetensors":
            assert "tokenizer.json" in kwargs["tokenizer_data"]
        else:
            assert not kwargs["tokenizer_data"]
        print(filename + ": key layout, dtype, quantization, tokenizer, fonts passed", flush=True)
    del kwargs, state, weight, embedding, tokenizer, construct
    gc.collect()
assert "diffusers" not in sys.modules
print("ENCODER LOADER CHECKS PASSED; no Diffusers imported", flush=True)
