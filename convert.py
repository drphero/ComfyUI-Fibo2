"""Convert a local Fibo 2 release into standalone ComfyUI checkpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comfy.diffusers_convert import convert_vae_state_dict


CONVERTER_VERSION = "0.3.1"
SOURCE_REPO = "briaai/Fibo-2.0-edit-turbo"
COMPONENT_FOLDERS = {"diffusion_model": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}


FILENAMES = {
    "diffusion_models": "fibo2_edit_turbo_bf16.safetensors",
    "text_encoders": "qwen3_vl_4b_fibo2_bf16.safetensors",
    "vae": "fibo2_vae.safetensors",
}


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Model component path is outside the source directory")
    return path


def read_tensors(path):
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        return {key: reader.get_tensor(key) for key in reader.keys()}


def source_revision(source, files):
    revisions = set()
    for name in files:
        cached = local_file(source, ".cache/huggingface/download/" + name + ".metadata")
        if not cached.is_file():
            return "unknown"
        revisions.add(cached.read_text().splitlines()[0])
    return revisions.pop() if len(revisions) == 1 else "unknown"


def write_checkpoint(state, path, component, architecture, config=None, revision="unknown"):
    metadata = {
        "fibo2_native_format": "1",
        "fibo2_component": component,
        "modelspec.architecture": architecture,
        "modelspec.implementation": "ComfyUI",
        "modelspec.source": SOURCE_REPO,
        "fibo2_source_revision": revision,
        "fibo2_converter_version": CONVERTER_VERSION,
    }
    if component == "diffusion_model":
        metadata.update({
            "modelspec.author": "Bria AI",
            "modelspec.license": "CC-BY-NC-4.0",
            "fibo2_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "fibo2_commercial_license_url": "https://platform.bria.ai/licensing/pricing?model=fibo-2",
            "fibo2_conversion_changes": "Merged transformer shards; embedded architecture and provenance metadata; tensor names, shapes, dtypes and values unchanged; no training or quantization.",
        })
    elif component == "text_encoder":
        metadata.update({
            "modelspec.base_model": "Qwen/Qwen3-VL-4B-Instruct",
            "modelspec.author": "Qwen Team, Alibaba Cloud",
            "modelspec.license": "Apache-2.0",
            "fibo2_asset_licenses": json.dumps({"tokenizer.json": "Apache-2.0", "Aileron-Regular.otf": "CC0-1.0", "DejaVuSans.ttf": "Bitstream-Vera"}),
            "fibo2_conversion_changes": "Mapped Qwen keys; embedded unchanged tokenizer, fonts and font notices; neural tensor values and dtypes unchanged.",
        })
    else:
        metadata["fibo2_conversion_changes"] = "Converted VAE keys and attention weight shapes with ComfyUI; learned values and dtypes unchanged. Refer to the source model and VAE provider for licensing."
    if config is not None:
        metadata["fibo2_config"] = json.dumps(config, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(state, str(temporary), metadata=metadata)
        # Publish only complete files and refuse to replace an existing checkpoint.
        os.link(temporary, path)
    finally:
        temporary.unlink()
    print(f"Wrote {path} ({path.stat().st_size:,} bytes, {len(state)} tensors)", flush=True)


def convert(source, models, components=("diffusion_model", "text_encoder", "vae")):
    source = source.resolve()
    selected = {COMPONENT_FOLDERS[component] for component in components}
    targets = {folder: models / folder / name for folder, name in FILENAMES.items() if folder in selected}
    for target in targets.values():
        if target.exists():
            raise FileExistsError(f"Checkpoint already exists: {target}")
    if "diffusion_models" in selected:
        config = json.loads(local_file(source, "transformer/config.json").read_text())
        if config.pop("_class_name") != "BriaFibo2Transformer2DModel":
            raise ValueError("The source directory must contain a Fibo 2 transformer")
        config = {key: value for key, value in config.items() if not key.startswith("_") and key != "axes_lens"}
        index_path = local_file(source, "transformer/diffusion_pytorch_model.safetensors.index.json")
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            shards = sorted(set(weight_map.values()))
        else:
            weight_map = None
            shards = ["diffusion_pytorch_model.safetensors"]
        print("Merging the transformer shards; tensor values and dtypes are preserved", flush=True)
        state = {}
        for shard in shards:
            tensors = read_tensors(local_file(source, "transformer/" + shard))
            if state.keys() & tensors.keys():
                raise ValueError(f"Duplicate transformer tensors in {shard}")
            state.update(tensors)
        del tensors
        if weight_map is not None and state.keys() != weight_map.keys():
            raise ValueError("Transformer shards do not match the source index")
        write_checkpoint(state, targets["diffusion_models"], "diffusion_model", "fibo2/dit", config, revision=source_revision(source, ["transformer/config.json"] + ["transformer/" + shard for shard in shards]))
        del state
        gc.collect()

    if "text_encoders" in selected:
        print("Converting text-encoder keys and embedding its tokenizer, fonts, and font licenses", flush=True)
        original = read_tensors(local_file(source, "text_encoder/model.safetensors"))
        prefixes = {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."}
        state = {}
        for key, value in original.items():
            prefix = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
            name = prefixes[prefix] + key[len(prefix):] if prefix is not None else key
            if name in state:
                raise ValueError(f"Duplicate text-encoder tensor: {name}")
            state[name] = value
        del original
        for name, relative in {
            "tokenizer.json": "processor/tokenizer.json",
            "Aileron-Regular.otf": "fonts/Aileron-Regular.otf",
            "DejaVuSans.ttf": "fonts/DejaVuSans.ttf",
            "LICENSE-Aileron.txt": "fonts/LICENSE-Aileron.txt",
            "LICENSE-DejaVu.txt": "fonts/LICENSE-DejaVu.txt",
        }.items():
            state["_fibo2_assets." + name] = torch.frombuffer(bytearray(local_file(source, relative).read_bytes()), dtype=torch.uint8)
        write_checkpoint(state, targets["text_encoders"], "text_encoder", "fibo2/qwen3_vl_4b", revision=source_revision(source, ["text_encoder/model.safetensors"]))
        del state
        gc.collect()

    if "vae" in selected:
        print("Converting the VAE to ComfyUI keys and convolution weight shapes", flush=True)
        state = convert_vae_state_dict(read_tensors(local_file(source, "vae/diffusion_pytorch_model.safetensors")))
        write_checkpoint(state, targets["vae"], "vae", "flux2/vae", revision=source_revision(source, ["vae/diffusion_pytorch_model.safetensors"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Hugging Face model directory")
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[2] / "models", help="ComfyUI models directory")
    parser.add_argument("--components", nargs="+", choices=list(COMPONENT_FOLDERS), default=list(COMPONENT_FOLDERS), help="Components to convert (default: all three)")
    args = parser.parse_args()
    convert(args.source, args.models, args.components)
