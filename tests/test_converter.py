"""Exercise offline component selection, provenance and no-overwrite behavior."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comfyui", type=Path, default=Path.cwd())
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(args.comfyui.resolve()))
spec = importlib.util.spec_from_file_location("fibo2_converter", Path(__file__).resolve().parents[1] / "convert.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.models = self.root / "models"
        self.source.mkdir()
        self.revision = "a" * 40

    def write(self, name, data):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def tensors(self, name, state):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(path))
        self.write(".cache/huggingface/download/" + name + ".metadata", (self.revision + "\nfixture\n0\n").encode())

    def diffusion(self):
        config = {"_class_name": "BriaFibo2Transformer2DModel", "_ignored": 1, "axes_lens": [1], "hidden_size": 2}
        self.write("transformer/config.json", json.dumps(config).encode())
        self.write(".cache/huggingface/download/transformer/config.json.metadata", (self.revision + "\nfixture\n0\n").encode())
        self.tensors("transformer/part1.safetensors", {"first.weight": torch.ones(2, dtype=torch.bfloat16)})
        self.tensors("transformer/part2.safetensors", {"second.weight": torch.zeros(3, dtype=torch.bfloat16)})
        self.write("transformer/diffusion_pytorch_model.safetensors.index.json", json.dumps({"weight_map": {"first.weight": "part1.safetensors", "second.weight": "part2.safetensors"}}).encode())

    def encoder(self):
        self.tensors("text_encoder/model.safetensors", {
            "model.language_model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "model.visual.weight": torch.zeros(2, dtype=torch.bfloat16),
            "lm_head.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        })
        for name in ("tokenizer.json", "Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"):
            self.write(("processor/" if name == "tokenizer.json" else "fonts/") + name, ("fixture " + name).encode())

    def vae(self):
        self.tensors("vae/diffusion_pytorch_model.safetensors", {
            "encoder.mid_block.attentions.0.to_q.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        })

    def test_diffusion_only_preserves_values_and_provenance(self):
        self.diffusion()
        # An unrelated existing encoder must not block a diffusion-only export.
        other = self.models / "text_encoders" / converter.FILENAMES["text_encoders"]
        other.parent.mkdir(parents=True)
        other.write_bytes(b"existing encoder")
        converter.convert(self.source, self.models, ["diffusion_model"])
        path = self.models / "diffusion_models" / converter.FILENAMES["diffusion_models"]
        with safe_open(str(path), framework="pt") as reader:
            self.assertEqual(set(reader.keys()), {"first.weight", "second.weight"})
            self.assertTrue(torch.equal(reader.get_tensor("first.weight"), torch.ones(2, dtype=torch.bfloat16)))
            self.assertEqual(reader.get_tensor("second.weight").dtype, torch.bfloat16)
            metadata = reader.metadata()
            self.assertEqual(json.loads(metadata["fibo2_config"]), {"hidden_size": 2})
            self.assertEqual(metadata["fibo2_source_revision"], self.revision)
            self.assertEqual(metadata["modelspec.author"], "Bria AI")
            self.assertEqual(metadata["modelspec.license"], "CC-BY-NC-4.0")
            self.assertEqual(metadata["fibo2_converter_version"], "0.3.1")
            self.assertEqual(metadata["fibo2_native_format"], "1")
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            converter.convert(self.source, self.models, ["diffusion_model"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(other.read_bytes(), b"existing encoder")
        self.assertFalse((self.models / "vae").exists())
        self.assertEqual(list(self.models.rglob("*.tmp")), [])

    def test_default_still_exports_all_components(self):
        self.diffusion()
        self.encoder()
        self.vae()
        converter.convert(self.source, self.models)
        self.assertEqual(len(list(self.models.rglob("*.safetensors"))), 3)
        with safe_open(str(self.models / "text_encoders" / converter.FILENAMES["text_encoders"]), framework="pt") as reader:
            self.assertIn("model.embed_tokens.weight", reader.keys())
            self.assertIn("visual.weight", reader.keys())
            self.assertIn("model.lm_head.weight", reader.keys())
            self.assertEqual(sum(key.startswith("_fibo2_assets.") for key in reader.keys()), 5)
            self.assertEqual(reader.metadata()["modelspec.license"], "Apache-2.0")
        with safe_open(str(self.models / "vae" / converter.FILENAMES["vae"]), framework="pt") as reader:
            self.assertEqual(reader.get_tensor("encoder.mid.attn_1.q.weight").shape, (2, 2, 1, 1))
            self.assertNotIn("modelspec.license", reader.metadata())

    def test_each_component_works_without_unselected_sources(self):
        for component, fixture in (("text_encoder", self.encoder), ("vae", self.vae)):
            with self.subTest(component=component):
                fixture()
                target = self.models / component
                converter.convert(self.source, target, [component])
                self.assertEqual(len(list(target.rglob("*.safetensors"))), 1)

    def test_missing_or_mixed_revision_is_not_guessed(self):
        self.assertEqual(converter.source_revision(self.source, ["missing"]), "unknown")
        self.diffusion()
        self.write(".cache/huggingface/download/transformer/part2.safetensors.metadata", ("b" * 40 + "\nfixture\n0\n").encode())
        self.assertEqual(converter.source_revision(self.source, ["transformer/part1.safetensors", "transformer/part2.safetensors"]), "unknown")

    def test_bad_shards_and_path_escape_fail_before_output(self):
        self.diffusion()
        self.tensors("transformer/part2.safetensors", {"first.weight": torch.ones(2)})
        with self.assertRaisesRegex(ValueError, "Duplicate transformer tensors"):
            converter.convert(self.source, self.models, ["diffusion_model"])
        self.assertFalse(self.models.exists())
        with self.assertRaisesRegex(ValueError, "outside the source directory"):
            converter.local_file(self.source, "../outside")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + remaining)
