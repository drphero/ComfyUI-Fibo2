# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-10-20 for ComfyUI-Fibo2; see NOTICE and CHANGELOG.md.
# Source: diffusers PR #14991, commit ab96e2b58c89adb9df418af74d81cc8b1e1be074.
# Changes: native ComfyUI operations, attention, prefetch, conditioning and model patcher integration.
import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
import comfy.supported_models_base
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder


def gist_count(length, minimum=256, maximum=512, step=64, min_length=256, max_length=2048):
    ratio = min(1.0, max(0.0, (length - min_length) / (max_length - min_length)))
    return min(maximum, max(minimum, math.ceil((minimum + ratio * (maximum - minimum)) / step) * step))


class Attention(nn.Module):
    def __init__(self, dim, heads, eps, kv_dim=None, *, operations, device, dtype):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        kw = dict(device=device, dtype=dtype)
        self.to_q = operations.Linear(dim, dim, bias=False, **kw)
        self.to_k = operations.Linear(kv_dim or dim, dim, bias=False, **kw)
        self.to_v = operations.Linear(kv_dim or dim, dim, bias=False, **kw)
        self.to_out = nn.ModuleList([operations.Linear(dim, dim, bias=False, **kw)])
        self.norm_q = operations.RMSNorm(dim // heads, eps=eps, **kw)
        self.norm_k = operations.RMSNorm(dim // heads, eps=eps, **kw)

    def forward(self, x, context=None, pe=None, mask=None, transformer_options={}):
        context = x if context is None else context
        q = self.to_q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.to_k(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        if pe is not None and not comfy.model_management.in_training:
            qw, _, qs = comfy.ops.cast_bias_weight(self.norm_q, q, offloadable=True)
            kw, _, ks = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, qw, kw, self.norm_q.eps)
            comfy.ops.uncast_bias_weight(self.norm_q, qw, None, qs)
            comfy.ops.uncast_bias_weight(self.norm_k, kw, None, ks)
        else:
            q, k = self.norm_q(q), self.norm_k(k)
            if pe is not None:
                q, k = apply_rope(q, k, pe)
        if mask is not None:
            valid = mask[:, None, None, :].bool()
            mask = torch.zeros(valid.shape, device=q.device, dtype=q.dtype).masked_fill_(~valid, float("-inf"))
        attention = optimized_attention_for_device(x.device, mask=mask is not None)
        out = attention(
            AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
            self.heads, skip_reshape=True, mask=mask, transformer_options=transformer_options,
            preferred_attention=self.comfy_attention,
        )
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim, *, operations, device, dtype):
        super().__init__()
        hidden = (dim * 8 // 3) // 64 * 64
        kw = dict(bias=False, device=device, dtype=dtype)
        self.w1 = operations.Linear(dim, hidden, **kw)
        self.w2 = operations.Linear(hidden, dim, **kw)
        self.w3 = operations.Linear(dim, hidden, **kw)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, eps, cross_dim=None, *, operations, device, dtype):
        super().__init__()
        kw = dict(operations=operations, device=device, dtype=dtype)
        self.attention = Attention(dim, heads, eps, **kw)
        self.feed_forward = FeedForward(dim, **kw)
        for name in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
            setattr(self, name, operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype))
        self.adaLN_modulation = nn.Sequential(operations.Linear(min(dim, 256), dim * 4, device=device, dtype=dtype))
        self.cross_attention = None
        if cross_dim is not None:
            self.cross_attention = Attention(dim, heads, eps, kv_dim=cross_dim, **kw)
            self.cross_attn_norm1 = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
            self.cross_attn_norm2 = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
            self.cross_attn_gate = nn.Parameter(torch.empty(1, device=device, dtype=dtype))

    def forward(self, x, pe, noisy, clean, noisy_slice, context=None, mask=None, text_mask=None, transformer_options={}):
        mod_noisy = self.adaLN_modulation(noisy).unsqueeze(1).chunk(4, dim=-1)
        mod_clean = self.adaLN_modulation(clean).unsqueeze(1).chunk(4, dim=-1)
        start, end = noisy_slice
        # Only target tokens receive the noisy timestep; references and gist receive t=0.
        def select(a, b, gate=False):
            if gate:
                a, b = a.tanh(), b.tanh()
            else:
                a, b = 1 + a, 1 + b
            return torch.cat([b.expand(-1, start, -1), a.expand(-1, end - start, -1), b.expand(-1, x.shape[1] - end, -1)], dim=1)
        scale_a = select(mod_noisy[0], mod_clean[0])
        gate_a = select(mod_noisy[1], mod_clean[1], True)
        scale_f = select(mod_noisy[2], mod_clean[2])
        gate_f = select(mod_noisy[3], mod_clean[3], True)
        attn = self.attention(self.attention_norm1(x) * scale_a, pe=pe, mask=mask, transformer_options=transformer_options)
        x = torch.addcmul(x, gate_a, self.attention_norm2(attn))
        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), context, mask=text_mask, transformer_options=transformer_options)
            x = x + comfy.ops.cast_to_input(self.cross_attn_gate, x).tanh() * self.cross_attn_norm2(cross)
        return torch.addcmul(x, gate_f, self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_f)))


class PerceiverLayer(nn.Module):
    def __init__(self, dim, heads, text_dim, eps, *, operations, device, dtype):
        super().__init__()
        kw = dict(operations=operations, device=device, dtype=dtype)
        self.cross_attn_norm = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.cross_attention = Attention(dim, heads, eps, kv_dim=text_dim, **kw)
        self.self_attn_norm = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.self_attention = Attention(dim, heads, eps, **kw)
        self.ffn_norm = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.ffn = FeedForward(dim, **kw)

    def forward(self, x, text, text_mask, mask, transformer_options):
        x = x + self.cross_attention(self.cross_attn_norm(x), text, mask=text_mask, transformer_options=transformer_options)
        x = x + self.self_attention(self.self_attn_norm(x), mask=mask, transformer_options=transformer_options)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    def __init__(self, dim, heads, text_dim, queries, layers, eps, *, operations, device, dtype):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.empty(queries, dim, device=device, dtype=dtype))
        self.layers = nn.ModuleList([PerceiverLayer(dim, heads, text_dim, eps, operations=operations, device=device, dtype=dtype) for _ in range(layers)])
        self.norm_out = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)

    def forward(self, text, count, text_mask, mask, transformer_options):
        x = comfy.ops.cast_to_input(self.latent_queries, text)[:count].unsqueeze(0).expand(text.shape[0], -1, -1)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for layer in self.layers:
            def core():
                nonlocal x
                x = layer(x, text, text_mask, mask, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope="fibo_perceiver")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="fibo_perceiver")
        return self.norm_out(x)


class FinalLayer(nn.Module):
    def __init__(self, dim, channels, *, operations, device, dtype):
        super().__init__()
        self.norm_final = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = operations.Linear(dim, channels, device=device, dtype=dtype)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(min(dim, 256), dim, device=device, dtype=dtype))

    def forward(self, x, t):
        return self.linear(self.norm_final(x) * (1 + self.adaLN_modulation(t).unsqueeze(1)))


class Fibo2(nn.Module):
    def __init__(self, in_channels=32, patch_size=2, dim=3584, n_layers=44, n_refiner_layers=2,
                 n_heads=28, norm_eps=1e-5, cap_feat_dim=7680, injection_layer_ids=(6, 14, 22, 30, 38),
                 perceiver_num_layers=2, min_num_gist_tokens=256, max_num_gist_tokens=512,
                 gist_step=64, gist_min_text_len=256, gist_max_text_len=2048,
                 rope_theta=256.0, t_scale=1000.0, axes_dims=(32, 48, 48),
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.t_scale = t_scale
        self.max_gist = max_num_gist_tokens
        self.gist_args = (min_num_gist_tokens, max_num_gist_tokens, gist_step, gist_min_text_len, gist_max_text_len)
        self.bundle_indices = {i: j + 1 for j, i in enumerate(injection_layer_ids)}
        kw = dict(operations=operations, device=device, dtype=dtype)
        packed_channels = in_channels * patch_size ** 2
        self.x_embedder = operations.Linear(packed_channels, dim, device=device, dtype=dtype)
        self.t_embedder = TimestepEmbedder(1024, output_size=min(dim, 256), **kw)
        self.rope_embedder = EmbedND(dim // n_heads, rope_theta, list(axes_dims))
        self.noise_refiner = nn.ModuleList([Block(dim, n_heads, norm_eps, **kw) for _ in range(n_refiner_layers)])
        self.perceiver = Perceiver(dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps, **kw)
        self.layers = nn.ModuleList([Block(dim, n_heads, norm_eps, cap_feat_dim if i in injection_layer_ids else None, **kw) for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, packed_channels, **kw)

    @staticmethod
    def image_ids(x, plane):
        h, w = x.shape[-2:]
        rows, cols = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
        return torch.stack([torch.full_like(rows, plane), rows, cols], dim=-1).reshape(1, -1, 3)

    def forward(self, x, timestep, context, reference_latents=None, attention_mask=None, transformer_options={}, **kwargs):
        b, c, h, w = x.shape
        n = h * w
        text = context.unflatten(-1, (6, -1)).movedim(2, 1)
        noisy = self.t_embedder(timestep * self.t_scale, x.dtype)
        clean = self.t_embedder(torch.zeros_like(timestep), x.dtype)
        refs = reference_latents or []
        image_tokens = [z.flatten(2).transpose(1, 2) for z in [x] + refs]
        ids = torch.cat([self.image_ids(z, self.max_gist + 1 + i) for i, z in enumerate([x] + refs)], dim=1)
        image_pe = self.rope_embedder(ids)
        hidden = self.x_embedder(torch.cat(image_tokens, dim=1))
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.noise_refiner), x.device, transformer_options)
        for block in self.noise_refiner:
            def core():
                nonlocal hidden
                hidden = block(hidden, image_pe, noisy, clean, (0, n), transformer_options=transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope="fibo_refiner")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="fibo_refiner")

        lengths = attention_mask.sum(1).tolist() if attention_mask is not None else [text.shape[2]] * b
        counts = [gist_count(length, *self.gist_args) for length in lengths]
        count = max(counts)
        gist_mask = None
        if min(counts) != count:
            gist_mask = torch.arange(count, device=x.device)[None] < torch.tensor(counts, device=x.device)[:, None]
        gist = self.perceiver(text[:, 0], count, attention_mask, gist_mask, transformer_options)
        gist_ids = torch.zeros((1, count, 3), device=x.device)
        gist_ids[:, :, 0] = torch.arange(1, count + 1, device=x.device)
        gist_pe = self.rope_embedder(gist_ids)
        mask = None
        if gist_mask is None:
            start = count
            hidden = torch.cat([gist, hidden], dim=1)
            pe = torch.cat([gist_pe, image_pe], dim=2)
        else:
            start = 0
            mask = torch.cat([torch.ones((b, hidden.shape[1]), device=x.device, dtype=torch.bool), gist_mask], dim=1)
            hidden = torch.cat([hidden, gist], dim=1)
            pe = torch.cat([image_pe, gist_pe], dim=2)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for i, block in enumerate(self.layers):
            cross = text[:, self.bundle_indices[i]] if i in self.bundle_indices else None
            def core():
                nonlocal hidden
                hidden = block(hidden, pe, noisy, clean, (start, start + n), cross, mask, attention_mask, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope="fibo_block")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="fibo_block")
        out = self.final_layer(hidden[:, start:start + n], noisy)
        return out.transpose(1, 2).reshape(b, c, h, w)


class FiboModel(comfy.model_base.BaseModel):
    def __init__(self, config, device=None):
        super().__init__(config, comfy.model_base.ModelType.FLUX, device=device, unet_model=Fibo2)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        # Do not repeat sequences to batch different prompt lengths: gist count depends on length.
        if "cross_attn" in kwargs:
            out["c_crossattn"] = comfy.conds.CONDRegular(kwargs["cross_attn"])
        if kwargs.get("attention_mask") is not None:
            out["attention_mask"] = comfy.conds.CONDRegular(kwargs["attention_mask"])
        if kwargs.get("reference_latents"):
            out["reference_latents"] = comfy.conds.CONDList(kwargs["reference_latents"])
        return out


class FiboConfig(comfy.supported_models_base.BASE):
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Flux2
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    sampling_settings = {"shift": 1.15}
    memory_usage_factor = 4.0

    def get_model(self, state_dict, prefix="", device=None):
        return FiboModel(self, device=device)
