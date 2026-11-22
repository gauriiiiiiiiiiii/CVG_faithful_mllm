"""CVG-E Model: LLaVA wrapper with CVG contrastive decoding pipeline.

Integrates VTD, DAC, GFF modules with a pre-trained MLLM for
hallucination-mitigated generation. Supports both CVG (dual-pass) and
CVG-E (single-pass approximation) decoding.

Key architecture:
- Embedding-level branch separation: chosen branch zeros out rejected
  visual token embeddings, rejected branch zeros out chosen ones.
- KV-cached autoregressive loop: prefill once per branch, then one
  forward pass per token per branch with KV cache.
- GFF fusion at each decoding step: compare chosen/rejected logits via
  KL divergence, apply contrastive decoding when divergent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlavaProcessor

from cvg.approximation import SinglePassBranchApproximation
from cvg.hooks import SingleLayerHook
from cvg.modules import (
    CVGConfig,
    CVGEConfig,
    DynamicAttentionCalibration,
    GroundingFlowFusion,
    VisualTokenDisentanglement,
)


@dataclass
class GenerationResult:
    """Result from CVG generation."""

    text: str
    tokens: list[int]
    generation_time: float
    num_steps: int
    kl_divergences: list[float] = field(default_factory=list)
    contrastive_applied_count: int = 0
    total_steps: int = 0
    alpha_used: float = 0.0
    flops_estimate: Optional[float] = None


class CVGModel:
    """MLLM with CVG decoding pipeline.

    Wraps LlavaForConditionalGeneration and applies CVG's three modules
    (VTD, DAC, GFF) during autoregressive generation.

    Two generation modes:
    - generate_with_cvg: dual-pass (chosen + rejected branches) + GFF fusion
    - generate_with_cvg_e: single-pass (chosen + approximated rejected) + GFF
    """

    def __init__(
        self,
        model_name: str = "bczhou/tiny-llava-v1-hf",
        config: Optional[CVGConfig] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_flash: bool = False,
    ):
        self.config = config or CVGConfig()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        attn_impl = "sdpa" if use_flash else "eager"

        print(f"Loading {model_name} on {self.device} ({dtype}, attn={attn_impl})...")
        auto_proc = AutoProcessor.from_pretrained(model_name)
        image_processor = auto_proc.image_processor
        tokenizer = auto_proc.tokenizer

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        ).to(self.device)

        # Re-read patch_size from loaded model config
        vis_cfg = self.model.config.vision_config
        patch_size = getattr(vis_cfg, "patch_size", 14)
        image_size = getattr(vis_cfg, "image_size", 336)
        image_processor.patch_size = patch_size
        image_processor.num_additional_image_tokens = 0

        self.processor = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            num_additional_image_tokens=0,
        )

        self.vtd = VisualTokenDisentanglement(self.config)
        self.dac = DynamicAttentionCalibration(self.config)
        self.gff = GroundingFlowFusion(self.config)

        if isinstance(self.config, CVGEConfig) and self.config.use_single_pass:
            self.approx = SinglePassBranchApproximation(self.config)
        else:
            self.approx = None

        self._hook = None  # installed after _detect_architecture
        self._detect_architecture()

        if self.approx is not None:
            self._hook = SingleLayerHook(self._lm, self.last_approx_layer)
            self._hook.install()

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    def _detect_architecture(self):
        """Detect layer counts and clamp config indices to valid ranges."""
        self._lm = self._resolve_attr("language_model")
        self._vt = self._resolve_attr("vision_tower")
        self._projector = self._resolve_attr("multi_modal_projector")

        lm_config = self._lm.config
        self.num_lm_layers = lm_config.num_hidden_layers
        self.num_heads = lm_config.num_attention_heads
        self.hidden_dim = lm_config.hidden_size
        self.head_dim = self.hidden_dim // self.num_heads

        if hasattr(self._vt, "vision_model") and hasattr(self._vt.vision_model, "encoder"):
            self.num_vision_layers = len(self._vt.vision_model.encoder.layers)
        else:
            self.num_vision_layers = 24

        self.image_token_id = getattr(self.model.config, "image_token_index", 32000)

        vis_cfg = self.model.config.vision_config
        patch_size = getattr(vis_cfg, "patch_size", 14)
        image_size = getattr(vis_cfg, "image_size", 336)
        self.grid_size = image_size // patch_size
        self.num_visual_tokens = self.grid_size ** 2

        def _clamp(indices, n, fallback):
            out = [i for i in indices if i < n]
            return out if out else list(range(max(0, n - fallback), n))

        self.cross_attn_layers = _clamp(self.config.cross_attn_layers, self.num_lm_layers, 4)
        self.self_attn_layers = _clamp(self.config.self_attn_layers, self.num_vision_layers, 3)
        self.dac_layers = _clamp(self.config.dac_layers, self.num_lm_layers, 10)

        if isinstance(self.config, CVGEConfig):
            self.approx_layers = _clamp(self.config.approx_layers, self.num_lm_layers, 5)
            self.last_approx_layer = self.approx_layers[-1]
        else:
            self.approx_layers = []
            self.last_approx_layer = self.num_lm_layers - 1

        print(f"  LM: {self.num_lm_layers} layers, Vision: {self.num_vision_layers} layers")
        print(f"  Visual tokens: {self.num_visual_tokens} ({self.grid_size}x{self.grid_size})")
        print(f"  Cross-attn layers: {self.cross_attn_layers}")
        print(f"  Self-attn layers:  {self.self_attn_layers}")

    def _resolve_attr(self, name):
        """Resolve model sub-module (handles both direct and nested .model access)."""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        if hasattr(self.model, "model") and hasattr(self.model.model, name):
            return getattr(self.model.model, name)
        raise AttributeError(f"Cannot find {name} in model")

    def _get_lm_layers(self):
        """Get the list of decoder layers."""
        base = self._lm.model if hasattr(self._lm, "model") else self._lm
        return base.layers

    def _get_lm_head(self):
        if hasattr(self._lm, "lm_head"):
            return self._lm.lm_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise AttributeError("Cannot find lm_head")

    def _get_final_norm(self):
        base = self._lm.model if hasattr(self._lm, "model") else self._lm
        if hasattr(base, "norm"):
            return base.norm
        return None

    def _get_lm_embed(self):
        return self._lm.get_input_embeddings()

    # ------------------------------------------------------------------
    # Embedding preparation
    # ------------------------------------------------------------------

    def _prepare_embeddings(self, inputs):
        """Run vision encoder, merge with text embeddings, return positions.

        Returns:
            merged: (B, merged_len, hidden_dim)
            vis_start: start of visual tokens in merged sequence
            vis_end: end of visual tokens (exclusive)
            attn_mask: (B, merged_len)
        """
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")

        with torch.no_grad():
            vis_out = self._vt(pixel_values, output_hidden_states=True)
            img_feats = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out[0]
            img_feats = self._projector(img_feats)

        text_embeds = self.model.get_input_embeddings()(input_ids)

        img_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) == 0:
            return text_embeds, 0, 0, inputs.get("attention_mask")

        img_pos = img_positions[0].item()
        n_img = img_feats.shape[1]
        self.num_visual_tokens = n_img
        self.grid_size = int(math.sqrt(n_img)) if n_img > 0 else 24

        merged = torch.cat(
            [text_embeds[:, :img_pos], img_feats, text_embeds[:, img_pos + 1:]],
            dim=1,
        )

        orig_mask = inputs.get("attention_mask")
        if orig_mask is not None:
            merged_mask = torch.cat([
                orig_mask[:, :img_pos],
                torch.ones(1, n_img, device=orig_mask.device, dtype=orig_mask.dtype),
                orig_mask[:, img_pos + 1:],
            ], dim=1)
        else:
            merged_mask = torch.ones(1, merged.shape[1], device=merged.device, dtype=torch.long)

        return merged, img_pos, img_pos + n_img, merged_mask

    # ------------------------------------------------------------------
    # VTD score computation
    # ------------------------------------------------------------------

    def _compute_vtd_scores(self, inputs, merged, vis_start, vis_end, attn_mask):
        """Compute VTD significance and text-protection scores.

        Runs LM forward to extract cross-attention proxy (text→visual attention)
        and vision tower forward to get ViT self-attention.
        """
        seq_len = merged.shape[1]
        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        with torch.no_grad():
            lm_out = self._lm(
                inputs_embeds=merged, attention_mask=attn_mask,
                position_ids=pos_ids, output_attentions=True, return_dict=True,
            )

        cross_maps = {}
        text_idx = list(range(0, vis_start)) + list(range(vis_end, seq_len))
        for li in self.cross_attn_layers:
            if li < len(lm_out.attentions):
                attn = lm_out.attentions[li]
                if text_idx:
                    cross_maps[li] = attn[:, :, text_idx, vis_start:vis_end]

        vis_self_maps = {}
        try:
            with torch.no_grad():
                vis_out = self._vt(inputs.get("pixel_values"), output_attentions=True)
            if hasattr(vis_out, "attentions") and vis_out.attentions:
                for li in self.self_attn_layers:
                    if li < len(vis_out.attentions):
                        vis_self_maps[li] = vis_out.attentions[li]
        except Exception:
            pass

        num_text = seq_len - (vis_end - vis_start)
        significance = self.vtd.extract_attention_from_hooks(
            cross_maps, vis_self_maps,
            self.num_visual_tokens, num_text, self.num_heads,
        )
        text_scores = self.vtd.compute_text_protection(
            vis_self_maps, self.num_visual_tokens, self.grid_size,
        )
        return significance, text_scores

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_vanilla(self, image, question: str, max_new_tokens: int = 512, **kw) -> GenerationResult:
        """Baseline: standard model.generate() without CVG."""
        t0 = time.time()
        img_token = self.processor.image_token
        text = img_token + " " + question
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        n = inputs["input_ids"].shape[1]
        text = self.processor.decode(out[0][n:], skip_special_tokens=True)
        return GenerationResult(
            text=text, tokens=out[0][n:].tolist(),
            generation_time=time.time() - t0, num_steps=len(out[0]) - n,
        )

    def generate_with_cvg(
        self, image, question: str, max_new_tokens: int = 512,
        temperature: float = 1.0, do_sample: bool = False, verbose: bool = False,
    ) -> GenerationResult:
        """Generate using CVG decoding (dual-pass + GFF fusion)."""
        t0 = time.time()
        img_token = self.processor.image_token
        text = img_token + " " + question
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        merged, vis_s, vis_e, mask = self._prepare_embeddings(inputs)
        sig, tscores = self._compute_vtd_scores(inputs, merged, vis_s, vis_e, mask)
        ch_m, rj_m, alpha = self.vtd.partition_tokens(sig, tscores)

        if verbose:
            print(f"VTD: α={alpha[0]:.3f}, chosen={ch_m.sum().item()}, rejected={rj_m.sum().item()}")

        ch_e, rj_e = self._branch_embeds(merged, vis_s, ch_m, rj_m)
        vis_mask = torch.zeros(merged.shape[1], dtype=torch.bool, device=self.device)
        vis_mask[vis_s:vis_e] = True

        result = self._dual_pass_generate(
            ch_e, rj_e, mask, vis_mask, max_new_tokens, temperature, do_sample, verbose,
        )
        result.generation_time = time.time() - t0
        result.alpha_used = alpha[0].item()
        return result

    def generate_with_cvg_e(
        self, image, question: str, max_new_tokens: int = 512,
        temperature: float = 1.0, do_sample: bool = False, verbose: bool = False,
    ) -> GenerationResult:
        """Generate using CVG-E (single-pass + approximated rejected + GFF)."""
        if self.approx is None:
            raise ValueError("Single-pass not enabled. Use CVGEConfig(use_single_pass=True).")
        t0 = time.time()
        img_token = self.processor.image_token
        text = img_token + " " + question
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        merged, vis_s, vis_e, mask = self._prepare_embeddings(inputs)
        sig, tscores = self._compute_vtd_scores(inputs, merged, vis_s, vis_e, mask)
        ch_m, rj_m, alpha = self.vtd.partition_tokens(sig, tscores)

        ch_e = merged.clone()
        rj_idx = rj_m[0].nonzero(as_tuple=True)[0]
        if len(rj_idx) > 0:
            ch_e[:, vis_s + rj_idx] *= 0.01

        vis_mask = torch.zeros(merged.shape[1], dtype=torch.bool, device=self.device)
        vis_mask[vis_s:vis_e] = True

        result = self._single_pass_generate(
            ch_e, mask, vis_mask, max_new_tokens, temperature, do_sample, verbose,
        )
        result.generation_time = time.time() - t0
        result.alpha_used = alpha[0].item()
        return result

    # ------------------------------------------------------------------
    # Branch embedding construction
    # ------------------------------------------------------------------

    @staticmethod
    def _branch_embeds(merged, vis_start, chosen_mask, rejected_mask,
                        suppress_factor=0.01):
        """Create branch embeddings by suppressing (not zeroing) visual tokens.

        Full zeroing destabilizes small models. Scale factor 0.01 preserves
        positional structure while removing most visual signal.
        """
        ch = merged.clone()
        rj = merged.clone()
        ri = rejected_mask[0].nonzero(as_tuple=True)[0]
        if len(ri) > 0:
            ch[:, vis_start + ri] *= suppress_factor
        ci = chosen_mask[0].nonzero(as_tuple=True)[0]
        if len(ci) > 0:
            rj[:, vis_start + ci] *= suppress_factor
        return ch, rj

    # ------------------------------------------------------------------
    # Dual-pass generation (original CVG)
    # ------------------------------------------------------------------

    def _dual_pass_generate(
        self, ch_emb, rj_emb, attn_mask, vis_mask,
        max_new_tokens, temperature, do_sample, verbose,
    ) -> GenerationResult:
        """Autoregressive generation with KV-cached dual branches + GFF."""
        seq_len = ch_emb.shape[1]
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        lm = self._lm

        with torch.no_grad():
            c_out = lm(inputs_embeds=ch_emb, attention_mask=attn_mask,
                        position_ids=pos, use_cache=True, return_dict=True)
            r_out = lm(inputs_embeds=rj_emb, attention_mask=attn_mask,
                        position_ids=pos, use_cache=True, return_dict=True)

        c_kv, r_kv = c_out.past_key_values, r_out.past_key_values
        c_log = self._get_lm_head()(c_out.last_hidden_state)[:, -1, :]
        r_log = self._get_lm_head()(r_out.last_hidden_state)[:, -1, :]

        gen, kls = [], []
        n_ctr = 0
        cur = seq_len
        eos = self.processor.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            kl = self.gff.kl_divergence(c_log, r_log)
            kls.append(kl)
            final = self.gff.adaptive_contrastive_decoding(c_log, r_log, kl) if kl >= self.config.epsilon else c_log
            if kl >= self.config.epsilon:
                n_ctr += 1

            nxt = self._sample(final, temperature, do_sample)
            gen.append(nxt.item())
            if nxt.item() == eos:
                break

            emb = self._get_lm_embed()(nxt)
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.device, dtype=attn_mask.dtype)], dim=1)
            npos = torch.tensor([[cur]], device=self.device)
            cur += 1

            with torch.no_grad():
                c_out = lm(inputs_embeds=emb, attention_mask=attn_mask,
                            position_ids=npos, past_key_values=c_kv,
                            use_cache=True, return_dict=True)
                r_out = lm(inputs_embeds=emb, attention_mask=attn_mask,
                            position_ids=npos, past_key_values=r_kv,
                            use_cache=True, return_dict=True)

            c_kv, r_kv = c_out.past_key_values, r_out.past_key_values
            c_log = self._get_lm_head()(c_out.last_hidden_state)[:, -1, :]
            r_log = self._get_lm_head()(r_out.last_hidden_state)[:, -1, :]

            if verbose and step % 20 == 0:
                print(f"  step {step}: KL={kl:.4f} contrastive={kl >= self.config.epsilon}")

        text = self.processor.decode(gen, skip_special_tokens=True)
        return GenerationResult(
            text=text, tokens=gen, generation_time=0.0,
            num_steps=len(gen), kl_divergences=kls,
            contrastive_applied_count=n_ctr, total_steps=len(gen),
        )

    # ------------------------------------------------------------------
    # Single-pass generation (CVG-E Improvement B)
    # ------------------------------------------------------------------

    def _single_pass_generate(
        self, ch_emb, attn_mask, vis_mask,
        max_new_tokens, temperature, do_sample, verbose,
    ) -> GenerationResult:
        """One forward pass per step; approximate rejected via hook-based attention.

        Uses SingleLayerHook to capture attention weights from ONE layer only,
        while all other layers use SDPA/Flash Attention. This is what makes
        single-pass genuinely faster than dual-pass.
        """
        seq_len = ch_emb.shape[1]
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        lm = self._lm

        with torch.no_grad():
            c_out = lm(
                inputs_embeds=ch_emb, attention_mask=attn_mask,
                position_ids=pos, use_cache=True, return_dict=True,
            )

        c_kv = c_out.past_key_values
        c_log = self._get_lm_head()(c_out.last_hidden_state)[:, -1, :]
        r_log = self._approx_rejected_hooked(c_kv, vis_mask, c_out.last_hidden_state)

        gen, kls = [], []
        n_ctr = 0
        cur = seq_len
        eos = self.processor.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            kl = self.gff.kl_divergence(c_log, r_log)
            kls.append(kl)
            final = self.gff.adaptive_contrastive_decoding(c_log, r_log, kl) if kl >= self.config.epsilon else c_log
            if kl >= self.config.epsilon:
                n_ctr += 1

            nxt = self._sample(final, temperature, do_sample)
            gen.append(nxt.item())
            if nxt.item() == eos:
                break

            emb = self._get_lm_embed()(nxt)
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=self.device, dtype=attn_mask.dtype)], dim=1)
            npos = torch.tensor([[cur]], device=self.device)
            cur += 1

            with torch.no_grad():
                c_out = lm(
                    inputs_embeds=emb, attention_mask=attn_mask,
                    position_ids=npos, past_key_values=c_kv,
                    use_cache=True, return_dict=True,
                )

            c_kv = c_out.past_key_values
            c_log = self._get_lm_head()(c_out.last_hidden_state)[:, -1, :]
            r_log = self._approx_rejected_hooked(c_kv, vis_mask, c_out.last_hidden_state)

            if verbose and step % 20 == 0:
                print(f"  step {step}: KL={kl:.4f} contrastive={kl >= self.config.epsilon}")

        text = self.processor.decode(gen, skip_special_tokens=True)
        return GenerationResult(
            text=text, tokens=gen, generation_time=0.0,
            num_steps=len(gen), kl_divergences=kls,
            contrastive_applied_count=n_ctr, total_steps=len(gen),
        )

    def _approx_rejected_hooked(self, cache, vis_mask, last_hidden) -> torch.Tensor:
        """Approximate rejected logits using hook-captured attention.

        Uses SingleLayerHook data (attn_weights + hidden_input from ONE layer)
        instead of output_attentions=True on all layers. Falls back to
        chosen logits if hook didn't fire.
        """
        if self._hook is None or not self._hook.has_data:
            return self._get_lm_head()(last_hidden)[:, -1, :]

        attn_w, h_in = self._hook.get()
        li = self.last_approx_layer

        if hasattr(cache, 'layers') and hasattr(cache.layers[li], 'values'):
            values = cache.layers[li].values
        elif hasattr(cache, 'value_cache'):
            values = cache.value_cache[li]
        else:
            values = cache[li][1]

        kv_len = attn_w.shape[-1]
        if kv_len > vis_mask.shape[0]:
            pad = torch.zeros(kv_len - vis_mask.shape[0], dtype=torch.bool, device=vis_mask.device)
            vp = torch.cat([vis_mask, pad])
        else:
            vp = vis_mask[:kv_len]

        w = torch.ones(kv_len, device=attn_w.device, dtype=attn_w.dtype)
        w[vp] = self.approx.lambda_val
        w[~vp] = 1.0 / self.approx.lambda_val

        rw = attn_w * w.view(1, 1, 1, kv_len)
        rw = rw / (rw.sum(dim=-1, keepdim=True) + 1e-8)

        num_q_heads = rw.shape[1]
        num_kv_heads = values.shape[1]
        if num_kv_heads < num_q_heads:
            rep = num_q_heads // num_kv_heads
            values = values.repeat_interleave(rep, dim=1)

        ctx = torch.matmul(rw, values)
        B, H, T, D = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, T, H * D)

        layers = self._get_lm_layers()
        layer = layers[li]
        ctx = layer.self_attn.o_proj(ctx)

        if h_in.shape[1] != T:
            h_in = h_in[:, -T:, :]

        mid = h_in + ctx
        if hasattr(layer, "post_attention_layernorm"):
            mid_n = layer.post_attention_layernorm(mid)
        else:
            mid_n = mid
        out = mid + layer.mlp(mid_n)

        fn = self._get_final_norm()
        if fn is not None:
            out = fn(out)

        return self._get_lm_head()(out)[:, -1, :]

    def _approx_rejected(self, c_out, cache, vis_mask) -> torch.Tensor:
        """Approximate rejected logits by re-weighting attention at the last layer.

        Re-weights chosen branch attention (suppress visual, boost textual),
        re-computes one transformer layer's output, then runs final_norm + LM head.
        Skips all other layers — genuine single-layer approximation that avoids
        a full second forward pass.
        """
        li = self.last_approx_layer
        if not c_out.attentions or li >= len(c_out.attentions):
            return c_out.logits[:, -1, :]

        attn_w = c_out.attentions[li]  # (B, H, T, kv_len), T=1 during generation
        kv_len = attn_w.shape[-1]
        if hasattr(cache, 'layers') and hasattr(cache.layers[li], 'values'):
            values = cache.layers[li].values
        elif hasattr(cache, 'value_cache'):
            values = cache.value_cache[li]
        else:
            values = cache[li][1]

        if kv_len > vis_mask.shape[0]:
            pad = torch.zeros(kv_len - vis_mask.shape[0], dtype=torch.bool, device=vis_mask.device)
            vp = torch.cat([vis_mask, pad])
        else:
            vp = vis_mask[:kv_len]
        w = torch.ones(kv_len, device=attn_w.device, dtype=attn_w.dtype)
        w[vp] = self.approx.lambda_val
        w[~vp] = 1.0 / self.approx.lambda_val

        rw = attn_w * w.view(1, 1, 1, kv_len)
        rw = rw / (rw.sum(dim=-1, keepdim=True) + 1e-8)

        num_q_heads = rw.shape[1]
        num_kv_heads = values.shape[1]
        if num_kv_heads < num_q_heads:
            rep = num_q_heads // num_kv_heads
            values = values.repeat_interleave(rep, dim=1)

        ctx = torch.matmul(rw, values)  # (B, H, T, head_dim)
        B, H, T, D = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, T, H * D)

        layers = self._get_lm_layers()
        layer = layers[li]
        ctx = layer.self_attn.o_proj(ctx)

        h_in = c_out.hidden_states[li]
        if h_in.shape[1] != T:
            h_in = h_in[:, -T:, :]

        mid = h_in + ctx
        if hasattr(layer, "post_attention_layernorm"):
            mid_n = layer.post_attention_layernorm(mid)
        else:
            mid_n = mid
        out = mid + layer.mlp(mid_n)

        fn = self._get_final_norm()
        if fn is not None:
            out = fn(out)

        return self._get_lm_head()(out)[:, -1, :]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(logits, temperature, do_sample):
        if do_sample and temperature > 0:
            return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
        return logits.argmax(dim=-1, keepdim=True)

    def get_model_info(self) -> dict:
        return {
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
            "device": str(self.device),
            "num_lm_layers": self.num_lm_layers,
            "num_vision_layers": self.num_vision_layers,
            "num_visual_tokens": self.num_visual_tokens,
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "tau": self.config.tau,
                "epsilon": self.config.epsilon,
                "gamma": self.config.gamma,
                "eta": self.config.eta,
            },
            "is_cvg_e": isinstance(self.config, CVGEConfig),
        }


__all__ = ["CVGModel", "GenerationResult"]
