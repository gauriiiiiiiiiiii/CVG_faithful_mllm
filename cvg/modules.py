"""CVG Core Modules: VTD, DAC, GFF — PyTorch implementation.

Implements Complementary Visual Grounding from:
Fu et al., "Boosting Faithful Multi-Modal LLMs via Complementary Visual Grounding",
IEEE TIP, Vol. 34, 2025.

All three modules are training-free and operate at inference time only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class CVGConfig:
    """Configuration for CVG decoding."""

    # VTD: Visual Token Disentanglement
    alpha: float = 0.7
    cross_attn_layers: list[int] = field(default_factory=lambda: [19, 20, 21, 22])
    self_attn_layers: list[int] = field(default_factory=lambda: [16, 17, 18])

    # DAC: Dynamic Attention Calibration
    beta: float = 0.2
    tau: float = 10.0
    dac_layers: list[int] = field(default_factory=lambda: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])

    # GFF: Grounding Flow Fusion
    epsilon: float = 0.01
    gamma: float = 1.0
    eta: float = 0.2

    # Generation
    max_new_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = False


@dataclass
class CVGEConfig(CVGConfig):
    """Extended config for CVG-E improvements."""

    # Improvement A: Adaptive Token Selection
    use_adaptive_alpha: bool = False
    alpha_min: float = 0.5
    alpha_max: float = 0.85

    use_text_protection: bool = False
    text_neighbor_threshold: float = 0.6

    use_score_normalization: bool = False
    w_cross: float = 0.5
    w_self: float = 0.3
    w_text: float = 0.2

    # Improvement B: Single-Pass Approximation
    use_single_pass: bool = False
    approx_lambda: float = 0.1
    approx_layers: list[int] = field(default_factory=lambda: [28, 29, 30, 31, 32])


ABLATIONS = {
    "vanilla": CVGConfig(alpha=1.0, beta=0.0, epsilon=float("inf")),
    "cvg": CVGConfig(),
    "cvg_adaptive_alpha": CVGEConfig(use_adaptive_alpha=True),
    "cvg_text_protection": CVGEConfig(use_text_protection=True),
    "cvg_score_norm": CVGEConfig(use_score_normalization=True),
    "cvg_all_A": CVGEConfig(
        use_adaptive_alpha=True,
        use_text_protection=True,
        use_score_normalization=True,
    ),
    "cvg_single_pass": CVGEConfig(use_single_pass=True),
    "cvg_e": CVGEConfig(
        use_adaptive_alpha=True,
        use_text_protection=True,
        use_score_normalization=True,
        use_single_pass=True,
    ),
    # Small-model variants: softer contrastive decoding for <=2B param models
    "cvg_small": CVGConfig(epsilon=0.1, gamma=0.5, beta=0.1),
    "cvg_e_small": CVGEConfig(
        use_adaptive_alpha=True,
        use_text_protection=True,
        use_score_normalization=True,
        use_single_pass=True,
        epsilon=0.1,
        gamma=0.5,
        beta=0.1,
    ),
    # 7B model configs (LLaVA-1.5-7B: 32 LM layers, 24 ViT layers)
    "cvg_7b": CVGConfig(
        cross_attn_layers=[19, 20, 21, 22],
        self_attn_layers=[16, 17, 18],
        dac_layers=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    ),
    "cvg_e_7b": CVGEConfig(
        use_adaptive_alpha=True,
        use_text_protection=True,
        use_score_normalization=True,
        use_single_pass=True,
        cross_attn_layers=[19, 20, 21, 22],
        self_attn_layers=[16, 17, 18],
        dac_layers=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        approx_layers=[28, 29, 30, 31],
    ),
}


class VisualTokenDisentanglement:
    """Module 1: VTD — separates visual tokens into chosen/rejected branches.

    Computes hybrid significance score per visual token:
        a = w_cross * across + w_self * aself [+ w_text * text_score]

    where across is query-to-image attention from the last query token,
    and aself is [CLS]-to-visual-token self-attention from ViT intermediate layers.
    """

    def __init__(self, config: CVGConfig):
        self.config = config

    def extract_attention_from_hooks(
        self,
        cross_attn_maps: dict[int, torch.Tensor],
        self_attn_maps: dict[int, torch.Tensor],
        num_visual_tokens: int,
        num_query_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Aggregate attention maps into per-token significance scores.

        Args:
            cross_attn_maps: layer -> (batch, heads, M, N) cross-attention from query to image
            self_attn_maps: layer -> (batch, heads, N, N) self-attention within ViT
            num_visual_tokens: N
            num_query_tokens: M
            num_heads: number of attention heads

        Returns:
            significance scores of shape (batch, N)
        """
        batch_size = None
        if cross_attn_maps:
            batch_size = next(iter(cross_attn_maps.values())).shape[0]
        elif self_attn_maps:
            batch_size = next(iter(self_attn_maps.values())).shape[0]
        else:
            batch_size = 1

        # Cross-attention: last query token to all visual tokens (Eq. 6)
        cross_scores = []
        for layer in self.config.cross_attn_layers:
            if layer not in cross_attn_maps:
                continue
            attn = cross_attn_maps[layer]  # (B, H, M, N)
            # Last query token (index M-1) to all visual tokens
            last_query = attn[:, :, -1, :]  # (B, H, N)
            cross_scores.append(last_query.mean(dim=1))  # (B, N) average heads

        if cross_scores:
            across = torch.stack(cross_scores).mean(dim=0)  # (B, N)
        else:
            across = torch.zeros(batch_size, num_visual_tokens, device=self._device(cross_attn_maps, self_attn_maps))

        # Self-attention: [CLS] token to all visual tokens (Eq. 8)
        self_scores = []
        for layer in self.config.self_attn_layers:
            if layer not in self_attn_maps:
                continue
            attn = self_attn_maps[layer]  # (B, H, total_N, total_N)
            total_N = attn.shape[-1]
            # First token is CLS, remaining are visual tokens
            cls_to_visual = attn[:, :, 0, 1:total_N]  # (B, H, total_N-1)
            # Trim or pad to match num_visual_tokens
            if cls_to_visual.shape[-1] > num_visual_tokens:
                cls_to_visual = cls_to_visual[:, :, :num_visual_tokens]
            elif cls_to_visual.shape[-1] < num_visual_tokens:
                pad = num_visual_tokens - cls_to_visual.shape[-1]
                cls_to_visual = F.pad(cls_to_visual, (0, pad))
            self_scores.append(cls_to_visual.mean(dim=1))  # (B, N)

        if self_scores:
            aself = torch.stack(self_scores).mean(dim=0)  # (B, N)
        else:
            aself = torch.zeros_like(across)

        # Hybrid significance score (Eq. 9)
        if isinstance(self.config, CVGEConfig) and self.config.use_score_normalization:
            across = self._zscore(across)
            aself = self._zscore(aself)
            a = (
                self.config.w_cross * across
                + self.config.w_self * aself
            )
        else:
            a = across + aself  # original CVG: equal weighting

        return a  # (B, N)

    def compute_text_protection(
        self,
        self_attn_maps: dict[int, torch.Tensor],
        num_visual_tokens: int,
        grid_size: int,
    ) -> torch.Tensor:
        """Detect text-containing patches via self-attention neighbor concentration.

        Text patches show high local connectivity (neighboring text patches
        attend strongly to each other). We measure:
            text_score_i = sum_{j in N(i)} a_self_ij / sum_j a_self_ij

        where N(i) are spatially adjacent patches.

        Returns:
            text scores of shape (batch, N)
        """
        if not isinstance(self.config, CVGEConfig) or not self.config.use_text_protection:
            return None

        batch_size = next(iter(self_attn_maps.values())).shape[0]
        device = self._device({}, self_attn_maps)

        # Build neighbor mask for grid
        # Patches are flattened row-major: index = row * grid_size + col
        neighbor_mask = torch.zeros(num_visual_tokens, num_visual_tokens, device=device)
        for idx in range(num_visual_tokens):
            row, col = divmod(idx, grid_size)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    nidx = nr * grid_size + nc
                    neighbor_mask[idx, nidx] = 1.0

        # Aggregate self-attention across VTD layers
        text_scores_list = []
        for layer in self.config.self_attn_layers:
            if layer not in self_attn_maps:
                continue
            attn = self_attn_maps[layer]  # (B, H, total_N, total_N)
            total_N = attn.shape[-1]
            # Exclude CLS token (first row/col) to get visual-token-only attention
            visual_attn = attn[:, :, 1:total_N, 1:total_N]  # (B, H, N, N)
            # Trim or pad to match num_visual_tokens
            if visual_attn.shape[-1] > num_visual_tokens:
                visual_attn = visual_attn[:, :, :num_visual_tokens, :num_visual_tokens]
            elif visual_attn.shape[-1] < num_visual_tokens:
                pad = num_visual_tokens - visual_attn.shape[-1]
                visual_attn = F.pad(visual_attn, (0, pad, 0, pad))
            # Average over heads
            attn_avg = visual_attn.mean(dim=1)  # (B, N, N)
            # Compute neighbor concentration
            neighbor_attn = (attn_avg * neighbor_mask.unsqueeze(0)).sum(dim=-1)  # (B, N)
            total_attn = attn_avg.sum(dim=-1)  # (B, N)
            text_score = neighbor_attn / (total_attn + 1e-8)  # (B, N)
            text_scores_list.append(text_score)

        if text_scores_list:
            text_score = torch.stack(text_scores_list).mean(dim=0)  # (B, N)
        else:
            text_score = torch.zeros(batch_size, num_visual_tokens, device=device)

        return text_score

    def partition_tokens(
        self,
        significance: torch.Tensor,
        text_scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Partition visual tokens into chosen/rejected sets (Eq. 9-10).

        Args:
            significance: (B, N) hybrid significance scores
            text_scores: (B, N) optional text protection scores

        Returns:
            chosen_mask: (B, N) boolean, True for chosen tokens
            rejected_mask: (B, N) boolean, True for rejected tokens
            alpha_used: (B,) effective alpha per batch element
        """
        batch_size, N = significance.shape

        # Apply text protection boost if enabled
        if text_scores is not None:
            boosted = significance + self.config.w_text * text_scores
        else:
            boosted = significance

        # Determine alpha per batch element
        if isinstance(self.config, CVGEConfig) and self.config.use_adaptive_alpha:
            # Entropy-adaptive alpha (Improvement A1)
            alpha_used = self._compute_adaptive_alpha(boosted)
        else:
            alpha_used = torch.full((batch_size,), self.config.alpha, device=significance.device)

        # Partition
        chosen_mask = torch.zeros_like(boosted, dtype=torch.bool)
        rejected_mask = torch.zeros_like(boosted, dtype=torch.bool)

        for b in range(batch_size):
            alpha_b = alpha_used[b].item()
            n_chosen = max(1, int(math.ceil(alpha_b * N)))

            # Sort by significance descending
            _, sorted_idx = torch.sort(boosted[b], descending=True)
            chosen_mask[b, sorted_idx[:n_chosen]] = True
            rejected_mask[b, sorted_idx[n_chosen:]] = True

        return chosen_mask, rejected_mask, alpha_used

    def _compute_adaptive_alpha(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute entropy-adaptive alpha (Improvement A1).

        High entropy (uniform attention, complex scene) -> higher alpha toward alpha_max.
        Low entropy (peaked attention, clear scene) -> lower alpha toward alpha_min.

        Eq: alpha* = alpha_min + (alpha_max - alpha_min) * H(a) / log(N)
        """
        batch_size, N = scores.shape
        device = scores.device

        # Normalize scores to probability distribution
        probs = F.softmax(scores, dim=-1)  # (B, N)
        # Entropy: H = -sum p * log p
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (B,)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy  # (B,) in [0, 1]

        alpha_min = self.config.alpha_min
        alpha_max = self.config.alpha_max
        alpha_used = alpha_min + (alpha_max - alpha_min) * normalized_entropy

        return alpha_used  # (B,)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        """Z-score normalize along last dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        return (x - mean) / std

    @staticmethod
    def _device(cross_maps: dict, self_maps: dict) -> torch.device:
        if cross_maps:
            return next(iter(cross_maps.values())).device
        if self_maps:
            return next(iter(self_maps.values())).device
        return torch.device("cpu")


class DynamicAttentionCalibration:
    """Module 2: DAC — dynamically calibrate attention during generation.

    Calibration strength grows with generation step:
        delta_i = beta * (1 - exp(-i / tau))  (Eq. 11)

    Chosen branch: +delta for visual tokens, -delta for textual (Eq. 12)
    Rejected branch: -delta for visual tokens, +delta for textual (Eq. 13)
    """

    def __init__(self, config: CVGConfig):
        self.config = config

    def compute_delta(self, step: int) -> float:
        """Compute calibration strength at generation step i (Eq. 11)."""
        return self.config.beta * (1.0 - math.exp(-step / self.config.tau))

    def calibrate_attention_scores(
        self,
        attn_scores: torch.Tensor,
        token_type_mask: torch.Tensor,  # True=visual, False=textual, shape (seq_len,)
        step: int,
        branch: str = "chosen",
    ) -> torch.Tensor:
        """Apply dynamic calibration to attention logits (Eq. 12-13).

        Args:
            attn_scores: (B, H, seq_len, seq_len) attention logits before softmax
            token_type_mask: (seq_len,) boolean, True for visual positions
            step: current generation step i
            branch: "chosen" or "rejected"

        Returns:
            Calibrated attention logits (same shape as input)
        """
        delta = self.compute_delta(step)
        batch_size, num_heads, seq_len, _ = attn_scores.shape
        device = attn_scores.device

        # Build calibration matrix: (seq_len, seq_len)
        # For each target token j, apply +/- delta based on its type
        calib = torch.zeros(seq_len, seq_len, device=device)
        visual_mask = token_type_mask.unsqueeze(0).expand(seq_len, -1)  # (seq_len, seq_len)

        if branch == "chosen":
            # +delta for visual targets, -delta for textual targets
            calib[visual_mask] = delta
            calib[~visual_mask] = -delta
        else:  # rejected
            # -delta for visual targets, +delta for textual targets
            calib[visual_mask] = -delta
            calib[~visual_mask] = delta

        # Expand to full shape and add
        calib = calib.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        return attn_scores + calib


class GroundingFlowFusion:
    """Module 3: GFF — fuse chosen/rejected branch outputs via contrastive decoding.

    Implements plausibility filtering + adaptive contrastive decoding (Eq. 14-16).
    """

    def __init__(self, config: CVGConfig):
        self.config = config

    def plausibility_filter(
        self,
        chosen_logits: torch.Tensor,  # (vocab_size,) or (B, vocab_size)
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """Filter candidate vocabulary (Eq. 14).

        Keep tokens with p >= eta * max(p) from chosen branch.
        """
        if eta is None:
            eta = self.config.eta
        if chosen_logits.dim() == 1:
            chosen_logits = chosen_logits.unsqueeze(0)

        probs = F.softmax(chosen_logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = eta * max_prob
        mask = probs >= threshold  # (B, vocab_size)
        return mask

    def adaptive_contrastive_decoding(
        self,
        chosen_logits: torch.Tensor,  # (B, vocab_size)
        rejected_logits: torch.Tensor,  # (B, vocab_size)
        kl_divergence: float,
    ) -> torch.Tensor:
        """Apply adaptive contrastive decoding (Eq. 15-16).

        If KL divergence >= epsilon: logit_final = chosen + gamma * (chosen - rejected)
        If KL divergence < epsilon: logit_final = chosen
        """
        if kl_divergence >= self.config.epsilon:
            # Divergent branches — apply contrastive decoding
            final_logits = chosen_logits + self.config.gamma * (chosen_logits - rejected_logits)
        else:
            # Convergent branches — use chosen branch only
            final_logits = chosen_logits
        return final_logits

    @staticmethod
    def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute KL divergence between two probability distributions."""
        p = F.softmax(p.float(), dim=-1)
        q = F.softmax(q.float(), dim=-1)
        kl = (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=-1)
        return kl.mean().item()


# ============================================================================
# CVG-E Improvement A: Adaptive Token Selection (already integrated above)
# Improvement B: Single-Pass Approximation (see approximation.py)
# ============================================================================

__all__ = [
    "CVGConfig",
    "CVGEConfig",
    "ABLATIONS",
    "VisualTokenDisentanglement",
    "DynamicAttentionCalibration",
    "GroundingFlowFusion",
]
