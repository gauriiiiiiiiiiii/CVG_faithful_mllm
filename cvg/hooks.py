"""Hook-based attention extraction for CVG-E single-pass approximation.

Registers a forward hook on a single transformer layer to capture attention
weights and hidden states, WITHOUT forcing eager attention globally.
This allows SDPA/Flash Attention on all other layers, eliminating the
O(L*n^2) overhead that made single-pass slower than dual-pass.

Usage:
    hook = AttentionHook(model, layer_idx=21)
    hook.install()
    output = model(inputs)  # only layer 21 uses eager attention
    attn_weights, hidden_in = hook.get()
    hook.remove()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


class AttentionHook:
    """Capture attention weights from a single decoder layer via forward hook.

    Instead of output_attentions=True (which materializes attention at ALL
    layers), this hooks into one layer's self-attention module and temporarily
    forces eager attention only there. All other layers use SDPA/Flash.
    """

    def __init__(self, language_model: nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self._lm = language_model
        self._handle_attn = None
        self._handle_layer = None
        self._attn_weights: Optional[torch.Tensor] = None
        self._hidden_in: Optional[torch.Tensor] = None
        self._layer = self._get_layer(layer_idx)
        self._attn_module = self._get_attn_module(self._layer)

    def _get_layer(self, idx: int) -> nn.Module:
        base = self._lm.model if hasattr(self._lm, "model") else self._lm
        return base.layers[idx]

    def _get_attn_module(self, layer: nn.Module) -> nn.Module:
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        raise AttributeError(f"Cannot find self_attn in layer {self.layer_idx}")

    def install(self):
        """Register hooks to capture attention weights and layer input."""
        self._attn_weights = None
        self._hidden_in = None

        def capture_layer_input(module, args, kwargs):
            if args:
                self._hidden_in = args[0].detach()
            elif "hidden_states" in kwargs:
                self._hidden_in = kwargs["hidden_states"].detach()

        def capture_attn_output(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                if output[1] is not None:
                    self._attn_weights = output[1].detach()

        self._handle_layer = self._layer.register_forward_pre_hook(
            capture_layer_input, with_kwargs=True
        )
        self._handle_attn = self._attn_module.register_forward_hook(
            capture_attn_output, with_kwargs=True
        )

    def remove(self):
        """Remove all hooks."""
        if self._handle_attn is not None:
            self._handle_attn.remove()
            self._handle_attn = None
        if self._handle_layer is not None:
            self._handle_layer.remove()
            self._handle_layer = None

    def get(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return captured (attn_weights, hidden_input) and clear buffer."""
        aw, hi = self._attn_weights, self._hidden_in
        self._attn_weights = None
        self._hidden_in = None
        return aw, hi

    @property
    def has_data(self) -> bool:
        return self._attn_weights is not None


class EagerAttentionContext:
    """Context manager that forces eager attention on a single layer.

    HuggingFace transformers check `config._attn_implementation` or the
    module's `_attn_implementation` attribute. We temporarily override it
    for one layer's attention module.
    """

    def __init__(self, attn_module: nn.Module):
        self._module = attn_module
        self._original = getattr(attn_module, "_attn_implementation", None)

    def __enter__(self):
        self._module._attn_implementation = "eager"
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            self._module._attn_implementation = self._original
        elif hasattr(self._module, "_attn_implementation"):
            delattr(self._module, "_attn_implementation")


class SingleLayerHook:
    """Combined hook + eager override for CVG-E single-pass.

    Installs on a single layer:
    1. Forces eager attention (so we get weights) on ONLY that layer
    2. Captures attention weights + pre-layer hidden state
    3. All other layers keep using SDPA/Flash Attention

    This is the key to making single-pass actually fast.
    """

    def __init__(self, language_model: nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self._lm = language_model
        self._hook = AttentionHook(language_model, layer_idx)
        self._eager_ctx: Optional[EagerAttentionContext] = None

    def install(self):
        """Install hooks and force eager attention on target layer."""
        self._hook.install()
        self._eager_ctx = EagerAttentionContext(self._hook._attn_module)
        self._eager_ctx.__enter__()

    def remove(self):
        """Remove hooks and restore original attention implementation."""
        self._hook.remove()
        if self._eager_ctx is not None:
            self._eager_ctx.__exit__(None, None, None)
            self._eager_ctx = None

    def get(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (attn_weights, hidden_input) from last forward pass."""
        return self._hook.get()

    @property
    def has_data(self) -> bool:
        return self._hook.has_data

    @contextmanager
    @staticmethod
    def scoped(language_model: nn.Module, layer_idx: int):
        """Context manager for temporary hook installation."""
        hook = SingleLayerHook(language_model, layer_idx)
        hook.install()
        try:
            yield hook
        finally:
            hook.remove()


__all__ = ["AttentionHook", "EagerAttentionContext", "SingleLayerHook"]
