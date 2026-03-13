"""Hooks for capturing activations and attention patterns."""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AttentionCapture:
    """
    Context manager to capture pre-softmax attention logits.

    Usage:
        with AttentionCapture(model) as cap:
            model(inputs)
        logits = cap.logits  # list of (batch, heads, seq, seq)
    """
    model: nn.Module
    logits: list = field(default_factory=list)
    handles: list = field(default_factory=list)

    def __enter__(self):
        self.logits = []
        self.handles = []

        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                handle = module.register_forward_hook(self._makeHook(name))
                self.handles.append(handle)

        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _makeHook(self, name: str):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2:
                if out[1] is not None:
                    self.logits.append(out[1].detach())
        return hook


@dataclass
class ActivationCapture:
    """
    Capture activations at specified layers.

    Usage:
        with ActivationCapture(model, ["mlp", "ln"]) as cap:
            model(inputs)
        acts = cap.activations  # dict[layerName] -> tensor
    """
    model: nn.Module
    patterns: list[str]
    activations: dict = field(default_factory=dict)
    handles: list = field(default_factory=list)

    def __enter__(self):
        self.activations = {}
        self.handles = []

        for name, module in self.model.named_modules():
            for pat in self.patterns:
                if pat in name.lower():
                    handle = module.register_forward_hook(self._makeHook(name))
                    self.handles.append(handle)
                    break

        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _makeHook(self, name: str):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.activations[name] = out.detach()
            elif isinstance(out, tuple) and len(out) > 0:
                self.activations[name] = out[0].detach()
        return hook


def computeLayerNorms(activations: dict) -> dict[str, float]:
    """Compute infinity norm for each captured activation."""
    return {
        name: act.abs().max().item()
        for name, act in activations.items()
    }
