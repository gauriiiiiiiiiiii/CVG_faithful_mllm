"""CVG-E: Efficient Adaptive Complementary Visual Grounding for MLLMs."""

from cvg.modules import (
    CVGConfig,
    CVGEConfig,
    ABLATIONS,
    VisualTokenDisentanglement,
    DynamicAttentionCalibration,
    GroundingFlowFusion,
)
from cvg.hooks import AttentionHook, SingleLayerHook
from cvg.model import CVGModel, GenerationResult

__all__ = [
    "CVGConfig",
    "CVGEConfig",
    "ABLATIONS",
    "VisualTokenDisentanglement",
    "DynamicAttentionCalibration",
    "GroundingFlowFusion",
    "AttentionHook",
    "SingleLayerHook",
    "CVGModel",
    "GenerationResult",
]
