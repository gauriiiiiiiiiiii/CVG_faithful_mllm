"""CVG-E Improvement B: Single-Pass Branch Approximation."""

from __future__ import annotations

from cvg.modules import CVGEConfig


class SinglePassBranchApproximation:
    """Re-weight cached attention at a single layer to approximate rejected branch logits."""

    def __init__(self, config: CVGEConfig):
        self.config = config
        self.lambda_val = config.approx_lambda
