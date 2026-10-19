import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tiny_model():
    import torch

    from model import CandombeTransformer, ModelConfig

    torch.manual_seed(0)
    return CandombeTransformer(ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0, max_bars=4)).eval()
