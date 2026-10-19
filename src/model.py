"""Masked (bidirectional) Transformer over the candombe grid.

One token per 16th step. Every (step, instrument) cell has velocity, offset and
articulation inputs that can each be replaced by a MASK id; the model predicts
the masked values from everything that is visible. Training only scores masked
cells, so the target is never part of the input (the old decoder leaked the
next step through unmasked attention).

The same network handles generation from scratch, infilling a track over fixed
ones (e.g. repique over chico + piano) and humanizing timing/articulation.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from representation import (
    N_ART,
    N_DENSITY,
    N_INSTR,
    N_OFF,
    N_TEMPO,
    N_VEL,
    SECTIONS,
    STEPS_PER_BAR,
    STYLES,
)

VEL_MASK = N_VEL
OFF_UNKNOWN = N_OFF
OFF_MASK = N_OFF + 1


def art_unknown(inst: int) -> int:
    return N_ART[inst]


def art_mask(inst: int) -> int:
    return N_ART[inst] + 1


@dataclass
class ModelConfig:
    d_model: int = 160
    nhead: int = 4
    num_layers: int = 5
    dim_feedforward: int = 640
    dropout: float = 0.1
    max_bars: int = 16


class CandombeTransformer(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.d_model

        self.vel_emb = nn.ModuleList([nn.Embedding(N_VEL + 1, d) for _ in range(N_INSTR)])
        self.off_emb = nn.ModuleList([nn.Embedding(N_OFF + 2, d) for _ in range(N_INSTR)])
        self.art_emb = nn.ModuleList([nn.Embedding(N_ART[i] + 2, d) for i in range(N_INSTR)])

        self.step_emb = nn.Embedding(STEPS_PER_BAR, d)
        self.bar_emb = nn.Embedding(self.config.max_bars, d)
        self.section_emb = nn.Embedding(len(SECTIONS) + 1, d)
        self.tempo_emb = nn.Embedding(N_TEMPO + 1, d)
        self.density_emb = nn.Embedding(N_DENSITY + 1, d)
        self.style_emb = nn.Embedding(len(STYLES) + 1, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.vel_head = nn.Linear(d, N_INSTR * N_VEL)
        self.off_head = nn.Linear(d, N_INSTR * N_OFF)
        self.art_heads = nn.ModuleList([nn.Linear(d, N_ART[i]) for i in range(N_INSTR)])

    def forward(
        self,
        vel: torch.Tensor,  # (B, S, 3) long, VEL_MASK where hidden
        off: torch.Tensor,  # (B, S, 3) long, OFF_UNKNOWN / OFF_MASK allowed
        art: torch.Tensor,  # (B, S, 3) long, art_unknown / art_mask allowed
        section: torch.Tensor,  # (B, S) long
        tempo: torch.Tensor,  # (B,) long
        density: torch.Tensor,  # (B,) long
        style: torch.Tensor,  # (B,) long
        pad_mask: Optional[torch.Tensor] = None,  # (B, S) bool, True = padding
    ) -> Dict[str, object]:
        batch, steps, _ = vel.shape
        positions = torch.arange(steps, device=vel.device)

        x = self.step_emb(positions % STEPS_PER_BAR) + self.bar_emb(positions // STEPS_PER_BAR)
        x = x.unsqueeze(0).expand(batch, -1, -1)
        x = x + self.section_emb(section)
        x = x + (self.tempo_emb(tempo) + self.density_emb(density) + self.style_emb(style)).unsqueeze(1)
        for inst in range(N_INSTR):
            x = x + self.vel_emb[inst](vel[..., inst]) + self.off_emb[inst](off[..., inst]) + self.art_emb[inst](art[..., inst])

        h = self.norm(self.encoder(x, src_key_padding_mask=pad_mask))
        return {
            "vel": self.vel_head(h).view(batch, steps, N_INSTR, N_VEL),
            "off": self.off_head(h).view(batch, steps, N_INSTR, N_OFF),
            "art": [head(h) for head in self.art_heads],
        }


def save_checkpoint(path: str, model: CandombeTransformer, extra: Optional[dict] = None) -> None:
    torch.save({"format": "candombe-v2", "config": asdict(model.config), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str, device: str = "cpu") -> CandombeTransformer:
    """Load a v2 checkpoint. Legacy (v1) state dicts raise ValueError: that model
    was trained with target leakage and is not compatible with this architecture."""
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != "candombe-v2":
        raise ValueError(f"{path} is not a candombe-v2 checkpoint (legacy v1 models are not supported)")
    model = CandombeTransformer(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
