"""Generation with the masked Transformer: iterative confidence-based decoding.

1. Velocities: start with every free cell masked; each iteration samples all
   masked cells (temperature + top-p, rules applied as logit masks), commits
   the most confident ones and re-masks the rest (cosine schedule).
2. Microtiming and articulation: same procedure over the cells that ended up
   as hits, now with every velocity visible.

Rules never overwrite the model: fixed cells (e.g. the chico cadence) are part
of the input and `allowed` removes forbidden classes before sampling.
Pieces longer than the model window are generated in overlapping windows.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from candombe_rules import allowed_velocity_mask, empty_fixed, lock_chico
from model import OFF_MASK, VEL_MASK, CandombeTransformer, art_mask, load_checkpoint
from representation import (
    DENSITY_UNKNOWN,
    N_ART,
    N_INSTR,
    N_OFF,
    N_VEL,
    SECTION_UNKNOWN,
    STEPS_PER_BAR,
    STYLE_UNKNOWN,
    TEMPO_UNKNOWN,
    VEL_CENTERS,
    Piece,
    default_structure,
    density_bucket,
    density_control_to_rate,
    make_piece,
    section_id,
    style_id,
    tempo_bucket,
)

NEG_INF = -1e9


def load_model(path: str, device: str = "cpu") -> CandombeTransformer:
    return load_checkpoint(path, device=device)


def _sample_logits(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator):
    """Sample the last dim of `logits`; returns (choice, log-prob of the choice)."""
    scaled = logits / max(1e-4, temperature)
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
        keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
    flat = probs.reshape(-1, probs.shape[-1])
    choice = torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])
    logp = torch.log(probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
    return choice, logp


def _commit_schedule(total: int, iteration: int, iterations: int) -> int:
    """How many cells stay masked after `iteration` (cosine schedule)."""
    return int(math.floor(total * math.cos(math.pi / 2 * (iteration + 1) / iterations)))


class _Window:
    """Tensors for one generation window."""

    def __init__(self, model, steps, sections, tempo, density_rate, style, device):
        self.model = model
        self.device = device
        self.section = torch.tensor([[section_id(name) for name in sections for _ in range(STEPS_PER_BAR)][:steps]], device=device)
        self.tempo = torch.tensor([tempo_bucket(tempo)], device=device)
        self.density = torch.tensor([density_bucket(density_rate)], device=device)
        self.style = torch.tensor([style_id(style)], device=device)
        self.uncond = (
            torch.full_like(self.section, SECTION_UNKNOWN),
            torch.tensor([TEMPO_UNKNOWN], device=device),
            torch.tensor([DENSITY_UNKNOWN], device=device),
            torch.tensor([STYLE_UNKNOWN], device=device),
        )

    def logits(self, vel, off, art, guidance: float):
        with torch.no_grad():
            out = self.model(vel, off, art, self.section, self.tempo, self.density, self.style)
            if guidance == 1.0:
                return out
            # Classifier-free guidance: push away from the unconditioned prediction.
            base = self.model(vel, off, art, *self.uncond)
            mix = lambda c, u: u + guidance * (c - u)  # noqa: E731
            return {
                "vel": mix(out["vel"], base["vel"]),
                "off": mix(out["off"], base["off"]),
                "art": [mix(c, u) for c, u in zip(out["art"], base["art"])],
            }


def _generate_window(
    model: CandombeTransformer,
    fixed: Dict[str, np.ndarray],
    allowed: np.ndarray,
    sections: Sequence[str],
    tempo: float,
    density_rate: float,
    style: Optional[str],
    temperature: float,
    top_p: float,
    iterations: int,
    silence_bias: float,
    guidance: float,
    timing_temperature: float,
    generator: torch.Generator,
    device: str,
) -> Dict[str, np.ndarray]:
    steps = fixed["vel"].shape[0]
    win = _Window(model, steps, sections, tempo, density_rate, style, device)

    fixed_vel = torch.from_numpy(fixed["vel"]).to(device)
    vel = torch.where(fixed_vel >= 0, fixed_vel, torch.full_like(fixed_vel, VEL_MASK)).unsqueeze(0)
    off = torch.full((1, steps, N_INSTR), OFF_MASK, dtype=torch.long, device=device)
    art = torch.stack([torch.full((1, steps), art_mask(i), dtype=torch.long, device=device) for i in range(N_INSTR)], dim=-1)
    allowed_t = torch.from_numpy(allowed).to(device)

    # --- phase 1: velocities --------------------------------------------------------
    free = (vel[0] == VEL_MASK)
    total_free = int(free.sum())
    for iteration in range(iterations):
        masked = vel[0] == VEL_MASK
        if not masked.any():
            break
        logits = win.logits(vel, off, art, guidance)["vel"][0].clone()
        logits[..., 0] += silence_bias
        logits = logits.masked_fill(~allowed_t, NEG_INF)
        choice, logp = _sample_logits(logits, temperature, top_p, generator)
        # Annealed noise keeps early commits diverse (MaskGIT); none on the last pass.
        anneal = temperature * (1.0 - (iteration + 1) / iterations)
        gumbel = -torch.log(-torch.log(torch.rand(logp.shape, generator=generator, device=device).clamp(1e-9, 1 - 1e-9)))
        confidence = torch.where(masked, logp + anneal * gumbel, torch.full_like(logp, float("inf")))

        keep_masked = _commit_schedule(total_free, iteration, iterations)
        new_vel = torch.where(masked, choice, vel[0])
        if keep_masked > 0:
            remask = torch.zeros_like(masked).flatten()
            remask[torch.argsort(confidence.flatten())[:keep_masked]] = True
            new_vel = torch.where(remask.view_as(masked), torch.full_like(new_vel, VEL_MASK), new_vel)
        vel = new_vel.unsqueeze(0)

    # --- phase 2: microtiming and articulation on hits ---------------------------------
    hits = vel[0] > 0
    fixed_off = torch.from_numpy(fixed["off"]).to(device)
    fixed_art = torch.from_numpy(fixed["art"]).to(device)
    off = torch.where(hits & (fixed_off >= 0), fixed_off, off[0]).unsqueeze(0)
    art = torch.where(hits & (fixed_art >= 0), fixed_art, art[0]).unsqueeze(0)
    pending = hits & (fixed_off < 0)
    pending_art = hits & (fixed_art < 0)
    total_pending = int((pending | pending_art).sum())
    timing_iterations = max(2, iterations // 4)
    for iteration in range(timing_iterations):
        if not (pending.any() or pending_art.any()):
            break
        out = win.logits(vel, off, art, guidance)
        off_choice, off_logp = _sample_logits(out["off"][0], timing_temperature, 1.0, generator)
        art_choice = torch.zeros_like(off_choice)
        art_logp = torch.zeros_like(off_logp)
        for inst in range(N_INSTR):
            choice, logp = _sample_logits(out["art"][inst][0], temperature, top_p, generator)
            art_choice[:, inst] = choice
            art_logp[:, inst] = logp

        active = pending | pending_art
        confidence = torch.where(active, off_logp + art_logp, torch.full_like(off_logp, float("inf")))
        keep = _commit_schedule(total_pending, iteration, timing_iterations)
        commit = active.clone().flatten()
        if keep > 0:
            commit[torch.argsort(confidence.flatten())[:keep]] = False
        commit = commit.view_as(active)

        off = torch.where(commit & pending, off_choice, off[0]).unsqueeze(0)
        art = torch.where(commit & pending_art, art_choice, art[0]).unsqueeze(0)
        pending = pending & ~commit
        pending_art = pending_art & ~commit

    vel_np = vel[0].cpu().numpy()
    off_np = np.where(vel_np > 0, off[0].cpu().numpy(), -1)
    art_np = np.where(vel_np > 0, art[0].cpu().numpy(), -1)
    # Masks/unknown ids never leak into the output.
    off_np = np.where(off_np >= N_OFF, -1, off_np)
    for inst in range(N_INSTR):
        art_np[:, inst] = np.where(art_np[:, inst] >= N_ART[inst], -1, art_np[:, inst])
    return {"vel": vel_np, "off": off_np, "art": art_np}


def generate_piece(
    model: CandombeTransformer,
    bars: int,
    tempo: float = 96.0,
    density: float = 0.55,
    structure: Optional[Sequence[str]] = None,
    style: Optional[str] = None,
    fixed: Optional[Dict[str, np.ndarray]] = None,
    chico_locked: bool = True,
    strict_rules: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.92,
    iterations: int = 16,
    guidance: float = 1.0,
    timing_temperature: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Piece:
    """Generate `bars` bars. `density` is the 0..1 UI knob (see representation).

    `timing_temperature` is kept below the note temperature on purpose: sampling
    microtiming hot turns the systematic swing into random sloppiness."""
    steps = bars * STEPS_PER_BAR
    structure = list(structure) if structure else default_structure(bars)
    if len(structure) != bars:
        raise ValueError("structure must have one section per bar")

    fixed = fixed if fixed is not None else empty_fixed(steps)
    if chico_locked:
        fixed = lock_chico(fixed)
    allowed = allowed_velocity_mask(steps, structure) if strict_rules else np.ones((steps, N_INSTR, N_VEL), dtype=bool)
    # A cell fixed by the caller is always allowed to keep its value.
    for step, inst in zip(*np.nonzero(fixed["vel"] >= 0)):
        allowed[step, inst, fixed["vel"][step, inst]] = True

    target_rate = density_control_to_rate(density)
    silence_bias = 1.5 * (0.5 - float(np.clip(density, 0.0, 1.0)))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31)))

    window_bars = model.config.max_bars
    hop_bars = max(1, window_bars // 2)
    result = {key: value.copy() for key, value in fixed.items()}
    start_bar = 0
    while True:
        end_bar = min(bars, start_bar + window_bars)
        lo, hi = start_bar * STEPS_PER_BAR, end_bar * STEPS_PER_BAR
        window_fixed = {key: result[key][lo:hi].copy() for key in result}
        window = _generate_window(
            model,
            window_fixed,
            allowed[lo:hi],
            structure[start_bar:end_bar],
            tempo,
            target_rate,
            style,
            temperature,
            top_p,
            iterations,
            silence_bias,
            guidance,
            timing_temperature,
            generator,
            device,
        )
        for key in result:
            result[key][lo:hi] = window[key]
        if end_bar >= bars:
            break
        # Next window starts inside this one: the overlap is fixed context.
        start_bar += hop_bars

    velocities = np.zeros((steps, N_INSTR), dtype=np.int32)
    for value, center in enumerate(VEL_CENTERS, start=1):
        velocities[result["vel"] == value] = center
    piece = make_piece(velocities, None, result["art"], structure, tempo=tempo, style=style, source="model")
    piece.off = np.where(result["vel"] > 0, result["off"], -1)
    return piece
