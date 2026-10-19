"""Train the masked candombe Transformer.

    python src/train.py --data-glob "data/processed/synthetic/*.npz@1,data/processed/candombe_dataset/*.npz@4"

`pattern@weight` oversamples a source (use it to give real recordings more
weight than synthetic pieces). The best checkpoint by validation loss is saved
as `<save-prefix>_best.pt` together with its config.
"""

import argparse
import glob
import json
import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import OFF_MASK, OFF_UNKNOWN, VEL_MASK, CandombeTransformer, ModelConfig, art_mask, art_unknown, save_checkpoint
from representation import (
    DENSITY_UNKNOWN,
    N_INSTR,
    SECTION_UNKNOWN,
    STEPS_PER_BAR,
    STYLE_UNKNOWN,
    TEMPO_UNKNOWN,
    Piece,
    density_bucket,
    hit_rate,
    load_piece,
    style_id,
    tempo_bucket,
)

IGNORE = -100
OFF_LOSS_WEIGHT = 0.5
ART_LOSS_WEIGHT = 0.5
OFF_TARGET_SIGMA = 1.0  # in bins
CONDITION_DROPOUT = {"section": 0.15, "tempo": 0.2, "density": 0.2, "style": 0.2}


def sample_masks(rng: np.random.Generator, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (velocity mask, timing/articulation mask), both (S, 3) bool.

    Strategies mirror the generation use cases: from scratch (all / random),
    infilling an instrument over the others, continuing a span of bars, and
    timing-only (humanize given hits).
    """
    steps = vel.shape[0]
    bars = max(1, steps // STEPS_PER_BAR)
    m_vel = np.zeros((steps, N_INSTR), dtype=bool)
    roll = rng.random()
    timing_only = False
    if roll < 0.35:
        rate = math.cos(math.pi / 2 * rng.random())
        m_vel = rng.random((steps, N_INSTR)) < rate
    elif roll < 0.60:
        insts = rng.choice(N_INSTR, size=int(rng.integers(1, 3)), replace=False)
        m_vel[:, insts] = True
    elif roll < 0.75:
        span = int(rng.integers(1, max(2, bars // 2 + 1)))
        start = int(rng.integers(0, bars - span + 1))
        m_vel[start * STEPS_PER_BAR : (start + span) * STEPS_PER_BAR] = True
    elif roll < 0.85:
        m_vel[:] = True
    else:
        timing_only = True

    if not timing_only and not m_vel.any():
        m_vel[int(rng.integers(0, steps)), int(rng.integers(0, N_INSTR))] = True

    m_time = m_vel.copy()
    if timing_only:
        m_time[:] = True
    elif rng.random() < 0.3:
        m_time |= rng.random((steps, N_INSTR)) < 0.5
    return m_vel, m_time


def encode_example(piece: Piece, start: int, steps: int, rng: np.random.Generator, drop_conditions: bool = True) -> Dict[str, np.ndarray]:
    vel = piece.vel[start : start + steps]
    off = piece.off[start : start + steps]
    art = piece.art[start : start + steps]
    m_vel, m_time = sample_masks(rng, vel)

    vel_in = np.where(m_vel, VEL_MASK, vel)
    off_in = np.where(m_time, OFF_MASK, np.where(off >= 0, off, OFF_UNKNOWN))
    art_in = np.empty_like(art)
    for inst in range(N_INSTR):
        art_in[:, inst] = np.where(m_time[:, inst], art_mask(inst), np.where(art[:, inst] >= 0, art[:, inst], art_unknown(inst)))

    def dropped(name: str) -> bool:
        return drop_conditions and rng.random() < CONDITION_DROPOUT[name]

    section = piece.section[start : start + steps]
    return {
        "vel_in": vel_in,
        "off_in": off_in,
        "art_in": art_in,
        "section": np.full(steps, SECTION_UNKNOWN) if dropped("section") else section,
        "tempo": np.int64(TEMPO_UNKNOWN if dropped("tempo") else tempo_bucket(piece.tempo)),
        "density": np.int64(DENSITY_UNKNOWN if dropped("density") else density_bucket(hit_rate(vel))),
        "style": np.int64(STYLE_UNKNOWN if dropped("style") else style_id(piece.style)),
        "vel_t": np.where(m_vel, vel, IGNORE),
        "off_t": np.where(m_time & (off >= 0), off, IGNORE),
        "art_t": np.where(m_time & (art >= 0), art, IGNORE),
    }


class MaskedWindowDataset(Dataset):
    """Random bar-aligned windows of each piece, masked on the fly.

    With `deterministic=True` (validation) window and mask depend only on the
    index, so the validation loss is comparable across epochs.
    """

    def __init__(self, pieces: Sequence[Piece], max_bars: int, repeats: int = 1, deterministic: bool = False, seed: int = 0):
        self.pieces = [p for p in pieces if p.bars >= 1]
        self.max_bars = max_bars
        self.repeats = repeats
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pieces) * self.repeats

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed + index) if self.deterministic else np.random.default_rng()
        piece = self.pieces[index % len(self.pieces)]
        max_window = min(self.max_bars, piece.bars)
        window_bars = int(rng.integers(min(4, max_window), max_window + 1))
        start_bar = int(rng.integers(0, piece.bars - window_bars + 1))
        return encode_example(piece, start_bar * STEPS_PER_BAR, window_bars * STEPS_PER_BAR, rng, drop_conditions=not self.deterministic)


def collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    max_steps = max(item["vel_in"].shape[0] for item in batch)
    size = len(batch)

    def padded(key: str, fill: int) -> torch.Tensor:
        shape = (size, max_steps) + batch[0][key].shape[1:]
        out = np.full(shape, fill, dtype=np.int64)
        for i, item in enumerate(batch):
            out[i, : item[key].shape[0]] = item[key]
        return torch.from_numpy(out)

    pad_mask = torch.ones(size, max_steps, dtype=torch.bool)
    for i, item in enumerate(batch):
        pad_mask[i, : item["vel_in"].shape[0]] = False

    # Padded positions get valid ids for embedding lookup; their targets are IGNORE.
    art_pad = [art_unknown(inst) for inst in range(N_INSTR)]
    art_in = padded("art_in", 0)
    for inst in range(N_INSTR):
        art_in[..., inst][pad_mask] = art_pad[inst]
    return {
        "vel": padded("vel_in", VEL_MASK),
        "off": padded("off_in", OFF_UNKNOWN),
        "art": art_in,
        "section": padded("section", SECTION_UNKNOWN),
        "tempo": torch.tensor([int(item["tempo"]) for item in batch]),
        "density": torch.tensor([int(item["density"]) for item in batch]),
        "style": torch.tensor([int(item["style"]) for item in batch]),
        "pad_mask": pad_mask,
        "vel_t": padded("vel_t", IGNORE),
        "off_t": padded("off_t", IGNORE),
        "art_t": padded("art_t", IGNORE),
    }


def compute_loss(model: CandombeTransformer, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model(
        batch["vel"], batch["off"], batch["art"], batch["section"], batch["tempo"], batch["density"], batch["style"], batch["pad_mask"]
    )
    vel_logits = out["vel"]
    vel_loss = F.cross_entropy(vel_logits.reshape(-1, vel_logits.shape[-1]), batch["vel_t"].reshape(-1), ignore_index=IGNORE)
    loss = vel_loss
    stats = {"vel": float(vel_loss)}

    off_t = batch["off_t"].reshape(-1)
    known = off_t != IGNORE
    if known.any():
        # Offset bins are ordered: a near miss should cost less than a far one,
        # so the target is a small Gaussian around the true bin.
        logits = out["off"].reshape(-1, out["off"].shape[-1])[known]
        bins = torch.arange(logits.shape[-1], device=logits.device)
        soft = torch.exp(-0.5 * ((bins[None, :] - off_t[known][:, None]) / OFF_TARGET_SIGMA) ** 2)
        soft = soft / soft.sum(-1, keepdim=True)
        off_loss = -(soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss = loss + OFF_LOSS_WEIGHT * off_loss
        stats["off"] = float(off_loss)

    art_losses = []
    for inst in range(N_INSTR):
        target = batch["art_t"][..., inst].reshape(-1)
        if (target != IGNORE).any():
            logits = out["art"][inst]
            art_losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target, ignore_index=IGNORE))
    if art_losses:
        art_loss = torch.stack(art_losses).mean()
        loss = loss + ART_LOSS_WEIGHT * art_loss
        stats["art"] = float(art_loss)

    valid = batch["vel_t"] != IGNORE
    if valid.any():
        stats["vel_acc"] = float((vel_logits.argmax(-1)[valid] == batch["vel_t"][valid]).float().mean())
    return loss, stats


def resolve_sources(data_glob: str) -> List[Tuple[str, float]]:
    """`pattern@weight,...` -> sorted (path, weight); a later pattern overrides an earlier weight."""
    weights: Dict[str, float] = {}
    for entry in (part.strip() for part in str(data_glob).split(",")):
        if not entry:
            continue
        pattern, _, weight = entry.partition("@")
        for path in glob.glob(pattern):
            weights[path] = float(weight) if weight else 1.0
    return sorted(weights.items())


def split_sources(sources: List[Tuple[str, float]], val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sources))
    n_val = int(round(len(sources) * val_fraction)) if len(sources) > 1 else 0
    val_idx = set(order[:n_val].tolist())
    train = [sources[i] for i in range(len(sources)) if i not in val_idx]
    val = [sources[i] for i in range(len(sources)) if i in val_idx]
    return train, val


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def evaluate(model: CandombeTransformer, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, stats = compute_loss(model, batch)
            stats["loss"] = float(loss)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_loop(
    data_glob: str = "data/processed/synthetic/*.npz",
    epochs: int = 30,
    lr: float = 5e-4,
    batch_size: int = 32,
    repeats: int = 4,
    val_fraction: float = 0.1,
    device: str = "cpu",
    save_prefix: str = "models/candombe_v2",
    config: ModelConfig = None,
    seed: int = 0,
    patience: int = 8,
) -> dict:
    torch.manual_seed(seed)
    sources = resolve_sources(data_glob)
    if not sources:
        raise FileNotFoundError(f"No training files found for pattern: {data_glob}")
    train_sources, val_sources = split_sources(sources, val_fraction, seed)
    config = config or ModelConfig()

    train_pieces = [load_piece(path) for path, _ in train_sources]
    train_ds = MaskedWindowDataset(train_pieces, config.max_bars, repeats=repeats)
    weights = [weight for _, weight in train_sources] * repeats
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate)
    val_loader = None
    if val_sources:
        val_ds = MaskedWindowDataset([load_piece(path) for path, _ in val_sources], config.max_bars, repeats=2, deterministic=True, seed=seed)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = CandombeTransformer(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup = max(1, min(500, total_steps // 20))
    print(f"{len(train_sources)} train / {len(val_sources)} val pieces, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {total_steps} steps")

    history = []
    best_val = float("inf")
    stale = 0
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, lr, warmup)
            loss, _ = compute_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss)
            step += 1

        record = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "seconds": round(time.time() - started, 1)}
        if val_loader is not None:
            record.update({f"val_{key}": value for key, value in evaluate(model, val_loader, device).items()})
        history.append(record)
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in record.items()))

        score = record.get("val_loss", record["train_loss"])
        save_checkpoint(f"{save_prefix}_last.pt", model, {"history": history})
        if score < best_val:
            best_val = score
            stale = 0
            save_checkpoint(f"{save_prefix}_best.pt", model, {"history": history, "data_glob": data_glob})
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop: no validation improvement in {patience} epochs")
                break

    with open(f"{save_prefix}_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return {"best_val": best_val, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob", type=str, default="data/processed/synthetic/*.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=4, help="Random windows drawn per piece and epoch.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--save-prefix", type=str, default="models/candombe_v2")
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--layers", type=int, default=ModelConfig.num_layers)
    parser.add_argument("--heads", type=int, default=ModelConfig.nhead)
    parser.add_argument("--max-bars", type=int, default=ModelConfig.max_bars)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_loop(
        data_glob=args.data_glob,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        repeats=args.repeats,
        val_fraction=args.val_fraction,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_prefix=args.save_prefix,
        config=ModelConfig(d_model=args.d_model, nhead=args.heads, num_layers=args.layers, dim_feedforward=4 * args.d_model, max_bars=args.max_bars),
        seed=args.seed,
        patience=args.patience,
    )
