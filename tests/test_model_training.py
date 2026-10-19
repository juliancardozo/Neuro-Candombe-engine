import numpy as np
import torch

from candombe_rules import random_structure, render_piece
from model import OFF_MASK, VEL_MASK, art_mask, load_checkpoint, save_checkpoint
from representation import N_INSTR
from train import IGNORE, MaskedWindowDataset, collate, compute_loss, encode_example


def test_targets_are_never_visible_in_the_input(rng):
    """Regression for the v1 bug: the model must never see what it is scored on."""
    piece = render_piece(rng, random_structure(rng, 8))
    for seed in range(300):
        example = encode_example(piece, 0, piece.steps, np.random.default_rng(seed))
        scored_vel = example["vel_t"] != IGNORE
        assert scored_vel.any() or (example["off_t"] != IGNORE).any()
        assert (example["vel_in"][scored_vel] == VEL_MASK).all()
        scored_off = example["off_t"] != IGNORE
        assert (example["off_in"][scored_off] == OFF_MASK).all()
        for inst in range(N_INSTR):
            scored_art = example["art_t"][:, inst] != IGNORE
            assert (example["art_in"][scored_art, inst] == art_mask(inst)).all()
        # A hidden velocity also hides that cell's timing and articulation.
        hidden = example["vel_in"] == VEL_MASK
        assert (example["off_in"][hidden] == OFF_MASK).all()


def test_padding_does_not_change_predictions(rng, tiny_model):
    short = render_piece(rng, ["base"] * 2)
    long = render_piece(rng, ["base"] * 4)
    a = encode_example(short, 0, short.steps, np.random.default_rng(1), drop_conditions=False)
    b = encode_example(long, 0, long.steps, np.random.default_rng(2), drop_conditions=False)

    def run(batch):
        with torch.no_grad():
            return tiny_model(batch["vel"], batch["off"], batch["art"], batch["section"], batch["tempo"], batch["density"], batch["style"], batch["pad_mask"])

    alone = run(collate([a]))["vel"][0]
    padded = run(collate([a, b]))["vel"][0, : short.steps]
    torch.testing.assert_close(alone, padded, atol=1e-5, rtol=1e-4)


def test_loss_backpropagates_and_learns(rng, tiny_model):
    pieces = [render_piece(np.random.default_rng(i), random_structure(np.random.default_rng(i), 4)) for i in range(8)]
    ds = MaskedWindowDataset(pieces, max_bars=4, deterministic=True)
    batch = collate([ds[i] for i in range(len(ds))])
    tiny_model.train()
    opt = torch.optim.Adam(tiny_model.parameters(), lr=3e-3)
    first = None
    for _ in range(40):
        loss, stats = compute_loss(tiny_model, batch)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.8
    assert {"vel", "off", "art"} <= set(stats)


def test_checkpoint_round_trip_and_legacy_rejected(tmp_path, tiny_model):
    path = tmp_path / "model.pt"
    save_checkpoint(str(path), tiny_model)
    loaded = load_checkpoint(str(path))
    assert loaded.config == tiny_model.config

    legacy = tmp_path / "legacy.pt"
    torch.save(tiny_model.state_dict(), legacy)
    try:
        load_checkpoint(str(legacy))
    except ValueError:
        pass
    else:
        raise AssertionError("legacy checkpoints must be rejected")
