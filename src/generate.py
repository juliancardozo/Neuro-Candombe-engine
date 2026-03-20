import numpy as np
import torch
try:
    from .model import SimpleCandombeTransformer
except ImportError:
    from model import SimpleCandombeTransformer

VEL_BUCKETS = 4
N_INSTR = 3
INPUT_DIM = N_INSTR * VEL_BUCKETS
INSTS = ["chico", "repique", "piano"]


def load_model(path, device="cpu"):
    model = SimpleCandombeTransformer(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


def _sample_bucket(probs, min_prob=0.10):
    """Sample one velocity bucket from probabilities.

    Returns -1 for silence or 0..VEL_BUCKETS-1 for a hit.
    """
    probs = np.asarray(probs, dtype=np.float64)
    best = int(np.argmax(probs))
    if probs[best] < min_prob:
        return -1

    p = probs / max(probs.sum(), 1e-9)
    return int(np.random.choice(np.arange(VEL_BUCKETS), p=p))


def sample(model, seed_seq, length_steps=64, temperature=0.9, density=0.55, device="cpu"):
    """Autoregressive generation with light musical constraints.

    density in [0,1] controls how often each instrument can fire.
    """
    if seed_seq.ndim != 2 or seed_seq.shape[1] != INPUT_DIM:
        raise ValueError(f"seed_seq must have shape (S, {INPUT_DIM})")

    seq = [torch.from_numpy(step).float().to(device) for step in seed_seq]

    for i in range(length_steps):
        src = torch.stack(seq, dim=0).unsqueeze(1)  # S, B=1, D
        with torch.no_grad():
            logits = model(src)

        next_logits = logits[-1, 0] / max(1e-6, temperature)
        probs = torch.sigmoid(next_logits).cpu().numpy().reshape(N_INSTR, VEL_BUCKETS)

        # Bias density by instrument role: chico steady, repique syncopated, piano anchors pulse.
        inst_density = np.array(
            [min(1.0, density + 0.20), max(0.05, density - 0.10), density], dtype=np.float32
        )

        next_step = np.zeros((N_INSTR, VEL_BUCKETS), dtype=np.float32)
        for inst_idx in range(N_INSTR):
            if np.random.rand() > inst_density[inst_idx]:
                continue
            bucket = _sample_bucket(probs[inst_idx])
            if bucket >= 0:
                next_step[inst_idx, bucket] = 1.0

        # Keep floor pulse if everything went silent.
        if next_step.sum() == 0 and ((i + 1) % 4 == 0):
            next_step[0, 0] = 1.0

        seq.append(torch.from_numpy(next_step.reshape(-1)).to(device))

    out = torch.stack(seq, dim=0).cpu().numpy()  # (seed_len+L, D)
    return out


def events_from_array(arr, step_duration=0.125):
    events = []
    for i in range(arr.shape[0]):
        row = arr[i]
        for inst_idx in range(N_INSTR):
            block = row[inst_idx * VEL_BUCKETS : (inst_idx + 1) * VEL_BUCKETS]
            if block.sum() > 0:
                vel_bucket = int(block.argmax())
                vel = int((vel_bucket + 1) / VEL_BUCKETS * 127)
                events.append(
                    {"time": i * step_duration, "instrument": INSTS[inst_idx], "velocity": vel}
                )
    return events
