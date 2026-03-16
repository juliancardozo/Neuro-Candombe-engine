import numpy as np
import torch
from model import SimpleCandombeTransformer

VEL_BUCKETS = 4
INSTS = ["chico","repique","piano"]

def load_model(path, device="cpu"):
    model = SimpleCandombeTransformer(input_dim=3*VEL_BUCKETS)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def sample(model, seed_seq, length_steps=64, temperature=1.0, device="cpu"):
    seq = [torch.from_numpy(seed_seq).float().to(device)]
    for _ in range(length_steps):
        src = torch.stack(seq, dim=0)  # S, D
        src_in = src.unsqueeze(1)  # S, B=1, D
        with torch.no_grad():
            logits = model(src_in)  # S, B, D
        next_logits = logits[-1,0] / max(1e-6, temperature)
        probs = torch.sigmoid(next_logits).cpu().numpy()
        samp = (probs > 0.5).astype(np.float32)
        seq.append(torch.from_numpy(samp).to(device))
    out = torch.stack(seq, dim=0).cpu().numpy()  # (S+L, D)
    return out

def events_from_array(arr, step_duration=0.125):
    events = []
    for i in range(arr.shape[0]):
        row = arr[i]
        for inst_idx in range(3):
            block = row[inst_idx*VEL_BUCKETS:(inst_idx+1)*VEL_BUCKETS]
            if block.sum() > 0:
                vel_bucket = int(block.argmax())
                vel = int((vel_bucket+1) / VEL_BUCKETS * 127)
                events.append({"time": i*step_duration, "instrument": INSTS[inst_idx], "velocity": vel})
    return events