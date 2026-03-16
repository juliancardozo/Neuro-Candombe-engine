import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import SimpleCandombeTransformer

VEL_BUCKETS = 4

class GridDataset(Dataset):
    def __init__(self, npz_paths):
        self.items = []
        for p in npz_paths:
            d = np.load(p)["data"]  # shape (steps, 3)
            # quantize into VEL_BUCKETS
            bins = np.linspace(0, 127, VEL_BUCKETS+1)
            q = np.digitize(d, bins)  # 0 means no hit, else 1..VEL_BUCKETS
            steps = []
            for row in q:
                vec = []
                for inst in range(3):
                    onehot = np.zeros(VEL_BUCKETS, dtype=np.float32)
                    if row[inst] > 0:
                        onehot[row[inst]-1] = 1.0
                    vec.append(onehot)
                steps.append(np.concatenate(vec))
            arr = np.stack(steps)  # (S, input_dim)
            self.items.append(arr)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_pad(batch):
    # simple pad to max length in batch
    lengths = [b.shape[0] for b in batch]
    maxlen = max(lengths)
    dim = batch[0].shape[1]
    out = np.zeros((maxlen, len(batch), dim), dtype=np.float32)
    for i, b in enumerate(batch):
        out[:b.shape[0], i, :] = b
    return torch.from_numpy(out)

def train_loop(data_glob="data/processed/*.npz", epochs=10, lr=1e-4, device="cpu"):
    paths = glob.glob(data_glob)
    ds = GridDataset(paths)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda x: collate_pad(x))
    model = SimpleCandombeTransformer(input_dim=3*VEL_BUCKETS).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for batch in loader:
            # batch shape (S, B, D)
            batch = batch.to(device)
            src = batch[:-1]   # predict next
            tgt = batch[1:]
            logits = model(src)
            loss = loss_fn(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch} loss={total:.4f}")
        torch.save(model.state_dict(), f"models/candombe_epoch{epoch}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loop(epochs=args.epochs, device=device)