import torch
import torch.nn as nn

class SimpleCandombeTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, tgt_mask=None):
        # x: (S, B, D)
        x_proj = self.input_proj(x)
        # memory and tgt both x_proj for simple modeling (teacher forcing elsewhere)
        memory = x_proj
        tgt = x_proj
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        logits = self.output_proj(out)
        return logits