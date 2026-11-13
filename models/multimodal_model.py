"""Multimodal models: CNN encoder + Transformer decoder for image captioning."""

import torch
import torch.nn as nn

from models.text_model import causal_mask


class ImageCaptioner(nn.Module):
    """Generates a caption conditioned on image region features."""

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 n_layer: int = 2, max_len: int = 32, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.enc_proj = nn.Linear(128, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=d_model * 4,
                                       dropout=dropout, batch_first=True),
            num_layers=n_layer,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)              # (B, 128, 4, 4)
        feats = feats.flatten(2).transpose(1, 2)  # (B, 16, 128)
        return self.enc_proj(feats)               # (B, 16, d)

    def forward(self, images: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encode_images(images)
        _, t = tgt.shape
        x = self.tok_emb(tgt) + self.pos_emb(torch.arange(t, device=tgt.device))
        x = self.drop(x)
        out = self.decoder(x, memory, tgt_mask=causal_mask(t, tgt.device),
                           tgt_key_padding_mask=(tgt == self.pad_idx))
        return self.head(out)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_idx: int, eos_idx: int,
                 max_len: int = None) -> torch.Tensor:
        """Greedy caption decoding. Returns token ids (B, <=max_len)."""
        self.eval()
        max_len = max_len or self.max_len
        memory = self.encode_images(images)
        b = images.shape[0]
        ids = torch.full((b, 1), bos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            t = ids.shape[1]
            x = self.tok_emb(ids) + self.pos_emb(torch.arange(t, device=ids.device))
            logits = self.head(self.decoder(x, memory, tgt_mask=causal_mask(t, ids.device)))
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            finished |= (nxt[:, 0] == eos_idx)
            if finished.all():
                break
        return ids
