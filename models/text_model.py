"""Text models: character-level GPT for generation, Transformer for classification."""

import torch
import torch.nn as nn


def causal_mask(size: int, device) -> torch.Tensor:
    """Float causal mask (version-proof across torch releases)."""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class CharTokenizer:
    """Character-level tokenizer built from a raw text corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Cannot build a tokenizer from empty text.")
        self.itos = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def state(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.itos = state["itos"]
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        tok.vocab_size = len(tok.itos)
        return tok


class WordTokenizer:
    """Word-level tokenizer with <pad>/<unk> specials."""

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, texts, min_freq: int = 1, max_vocab: int = 20000):
        from collections import Counter

        counts = Counter()
        for t in texts:
            counts.update(self.split(t))
        words = [w for w, c in counts.most_common(max_vocab) if c >= min_freq]
        self.itos = [self.PAD, self.UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.pad_idx = 0

    @staticmethod
    def split(text: str) -> list:
        return str(text).lower().split()

    def encode(self, text: str, max_len: int) -> list:
        ids = [self.stoi.get(w, 1) for w in self.split(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    def state(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_state(cls, state: dict) -> "WordTokenizer":
        tok = cls.__new__(cls)
        tok.itos = state["itos"]
        tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        tok.vocab_size = len(tok.itos)
        tok.pad_idx = 0
        return tok


class CaptionTokenizer(WordTokenizer):
    """Word tokenizer with <pad>/<unk>/<bos>/<eos> for captioning."""

    BOS, EOS = "<bos>", "<eos>"

    def __init__(self, texts, min_freq: int = 1, max_vocab: int = 20000):
        super().__init__(texts, min_freq, max_vocab)
        # Rebuild with extra specials: pad=0, unk=1, bos=2, eos=3
        words = [w for w in self.itos if w not in (self.PAD, self.UNK)]
        self.itos = [self.PAD, self.UNK, self.BOS, self.EOS] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.pad_idx, self.bos_idx, self.eos_idx = 0, 2, 3

    def encode_caption(self, text: str, max_len: int) -> list:
        ids = [self.bos_idx]
        ids += [self.stoi.get(w, 1) for w in self.split(text)][: max_len - 2]
        ids.append(self.eos_idx)
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids[:max_len]

    def decode_caption(self, ids) -> str:
        words = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i in (self.pad_idx, self.bos_idx):
                continue
            words.append(self.itos[i] if 0 <= i < len(self.itos) else self.UNK)
        return " ".join(words)

    @classmethod
    def from_state(cls, state: dict) -> "CaptionTokenizer":
        tok = cls.__new__(cls)
        tok.itos = state["itos"]
        tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        tok.vocab_size = len(tok.itos)
        tok.pad_idx, tok.bos_idx, tok.eos_idx = 0, 2, 3
        return tok


class CharGPT(nn.Module):
    """Small GPT-style Transformer language model (character level)."""

    def __init__(self, vocab_size: int, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 4, block_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head,
                                       dim_feedforward=n_embd * 4, dropout=dropout,
                                       batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        if t > self.block_size:
            idx = idx[:, -self.block_size:]
            t = self.block_size
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = self.drop(x)
        mask = causal_mask(t, idx.device)
        for blk in self.blocks:
            x = blk(x, src_mask=mask)
        return self.head(self.ln(x))


class TextClassifier(nn.Module):
    """Transformer text classifier with masked mean pooling."""

    def __init__(self, vocab_size: int, n_classes: int, n_embd: int = 64,
                 n_head: int = 4, n_layer: int = 2, max_len: int = 64,
                 dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head,
                                       dim_feedforward=n_embd * 4, dropout=dropout,
                                       batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_embd, n_classes),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = self.drop(x)
        pad_mask = idx == self.pad_idx
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)
        keep = (~pad_mask).float().unsqueeze(-1).clamp(min=1e-6)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: CharTokenizer, prompt: str,
                  max_new_tokens: int = 120, temperature: float = 0.8,
                  top_k: int = 40, device="cpu") -> str:
    """Sample a continuation for `prompt` from a trained CharGPT."""
    model.eval()
    ids = tokenizer.encode(prompt) if prompt else [0]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    temperature = max(temperature, 1e-6)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k and top_k < logits.shape[-1]:
            cutoff = torch.topk(logits, top_k).values[:, -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0].tolist())
