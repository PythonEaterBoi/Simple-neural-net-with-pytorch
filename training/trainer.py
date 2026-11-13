"""Training loops for every modality/task.

Each ``train_*`` function has the signature::

    train_*(data_dir: str, config: dict, cb: Callback, out_path: str) -> dict

``cb`` is any object with ``log(str)``, ``metric(dict)`` and ``cancelled()``.
Returns a summary dict including the checkpoint path.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.image_model import Diffusion, ImageClassifier, TinyUNet
from models.multimodal_model import ImageCaptioner
from models.tabular_model import TabularMLP
from models.text_model import (CaptionTokenizer, CharGPT, CharTokenizer,
                               TextClassifier, WordTokenizer)
from models.video_model import VideoClassifier
from training.datasets import (CaptionDataset, CharDataset, DiffusionDataset,
                               ImageFolderDataset, TabularData, TextClassDataset,
                               VideoFolderDataset, load_caption_data,
                               load_text_class_csv, load_text_corpus,
                               train_val_split)


# ------------------------------------------------------------------ helpers
def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def save_checkpoint(out_path: str, modality: str, task: str, config: dict,
                    arch: dict, meta: dict, model: nn.Module) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"modality": modality, "task": task, "config": config,
                "arch": arch, "meta": meta,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()}},
               out_path)
    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"modality": modality, "task": task, "config": config,
                   "arch": arch, "meta": meta, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "path": os.path.basename(out_path)}, f, indent=2)
    return out_path


def _loader(ds, batch_size: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _evaluate_classifier(model, loader, device, loss_fn):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        tot_loss += loss_fn(logits, yb).item() * len(xb)
        tot_acc += accuracy(logits, yb) * len(xb)
        n += len(xb)
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


# ---------------------------------------------------------- text generation
def train_text_generation(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    text = load_text_corpus(data_dir)
    tok = CharTokenizer(text)
    cb.log(f"Corpus: {len(text):,} chars, vocab size {tok.vocab_size} on {device}.")
    ds = CharDataset(text, config["block_size"], tok.stoi)
    train_ds, val_ds = train_val_split(ds, 0.05)
    train_dl, val_dl = _loader(train_ds, config["batch_size"], True), \
        (_loader(val_ds, config["batch_size"], False) if val_ds else None)
    arch = {"vocab_size": tok.vocab_size, "n_embd": config["n_embd"],
            "n_head": config["n_head"], "n_layer": config["n_layer"],
            "block_size": config["block_size"], "dropout": config["dropout"]}
    model = CharGPT(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).reshape(-1, tok.vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        train_loss = tot / max(n, 1)
        row = {"epoch": epoch, "loss": train_loss}
        if val_dl:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    vt += loss_fn(model(xb).reshape(-1, tok.vocab_size), yb.reshape(-1)).item() * len(xb)
                    vn += len(xb)
            row["val_loss"] = vt / max(vn, 1)
            if row["val_loss"] < best:
                best, best_state = row["val_loss"], {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={train_loss:.4f}"
               + (f" val_loss={row['val_loss']:.4f}" if "val_loss" in row else ""))
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "text", "generation", config, arch,
                    {"tokenizer": tok.state()}, model)
    return {"checkpoint": out_path, "vocab_size": tok.vocab_size}


# -------------------------------------------------------- text classification
def train_text_classification(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    texts, labels, label_names = load_text_class_csv(data_dir, config["text_col"], config["label_col"])
    tok = WordTokenizer(texts)
    cb.log(f"{len(texts)} samples, {len(label_names)} classes {label_names}, vocab {tok.vocab_size} on {device}.")
    ds = TextClassDataset(texts, labels, tok, config["max_len"])
    train_ds, val_ds = train_val_split(ds, 0.15)
    train_dl = _loader(train_ds, config["batch_size"], True)
    val_dl = _loader(val_ds, config["batch_size"], False) if val_ds else None
    arch = {"vocab_size": tok.vocab_size, "n_classes": len(label_names),
            "n_embd": config["n_embd"], "n_head": config["n_head"],
            "n_layer": config["n_layer"], "max_len": config["max_len"]}
    model = TextClassifier(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, ta, n = 0.0, 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            ta += accuracy(logits, yb) * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1), "acc": ta / max(n, 1)}
        if val_dl:
            vl, va = _evaluate_classifier(model, val_dl, device, loss_fn)
            row.update(val_loss=vl, val_acc=va)
            if vl < best:
                best, best_state = vl, {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f} acc={row['acc']:.3f}"
               + (f" val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f}" if val_dl else ""))
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "text", "classification", config, arch,
                    {"tokenizer": tok.state(), "labels": label_names}, model)
    return {"checkpoint": out_path, "labels": label_names}


# ------------------------------------------------------- image classification
def train_image_classification(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    ds = ImageFolderDataset(data_dir, config["img_size"], config.get("augment", True))
    cb.log(f"{len(ds)} images, {len(ds.classes)} classes {ds.classes} on {device}.")
    train_ds, val_ds = train_val_split(ds, 0.15)
    train_dl = _loader(train_ds, config["batch_size"], True)
    val_dl = _loader(val_ds, config["batch_size"], False) if val_ds else None
    arch = {"n_classes": len(ds.classes), "base": config.get("base_channels", 32)}
    model = ImageClassifier(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, ta, n = 0.0, 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            ta += accuracy(logits, yb) * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1), "acc": ta / max(n, 1)}
        if val_dl:
            vl, va = _evaluate_classifier(model, val_dl, device, loss_fn)
            row.update(val_loss=vl, val_acc=va)
            if vl < best:
                best, best_state = vl, {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f} acc={row['acc']:.3f}"
               + (f" val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f}" if val_dl else ""))
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "image", "classification", config, arch,
                    {"labels": ds.classes, "img_size": config["img_size"]}, model)
    return {"checkpoint": out_path, "labels": ds.classes}


# ---------------------------------------------------------- image generation
def train_image_generation(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    ds = DiffusionDataset(data_dir)
    cb.log(f"{len(ds)} images, {len(ds.classes)} classes {ds.classes} on {device}.")
    dl = _loader(ds, config["batch_size"], True)
    arch = {"n_classes": len(ds.classes), "base": config.get("base_channels", 32)}
    model = TinyUNet(**arch).to(device)
    diffusion = Diffusion(timesteps=config.get("timesteps", 150), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = diffusion.training_step(model, xb, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1)}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f}")
    save_checkpoint(out_path, "image", "generation", config, arch,
                    {"labels": ds.classes, "timesteps": config.get("timesteps", 150)}, model)
    return {"checkpoint": out_path, "labels": ds.classes}


# ------------------------------------------------------- video classification
def train_video_classification(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    ds = VideoFolderDataset(data_dir, config.get("num_frames", 8), config["img_size"])
    cb.log(f"{len(ds)} clips, {len(ds.classes)} classes {ds.classes} on {device}.")
    train_ds, val_ds = train_val_split(ds, 0.15)
    train_dl = _loader(train_ds, config["batch_size"], True)
    val_dl = _loader(val_ds, config["batch_size"], False) if val_ds else None
    arch = {"n_classes": len(ds.classes), "emb_dim": config.get("emb_dim", 128)}
    model = VideoClassifier(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, ta, n = 0.0, 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            ta += accuracy(logits, yb) * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1), "acc": ta / max(n, 1)}
        if val_dl:
            vl, va = _evaluate_classifier(model, val_dl, device, loss_fn)
            row.update(val_loss=vl, val_acc=va)
            if vl < best:
                best, best_state = vl, {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f} acc={row['acc']:.3f}"
               + (f" val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f}" if val_dl else ""))
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "video", "classification", config, arch,
                    {"labels": ds.classes, "img_size": config["img_size"],
                     "num_frames": config.get("num_frames", 8)}, model)
    return {"checkpoint": out_path, "labels": ds.classes}


# ------------------------------------------------------- multimodal captioning
def train_captioning(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    pairs = load_caption_data(data_dir, config["image_col"], config["caption_col"])
    tok = CaptionTokenizer([c for _, c in pairs])
    cb.log(f"{len(pairs)} image-caption pairs, vocab {tok.vocab_size} on {device}.")
    ds = CaptionDataset(pairs, tok, config["img_size"], config["max_len"])
    train_ds, val_ds = train_val_split(ds, 0.1)
    train_dl = _loader(train_ds, config["batch_size"], True)
    val_dl = _loader(val_ds, config["batch_size"], False) if val_ds else None
    arch = {"vocab_size": tok.vocab_size, "d_model": config["n_embd"],
            "nhead": config["n_head"], "n_layer": config["n_layer"],
            "max_len": config["max_len"]}
    model = ImageCaptioner(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_idx)
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, yb[:, :-1]).reshape(-1, tok.vocab_size), yb[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1)}
        if val_dl:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    vt += loss_fn(model(xb, yb[:, :-1]).reshape(-1, tok.vocab_size),
                                  yb[:, 1:].reshape(-1)).item() * len(xb)
                    vn += len(xb)
            row["val_loss"] = vt / max(vn, 1)
            if row["val_loss"] < best:
                best, best_state = row["val_loss"], {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        cb.log(f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f}"
               + (f" val_loss={row['val_loss']:.4f}" if "val_loss" in row else ""))
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "multimodal", "captioning", config, arch,
                    {"tokenizer": tok.state(), "img_size": config["img_size"],
                     "max_len": config["max_len"]}, model)
    return {"checkpoint": out_path}


# ------------------------------------------------------------------ tabular
def train_tabular(data_dir, config, cb, out_path):
    device = get_device(config.get("device", "auto"))
    data = TabularData(data_dir, config["target_col"], config.get("problem_type", "auto"))
    cb.log(f"{len(data.X)} rows, {len(data.feature_names)} features, "
           f"{data.problem_type} on {device}.")
    train_ds, val_ds = train_val_split(data.dataset(), 0.15)
    train_dl = _loader(train_ds, config["batch_size"], True)
    val_dl = _loader(val_ds, config["batch_size"], False) if val_ds else None
    arch = {"in_dim": len(data.feature_names), "out_dim": data.out_dim,
            "hidden": config.get("hidden_size", 64),
            "n_layers": config.get("n_layers", 2),
            "dropout": config.get("dropout", 0.1)}
    model = TabularMLP(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    is_clf = data.problem_type == "classification"
    loss_fn = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    best, best_state = float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        if cb.cancelled():
            break
        model.train()
        tot, ta, n = 0.0, 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb) if is_clf else loss_fn(out[:, 0], yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            if is_clf:
                ta += accuracy(out, yb) * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "loss": tot / max(n, 1)}
        if is_clf:
            row["acc"] = ta / max(n, 1)
        if val_dl:
            model.eval()
            vt, va, vn = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vt += (loss_fn(out, yb) if is_clf else loss_fn(out[:, 0], yb)).item() * len(xb)
                    if is_clf:
                        va += accuracy(out, yb) * len(xb)
                    vn += len(xb)
            row["val_loss"] = vt / max(vn, 1)
            if is_clf:
                row["val_acc"] = va / max(vn, 1)
            if row["val_loss"] < best:
                best, best_state = row["val_loss"], {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cb.metric(row)
        msg = f"epoch {epoch}/{config['epochs']} loss={row['loss']:.4f}"
        if is_clf:
            msg += f" acc={row['acc']:.3f}"
        if val_dl:
            msg += f" val_loss={row['val_loss']:.4f}"
            if is_clf:
                msg += f" val_acc={row['val_acc']:.3f}"
        cb.log(msg)
        if epoch % 10 == 0 or epoch == config["epochs"]:
            pass
    if best_state:
        model.load_state_dict(best_state)
    save_checkpoint(out_path, "tabular", "predict", config, arch,
                    data.meta(), model)
    return {"checkpoint": out_path, "problem_type": data.problem_type}


TRAIN_FUNCS = {
    ("text", "generation"): train_text_generation,
    ("text", "classification"): train_text_classification,
    ("image", "classification"): train_image_classification,
    ("image", "generation"): train_image_generation,
    ("video", "classification"): train_video_classification,
    ("multimodal", "captioning"): train_captioning,
    ("tabular", "predict"): train_tabular,
}
