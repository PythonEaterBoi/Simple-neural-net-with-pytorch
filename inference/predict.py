"""Prompt interface: load a checkpoint and run inference for any modality."""

import base64
import io
import json
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.image_model import Diffusion, ImageClassifier, TinyUNet
from models.multimodal_model import ImageCaptioner
from models.tabular_model import TabularMLP
from models.text_model import (CaptionTokenizer, CharGPT, CharTokenizer,
                               TextClassifier, WordTokenizer, generate_text)
from models.video_model import VideoClassifier
from training.datasets import (load_clip_for_inference, load_image_rgb,
                               pil_to_tensor)
from training.trainer import get_device

CHECKPOINTS_ROOT = "checkpoints"
PREDICTIONS_ROOT = "predictions"

_cache: dict = {}
_cache_lock = threading.Lock()


# --------------------------------------------------------------- checkpoints
def list_checkpoints(root: str = CHECKPOINTS_ROOT) -> list:
    os.makedirs(root, exist_ok=True)
    out = []
    for f in sorted(os.listdir(root)):
        if not f.endswith(".pt"):
            continue
        sidecar = os.path.join(root, os.path.splitext(f)[0] + ".json")
        info = {"name": f, "modality": "?", "task": "?",
                "size_mb": round(os.path.getsize(os.path.join(root, f)) / 1e6, 2)}
        meta = None
        if os.path.exists(sidecar):
            try:
                with open(sidecar, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                meta = None
        if meta is None:
            # Fall back to reading the header embedded in the weights file
            # (covers checkpoints copied in by hand / trained via CLI elsewhere).
            try:
                payload = torch.load(os.path.join(root, f), map_location="cpu")
                meta = {"modality": payload.get("modality"), "task": payload.get("task"),
                        "config": payload.get("config", {}), "meta": payload.get("meta", {}),
                        "created": "(from weights file)"}
            except Exception:
                meta = None
        if meta:
            info.update({k: meta.get(k) for k in ("modality", "task", "created", "config")})
            info["meta"] = _ui_meta(meta.get("meta", {}))
        out.append(info)
    return sorted(out, key=lambda d: d["name"], reverse=True)


def _ui_meta(meta: dict) -> dict:
    """Trim checkpoint meta down to what the UI needs (no giant vocabs)."""
    slim = {}
    for key in ("labels", "feature_names", "problem_type", "img_size",
                "num_frames", "max_len", "timesteps"):
        if key in meta:
            slim[key] = meta[key]
    tok = (meta.get("tokenizer") or {})
    if tok.get("itos"):
        slim["vocab_size"] = len(tok["itos"])
    return slim


def delete_checkpoint(name: str, root: str = CHECKPOINTS_ROOT):
    path = os.path.join(root, name)
    if not os.path.isfile(path) or not name.endswith(".pt"):
        raise FileError(f"Unknown checkpoint: {name}")
    os.remove(path)
    sidecar = os.path.join(root, os.path.splitext(name)[0] + ".json")
    if os.path.exists(sidecar):
        os.remove(sidecar)
    with _cache_lock:
        _cache.pop(name, None)


class FileError(Exception):
    pass


def _build_model(modality: str, task: str, arch: dict):
    if modality == "text" and task == "generation":
        return CharGPT(**arch)
    if modality == "text" and task == "classification":
        return TextClassifier(**arch)
    if modality == "image" and task == "classification":
        return ImageClassifier(**arch)
    if modality == "image" and task == "generation":
        return TinyUNet(**arch)
    if modality == "video":
        return VideoClassifier(**arch)
    if modality == "multimodal":
        return ImageCaptioner(**arch)
    if modality == "tabular":
        return TabularMLP(**arch)
    raise ValueError(f"Unsupported checkpoint type: {modality}/{task}")


def load_checkpoint(name: str, root: str = CHECKPOINTS_ROOT,
                    device=None) -> dict:
    """Load (and cache) a checkpoint. Returns {payload, model, device}."""
    with _cache_lock:
        if name in _cache:
            return _cache[name]
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        raise FileError(f"Unknown checkpoint: {name}")
    device = device or get_device("auto")
    payload = torch.load(path, map_location=device)
    model = _build_model(payload["modality"], payload["task"], payload["arch"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    entry = {"payload": payload, "model": model, "device": device}
    with _cache_lock:
        _cache[name] = entry
    return entry


# ------------------------------------------------------------------ predict
def _probs_result(labels, probs: torch.Tensor, top_k: int = 3) -> dict:
    k = min(top_k, len(labels))
    vals, idxs = torch.topk(probs, k)
    ranked = [{"label": labels[i], "prob": round(float(v), 4)}
              for v, i in zip(vals, idxs)]
    return {"type": "label", "label": ranked[0]["label"],
            "confidence": ranked[0]["prob"], "ranking": ranked}


def predict(checkpoint: str, prompt_text: str = "", media_path: str = None,
            values=None, options: dict = None) -> dict:
    """Run inference with a trained checkpoint.

    Args:
        checkpoint: checkpoint file name (from list_checkpoints).
        prompt_text: text prompt (text gen/classification, image-gen class).
        media_path: uploaded image/video path (image/video/captioning).
        values: dict or list of feature values (tabular).
        options: generation options (temperature, max_tokens, top_k, n_images).
    """
    options = options or {}
    entry = load_checkpoint(checkpoint)
    payload, model, device = entry["payload"], entry["model"], entry["device"]
    modality, task, meta = payload["modality"], payload["task"], payload["meta"]

    if modality == "text" and task == "generation":
        tok = CharTokenizer.from_state(meta["tokenizer"])
        text = generate_text(
            model, tok, prompt_text or "",
            max_new_tokens=int(options.get("max_tokens", 120)),
            temperature=float(options.get("temperature", 0.8)),
            top_k=int(options.get("top_k", 40)), device=device)
        return {"type": "text", "text": text}

    if modality == "text" and task == "classification":
        if not (prompt_text or "").strip():
            raise ValueError("Enter some text to classify.")
        tok = WordTokenizer.from_state(meta["tokenizer"])
        max_len = payload["arch"].get("max_len", 64)
        ids = torch.tensor([tok.encode(prompt_text, max_len)], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = F.softmax(model(ids), dim=-1)[0].cpu()
        return _probs_result(meta["labels"], probs)

    if modality == "image" and task == "classification":
        if not media_path:
            raise ValueError("Upload an image to classify.")
        img = load_image_rgb(media_path, meta.get("img_size", 64))
        xb = pil_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(xb), dim=-1)[0].cpu()
        return _probs_result(meta["labels"], probs)

    if modality == "image" and task == "generation":
        labels = meta["labels"]
        want = (prompt_text or "").strip()
        if want == "":
            yi = 0
        elif want.isdigit() and int(want) < len(labels):
            yi = int(want)
        else:
            match = [i for i, l in enumerate(labels) if l.lower() == want.lower()]
            if not match:
                raise ValueError(f"Unknown class '{want}'. Available: {', '.join(labels)}.")
            yi = match[0]
        n = max(1, min(int(options.get("n_images", 2)), 4))
        diffusion = Diffusion(timesteps=meta.get("timesteps", 150), device=device)
        y = torch.full((n,), yi, dtype=torch.long, device=device)
        imgs = diffusion.sample(model, y)
        os.makedirs(PREDICTIONS_ROOT, exist_ok=True)
        out_images = []
        for i in range(n):
            arr = (imgs[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            pil = Image.fromarray((arr * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out_images.append(base64.b64encode(buf.getvalue()).decode())
            pil.save(os.path.join(PREDICTIONS_ROOT, f"{os.path.splitext(checkpoint)[0]}_{labels[yi]}_{i}.png"))
        return {"type": "images", "label": labels[yi], "images": out_images}

    if modality == "video":
        if not media_path:
            raise ValueError("Upload a video (or frame-folder zip) to classify.")
        clip = load_clip_for_inference(media_path, meta.get("num_frames", 8),
                                       meta.get("img_size", 64)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(clip), dim=-1)[0].cpu()
        return _probs_result(meta["labels"], probs)

    if modality == "multimodal":
        if not media_path:
            raise ValueError("Upload an image to caption.")
        tok = CaptionTokenizer.from_state(meta["tokenizer"])
        img = load_image_rgb(media_path, meta.get("img_size", 64))
        xb = pil_to_tensor(img).unsqueeze(0).to(device)
        ids = model.generate(xb, tok.bos_idx, tok.eos_idx,
                             max_len=meta.get("max_len", 32))
        return {"type": "text", "text": tok.decode_caption(ids[0].tolist())}

    if modality == "tabular":
        feats = meta["feature_names"]
        if isinstance(values, dict):
            missing = [f for f in feats if f not in values or str(values[f]).strip() == ""]
            if missing:
                raise ValueError(f"Missing values for: {', '.join(missing)}")
            row = [float(values[f]) for f in feats]
        elif isinstance(values, (list, tuple)) and len(values) == len(feats):
            row = [float(v) for v in values]
        else:
            raise ValueError(f"Provide {len(feats)} feature values: {', '.join(feats)}.")
        x = (np.array(row, np.float32) - np.array(meta["means"], np.float32)) \
            / np.array(meta["stds"], np.float32)
        xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(xb)[0].cpu()
        if meta["problem_type"] == "classification":
            return _probs_result(meta["labels"], F.softmax(out, dim=-1))
        return {"type": "number", "value": round(float(out.item()), 4)}

    raise ValueError(f"Unsupported checkpoint type: {modality}/{task}")
