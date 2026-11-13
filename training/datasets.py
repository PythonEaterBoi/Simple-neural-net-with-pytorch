"""Dataset loaders for every supported modality/task.

All loaders read from a plain directory of uploaded (and zip-extracted) files,
so training works identically from the web UI and the CLI.
"""

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ------------------------------------------------------------------ helpers
def find_files(root: str, exts: set) -> list:
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def load_image_rgb(path: str, img_size: int) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").resize((img_size, img_size))


def pil_to_tensor(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - 0.5) / 0.5 if normalize else t


def read_csv_rows(path: str) -> tuple:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {os.path.basename(path)}.")
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    return reader.fieldnames, rows


def train_val_split(dataset: Dataset, val_frac: float = 0.15, seed: int = 42):
    if len(dataset) < 5 or val_frac <= 0:
        return dataset, None
    n_val = max(1, int(len(dataset) * val_frac))
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [len(dataset) - n_val, n_val], generator=gen)


def find_image_root(data_dir: str) -> str:
    """Return the directory whose immediate children are class folders.

    Handles zips that wrap everything in a single top-level folder.
    """
    def has_direct_images(d: str) -> bool:
        """Does directory `d` directly contain image files (non-recursive)?"""
        try:
            return any(os.path.isfile(os.path.join(d, f))
                       and os.path.splitext(f)[1].lower() in IMAGE_EXTS
                       for f in os.listdir(d))
        except OSError:
            return False

    def has_class_images(d: str) -> bool:
        try:
            subs = [os.path.join(d, s) for s in os.listdir(d)
                    if os.path.isdir(os.path.join(d, s))]
        except OSError:
            return False
        return any(has_direct_images(s) for s in subs)

    if has_class_images(data_dir):
        return data_dir
    for entry in sorted(os.listdir(data_dir)):
        cand = os.path.join(data_dir, entry)
        if os.path.isdir(cand) and has_class_images(cand):
            return cand
    raise ValueError(
        "Could not find class folders with images. Expected a folder per "
        "class, e.g. mydata.zip containing cats/*.jpg and dogs/*.jpg."
    )


# -------------------------------------------------------------------- text
class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int, stoi: dict):
        self.ids = [stoi.get(ch, 0) for ch in text]
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + self.block_size + 1], dtype=torch.long)
        return x, y


def _as_file_list(path: str, exts: set, kind: str) -> list:
    """Accept either a single file or a directory; return matching files."""
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() not in exts:
            raise ValueError(f"Expected a {kind} file, got: {os.path.basename(path)}")
        return [path]
    return find_files(path, exts)


def load_text_corpus(data_dir: str) -> str:
    files = _as_file_list(data_dir, {".txt"}, ".txt")
    if not files:
        raise ValueError("No .txt files found in the uploaded data.")
    parts = []
    for p in files:
        with open(p, encoding="utf-8", errors="ignore") as f:
            parts.append(f.read())
    text = "\n".join(parts)
    if len(text.strip()) < 200:
        raise ValueError("Text corpus is too small (need at least ~200 characters).")
    return text


class TextClassDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (torch.tensor(self.tok.encode(self.texts[i], self.max_len), dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))


def load_text_class_csv(data_dir: str, text_col: str, label_col: str):
    files = _as_file_list(data_dir, {".csv"}, ".csv")
    if not files:
        raise ValueError("No .csv file found in the uploaded data.")
    fields, rows = read_csv_rows(files[0])
    if text_col not in fields or label_col not in fields:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns "
                         f"(found: {', '.join(fields)}).")
    texts = [r[text_col] for r in rows]
    raw_labels = [str(r[label_col]).strip() for r in rows]
    labels_sorted = sorted(set(raw_labels))
    lmap = {l: i for i, l in enumerate(labels_sorted)}
    return texts, [lmap[l] for l in raw_labels], labels_sorted


# ------------------------------------------------------------------- image
class ImageFolderDataset(Dataset):
    def __init__(self, root: str, img_size: int = 64, augment: bool = False):
        self.root = find_image_root(root)
        self.img_size, self.augment = img_size, augment
        self.classes = sorted(d for d in os.listdir(self.root)
                              if os.path.isdir(os.path.join(self.root, d)))
        self.samples = []
        for ci, cls in enumerate(self.classes):
            for p in find_files(os.path.join(self.root, cls), IMAGE_EXTS):
                self.samples.append((p, ci))
        if len(self.classes) < 2:
            raise ValueError(f"Need at least 2 class folders, found: {self.classes}.")
        if not self.samples:
            raise ValueError("No images found in the class folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image_rgb(path, self.img_size)
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return pil_to_tensor(img), torch.tensor(label, dtype=torch.long)


class DiffusionDataset(ImageFolderDataset):
    """Same layout as ImageFolderDataset but fixed at 32x32 in [-1, 1]."""

    def __init__(self, root: str):
        super().__init__(root, img_size=32, augment=False)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image_rgb(path, 32)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return pil_to_tensor(img), torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------- video
def read_video_frames(path: str, num_frames: int, img_size: int) -> torch.Tensor:
    """Uniformly sample `num_frames` frames from a video file. (T, C, H, W)."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "Decoding video files requires opencv-python-headless "
            "(`pip install opencv-python-headless`), or use frame-folder clips instead."
        ) from e
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {os.path.basename(path)}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # Fall back to reading everything.
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            total = len(frames)
            if total == 0:
                raise ValueError(f"No frames decoded from {os.path.basename(path)}")
            idxs = np.linspace(0, total - 1, num_frames).round().astype(int)
            picked = [frames[i] for i in idxs]
        else:
            idxs = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int)
            picked = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, fr = cap.read()
                if ok:
                    picked.append(fr)
            while len(picked) < num_frames:  # pad short clips
                picked.append(picked[-1].copy() if picked else np.zeros((img_size, img_size, 3), np.uint8))
        out = []
        for fr in picked[:num_frames]:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(fr).resize((img_size, img_size))
            out.append(pil_to_tensor(img))
        return torch.stack(out)
    finally:
        cap.release()


def read_frame_dir(path: str, num_frames: int, img_size: int) -> torch.Tensor:
    """Read a folder of image frames as one clip. (T, C, H, W)."""
    files = find_files(path, IMAGE_EXTS)
    if not files:
        raise ValueError(f"No frames found in {path}.")
    idxs = np.linspace(0, len(files) - 1, num_frames).round().astype(int)
    return torch.stack([pil_to_tensor(load_image_rgb(files[i], img_size)) for i in idxs])


class VideoFolderDataset(Dataset):
    """root/<class>/{video file | frame-folder | image} layout.

    Each class folder may hold video files, frame-folders (a subfolder of
    image frames counts as one clip) and/or loose images (each image counts
    as a single-frame clip).
    """

    def __init__(self, root: str, num_frames: int = 8, img_size: int = 64):
        self.root = self._resolve_root(root)
        self.num_frames, self.img_size = num_frames, img_size
        self.classes = sorted(d for d in os.listdir(self.root)
                              if os.path.isdir(os.path.join(self.root, d))
                              and self._is_class_dir(os.path.join(self.root, d)))
        self.samples = []  # (kind, path, label)
        for ci, cls in enumerate(self.classes):
            cdir = os.path.join(self.root, cls)
            for entry in sorted(os.listdir(cdir)):
                p = os.path.join(cdir, entry)
                if os.path.isfile(p):
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in VIDEO_EXTS:
                        self.samples.append(("video", p, ci))
                    elif ext in IMAGE_EXTS:
                        self.samples.append(("image", p, ci))
                elif os.path.isdir(p) and find_files(p, IMAGE_EXTS):
                    self.samples.append(("frames", p, ci))
        if len(self.classes) < 2:
            raise ValueError(f"Need at least 2 class folders, found: {self.classes}.")
        if not self.samples:
            raise ValueError("No video files, frame-folders or images found in the class folders.")

    @classmethod
    def _is_class_dir(cls, path: str) -> bool:
        try:
            entries = os.listdir(path)
        except OSError:
            return False
        for e in entries:
            p = os.path.join(path, e)
            if os.path.isfile(p) and os.path.splitext(e)[1].lower() in VIDEO_EXTS | IMAGE_EXTS:
                return True
            if os.path.isdir(p) and find_files(p, IMAGE_EXTS):
                return True
        return False

    @classmethod
    def _resolve_root(cls, root: str) -> str:
        """Find the directory whose children are class folders (unwrap one level)."""
        def n_class_dirs(d: str) -> int:
            try:
                return sum(1 for e in os.listdir(d)
                           if os.path.isdir(os.path.join(d, e))
                           and cls._is_class_dir(os.path.join(d, e)))
            except OSError:
                return 0

        if n_class_dirs(root) >= 2:
            return root
        try:
            children = sorted(e for e in os.listdir(root)
                              if os.path.isdir(os.path.join(root, e)))
        except OSError:
            children = []
        for c in children:
            if n_class_dirs(os.path.join(root, c)) >= 2:
                return os.path.join(root, c)
        raise ValueError(
            "Could not find class folders with videos/frames. Expected one folder "
            "per class holding video files, frame-folders or images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        _, path, label = self.samples[i]
        frames = load_clip_for_inference(path, self.num_frames, self.img_size)
        return frames, torch.tensor(label, dtype=torch.long)


def load_clip_for_inference(path: str, num_frames: int, img_size: int) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path) or ext not in VIDEO_EXTS:
        if os.path.isdir(path):
            return read_frame_dir(path, num_frames, img_size)
        # Single image counts as a 1-frame clip (repeated).
        img = load_image_rgb(path, img_size)
        return torch.stack([pil_to_tensor(img)] * num_frames)
    return read_video_frames(path, num_frames, img_size)


# -------------------------------------------------------------- multimodal
def load_caption_data(data_dir: str, image_col: str, caption_col: str):
    files = find_files(data_dir, {".csv"})
    if not files:
        raise ValueError("No .csv file found in the uploaded data.")
    fields, rows = read_csv_rows(files[0])
    if image_col not in fields or caption_col not in fields:
        raise ValueError(f"CSV must contain '{image_col}' and '{caption_col}' columns "
                         f"(found: {', '.join(fields)}).")
    images = find_files(data_dir, IMAGE_EXTS)
    by_name = {}
    for p in images:
        by_name.setdefault(os.path.basename(p), p)
        by_name.setdefault(os.path.relpath(p, data_dir), p)
    pairs = []
    for r in rows:
        key = (r[image_col] or "").strip()
        cap = (r[caption_col] or "").strip()
        if key in by_name and cap:
            pairs.append((by_name[key], cap))
    if not pairs:
        raise ValueError("No CSV rows matched images in the upload. The image column "
                         "must list file names present in the zip.")
    return pairs


class CaptionDataset(Dataset):
    def __init__(self, pairs, tokenizer, img_size: int = 64, max_len: int = 32):
        self.pairs = pairs
        self.tok = tokenizer
        self.img_size, self.max_len = img_size, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, cap = self.pairs[i]
        return (pil_to_tensor(load_image_rgb(path, self.img_size)),
                torch.tensor(self.tok.encode_caption(cap, self.max_len), dtype=torch.long))


# ----------------------------------------------------------------- tabular
class TabularData:
    """Loads a CSV into normalized features + classification/regression targets."""

    def __init__(self, data_dir: str, target_col: str, problem_type: str = "auto"):
        files = _as_file_list(data_dir, {".csv"}, ".csv")
        if not files:
            raise ValueError("No .csv file found in the uploaded data.")
        fields, rows = read_csv_rows(files[0])
        if target_col not in fields:
            raise ValueError(f"Target column '{target_col}' not found (found: {', '.join(fields)}).")
        feature_names = [c for c in fields if c != target_col]
        if not feature_names:
            raise ValueError("CSV has no feature columns besides the target.")
        X, raw_y = [], []
        for r in rows:
            try:
                X.append([float(r[c]) if str(r[c]).strip() != "" else float("nan") for c in feature_names])
            except (ValueError, TypeError):
                raise ValueError(f"Non-numeric value found in feature columns {feature_names}. "
                                 "Tabular training needs numeric features.")
            raw_y.append(str(r[target_col]).strip())
        if not X:
            raise ValueError("CSV contains no data rows.")
        X = np.array(X, dtype=np.float64)
        means = np.nanmean(X, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        stds = np.nanstd(X, axis=0)
        stds = np.where(stds < 1e-9, 1.0, stds)
        self.feature_names = feature_names
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)
        self.X = torch.tensor(((X - means) / stds).astype(np.float32))

        uniq = sorted(set(raw_y))
        if problem_type == "auto":
            # Few distinct values -> classification (string or numeric labels);
            # many distinct values -> regression (must be numeric).
            problem_type = "classification" if len(uniq) <= 50 else "regression"
        self.problem_type = problem_type
        if problem_type == "classification":
            self.labels = uniq
            lmap = {l: i for i, l in enumerate(uniq)}
            self.y = torch.tensor([lmap[v] for v in raw_y], dtype=torch.long)
            self.out_dim = len(uniq)
        else:
            try:
                y = np.array([float(v) for v in raw_y], dtype=np.float32)
            except ValueError:
                raise ValueError("Regression targets must be numeric.")
            self.labels = []
            self.y = torch.tensor(y, dtype=torch.float32)
            self.out_dim = 1

    def dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(self.X, self.y)

    def meta(self) -> dict:
        return {"feature_names": self.feature_names,
                "means": self.means.tolist(), "stds": self.stds.tolist(),
                "problem_type": self.problem_type, "labels": self.labels}
