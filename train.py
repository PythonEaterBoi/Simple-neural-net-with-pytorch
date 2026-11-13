"""CLI training: same trainers as the web UI, run from the terminal.

Examples:
    python train.py --modality text --task generation --data demo_data/text --epochs 5
    python train.py --modality image --task classification --data demo_data/image_classes
    python train.py --modality tabular --task predict --data demo_data/tabular --target-col label
    python train.py --list
"""

import argparse
import os
import sys
import time

from config import MODALITIES, default_config
from training.trainer import TRAIN_FUNCS


class PrintCallback:
    def __init__(self):
        self.cancel = False

    def log(self, msg: str):
        print(msg, flush=True)

    def metric(self, row: dict):
        pass

    def cancelled(self) -> bool:
        return self.cancel


def list_tasks():
    for mod, m in MODALITIES.items():
        print(f"{m['icon']} {mod} - {m['description']}")
        for task, t in m["tasks"].items():
            print(f"    {task:15s} {t['label']} - {t['description']}")
            print(f"                     data: {t['data_help']}")


def main():
    p = argparse.ArgumentParser(description="Train a neural net (any modality).")
    p.add_argument("--list", action="store_true", help="List modalities/tasks and exit.")
    p.add_argument("--modality", help="text | image | video | multimodal | tabular")
    p.add_argument("--task", help="generation | classification | captioning | predict")
    p.add_argument("--data", help="Training data: a file (.txt/.csv) or a folder (or extracted zip).")
    p.add_argument("--output", default=None, help="Checkpoint path (.pt).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # Task-specific extras (only applied when relevant):
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--image-col", default=None)
    p.add_argument("--caption-col", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    args = p.parse_args()

    if args.list:
        list_tasks()
        return

    if not args.modality or not args.task or not args.data:
        p.error("--modality, --task and --data are required (or use --list).")
    if (args.modality, args.task) not in TRAIN_FUNCS:
        sys.exit(f"Unknown combination: {args.modality}/{args.task}. Use --list.")
    if not os.path.exists(args.data):
        sys.exit(f"Data path not found: {args.data}")

    config = default_config(args.modality, args.task)
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "device": args.device, "text_col": args.text_col, "label_col": args.label_col,
                 "image_col": args.image_col, "caption_col": args.caption_col,
                 "target_col": args.target_col, "img_size": args.img_size,
                 "num_frames": args.num_frames, "block_size": args.block_size,
                 "max_len": args.max_len, "n_embd": args.n_embd,
                 "n_layer": args.n_layer, "n_head": args.n_head, "timesteps": args.timesteps}
    for k, v in overrides.items():
        if v is not None and k in config or (v is not None and k == "device"):
            config[k] = v
    if "device" not in config:
        config["device"] = args.device

    os.makedirs("checkpoints", exist_ok=True)
    out = args.output or os.path.join(
        "checkpoints", f"cli_{args.modality}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    print(f"Training {args.modality}/{args.task} -> {out}")
    print(f"Config: {config}")
    TRAIN_FUNCS[(args.modality, args.task)](args.data, config, PrintCallback(), out)
    print(f"Saved checkpoint: {out}")


if __name__ == "__main__":
    main()
