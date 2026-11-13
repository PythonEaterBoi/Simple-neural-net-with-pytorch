"""Generate small synthetic demo datasets (no torch needed).

Creates demo_data/ with ready-to-train examples for every modality:
    text/shakespeare.txt        text generation corpus
    text_classes.csv            text classification (pos/neg)
    image_classes/<cls>/*.png   image classification (+ generation)
    image_classes.zip           zipped version for web upload
    captions/images/*.png       captioning images
    captions/captions.csv       captioning pairs
    captions.zip                zipped version for web upload
    video_classes/<cls>/...     frame-folder clips for video classification
    video_classes.zip           zipped version for web upload
    tabular/shapes.csv          tabular classification
"""

import csv
import os
import random
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_data")
random.seed(7)
np.random.seed(7)

SHAPES = ["circle", "square", "triangle"]
COLORS = {"circle": (235, 90, 90), "square": (90, 160, 235), "triangle": (90, 200, 130)}


def draw_shape(shape: str, size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), (24, 26, 34))
    d = ImageDraw.Draw(img)
    m, col = size // 4, COLORS[shape]
    jx, jy = random.randint(-4, 4), random.randint(-4, 4)
    if shape == "circle":
        d.ellipse([m + jx, m + jy, size - m + jx, size - m + jy], fill=col)
    elif shape == "square":
        d.rectangle([m + jx, m + jy, size - m + jx, size - m + jy], fill=col)
    else:
        d.polygon([(size // 2 + jx, m + jy), (m + jx, size - m + jy),
                   (size - m + jx, size - m + jy)], fill=col)
    arr = np.asarray(img).astype(np.int16) + np.random.randint(-12, 13, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def zip_dir(src: str, dst: str):
    if os.path.exists(dst):
        os.remove(dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, files in os.walk(src):
            for f in files:
                full = os.path.join(dirpath, f)
                zf.write(full, os.path.relpath(full, src))


def make_text():
    d = os.path.join(ROOT, "text")
    os.makedirs(d, exist_ok=True)
    paras = [
        "the little robot dreamed of electric sheep beneath a copper moon. ",
        "every morning it polished its gears and sang to the rising sun. ",
        "the old forest whispered secrets to those who walked softly. ",
        "stars are the lanterns of travellers who never came home. ",
        "a river of light runs through the valley of quiet machines. ",
    ]
    corpus = "".join(random.choice(paras) for _ in range(400))
    with open(os.path.join(d, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"  text corpus: {len(corpus):,} chars")


def make_text_classes():
    rows = []
    pos = ["i love this", "great and wonderful", "amazing work", "so happy today",
           "fantastic brilliant result", "truly excellent stuff"]
    neg = ["i hate this", "terrible and awful", "horrible work", "so sad today",
           "dreadful poor result", "truly bad stuff"]
    for _ in range(60):
        s = random.choice(pos)
        rows.append([s + " " + random.choice(pos), "pos"])
    for _ in range(60):
        s = random.choice(neg)
        rows.append([s + " " + random.choice(neg), "neg"])
    random.shuffle(rows)
    with open(os.path.join(ROOT, "text_classes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"  text_classes.csv: {len(rows)} rows")


def make_images():
    d = os.path.join(ROOT, "image_classes")
    shutil.rmtree(d, ignore_errors=True)
    for shape in SHAPES:
        cdir = os.path.join(d, shape)
        os.makedirs(cdir, exist_ok=True)
        for i in range(30):
            draw_shape(shape).save(os.path.join(cdir, f"{shape}_{i:02d}.png"))
    zip_dir(d, os.path.join(ROOT, "image_classes.zip"))
    print("  image_classes: 3 classes x 30 images (+zip)")


def make_captions():
    d = os.path.join(ROOT, "captions", "images")
    shutil.rmtree(os.path.join(ROOT, "captions"), ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    rows = []
    templates = ["a {c} {s} on a dark background", "a dark background with a {c} {s}",
                 "there is a {c} {s} in the image"]
    cnames = {"circle": "red", "square": "blue", "triangle": "green"}
    for i in range(90):
        shape = SHAPES[i % 3]
        name = f"img_{i:03d}.png"
        draw_shape(shape).save(os.path.join(d, name))
        rows.append([name, random.choice(templates).format(c=cnames[shape], s=shape)])
    with open(os.path.join(ROOT, "captions", "captions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption"])
        w.writerows(rows)
    zip_dir(os.path.join(ROOT, "captions"), os.path.join(ROOT, "captions.zip"))
    print("  captions: 90 pairs (+zip)")


def make_videos():
    d = os.path.join(ROOT, "video_classes")
    shutil.rmtree(d, ignore_errors=True)
    # "bounce": circle moves vertically; "still": static square.
    for cls in ["bounce", "still"]:
        cdir = os.path.join(d, cls)
        os.makedirs(cdir, exist_ok=True)
        for v in range(6):
            vdir = os.path.join(cdir, f"clip_{v:02d}")
            os.makedirs(vdir, exist_ok=True)
            for t in range(8):
                img = Image.new("RGB", (64, 64), (24, 26, 34))
                dr = ImageDraw.Draw(img)
                if cls == "bounce":
                    y = 8 + int(40 * abs(np.sin(t / 7 * np.pi)))
                    dr.ellipse([22, y - 10, 42, y + 10], fill=COLORS["circle"])
                else:
                    dr.rectangle([22, 22, 42, 42], fill=COLORS["square"])
                img.save(os.path.join(vdir, f"frame_{t:02d}.png"))
    zip_dir(d, os.path.join(ROOT, "video_classes.zip"))
    print("  video_classes: 2 classes x 6 clips (+zip)")


def make_tabular():
    d = os.path.join(ROOT, "tabular")
    os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(300):
        shape = SHAPES[i % 3]
        # Features loosely correlated with the class + noise.
        base = {"circle": (1.0, 5.0), "square": (5.0, 1.0), "triangle": (5.0, 5.0)}[shape]
        x = base[0] + random.gauss(0, 0.8)
        y = base[1] + random.gauss(0, 0.8)
        rows.append([round(x, 3), round(y, 3), round(x * y, 3), shape])
    with open(os.path.join(d, "shapes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "area", "label"])
        w.writerows(rows)
    print(f"  tabular/shapes.csv: {len(rows)} rows")


def main():
    os.makedirs(ROOT, exist_ok=True)
    print(f"Generating demo data in {ROOT}/")
    make_text()
    make_text_classes()
    make_images()
    make_captions()
    make_videos()
    make_tabular()
    print("Done.")


if __name__ == "__main__":
    main()
