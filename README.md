# 🧠 OmniTrain — Multi-Modal Neural Net Training Interface

Train **and prompt** PyTorch neural nets across **all modalities** — text, image, video, multimodal (vision+language) and tabular — from one web interface or the CLI. (Speech — TTS/STT — is intentionally out of scope.)

![modalities](https://img.shields.io/badge/modalities-text%E2%80%A2image%E2%80%A2video%E2%80%A2multimodal%E2%80%A2tabular-blue)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.0-red)

## What you can do

| Modality | Task | Train on | Prompt with |
|---|---|---|---|
| 📝 Text | **Generation** (char-GPT) | `.txt` / `.zip` corpus | text → generated continuation |
| 📝 Text | **Classification** (Transformer) | CSV (`text`, `label`) | text → predicted label |
| 🖼️ Image | **Classification** (CNN) | zip of class folders | image → predicted label |
| 🖼️ Image | **Generation** (diffusion UNet) | zip of class folders | class name → generated images |
| 🎬 Video | **Classification** (frame-CNN + pooling) | zip of class folders (video files or frame-folders) | video → predicted label |
| 🔗 Multimodal | **Captioning** (CNN + Transformer) | zip with images + CSV (`image`, `caption`) | image → caption |
| 📊 Tabular | **Prediction** (MLP, class. or regr.) | CSV with numeric features + target | feature values → prediction |

Features:

- 🖥️ **Web UI** — pick a modality, upload data, tune hyperparameters, watch live loss curves + logs, then prompt the trained model.
- ⌨️ **CLI training** — `train.py` runs the exact same trainers headlessly.
- 💾 **Checkpoints** — one `.pt` file per model (architecture + weights + vocab/labels), managed in the UI.
- 🧪 **Demo data generator** — synthetic datasets for every modality so you can try everything in minutes.
- 🖥️/⚡ **CPU or CUDA** — auto-detected.

## Quickstart

```bash
# 1. Install (CPU torch recommended unless you have an NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Generate demo datasets for every modality
python scripts/make_demo_data.py

# 3. Start the web interface
uvicorn app:app --host 0.0.0.0 --port 8000
# → open http://localhost:8000
```

Then in the **Train** tab: pick e.g. `Image → Classification`, upload `demo_data/image_classes.zip`, press **Start training**, watch the loss curve — and in the **Prompt** tab select the finished checkpoint and upload an image to classify it.

### CLI training

```bash
python train.py --list   # show all modalities/tasks
python train.py --modality text --task generation --data demo_data/text --epochs 5
python train.py --modality image --task classification --data demo_data/image_classes --epochs 5
python train.py --modality video --task classification --data demo_data/video_classes --epochs 5
python train.py --modality multimodal --task captioning --data demo_data/captions --epochs 5
python train.py --modality tabular --task predict --data demo_data/tabular --target-col label
```

### Prompt from Python

```python
from inference.predict import predict
print(predict("abc123_text_generation.pt", prompt_text="the robot "))
print(predict("def456_image_classification.pt", media_path="cat.jpg"))
```

## Data formats

- **Text generation**: one or more `.txt` files (plain upload or inside a zip; all concatenated).
- **Text classification**: CSV with configurable text/label columns (default `text`, `label`).
- **Image classification/generation**: zip with one folder per class: `cats/*.jpg`, `dogs/*.jpg`. (Generation resizes to 32×32.)
- **Video classification**: zip with one folder per class; each class folder holds video files (`.mp4/.avi/.mov/.mkv`) **or** frame-folders (a subfolder of frames = one clip — no codec needed).
- **Captioning**: zip containing images plus a CSV mapping `image` → `caption`.
- **Tabular**: CSV with numeric feature columns + a target column. Classification vs regression is auto-detected (≤50 distinct target values → classification) or forced via `problem_type`.

## API reference

| Method & path | Description |
|---|---|
| `GET /api/health` | server + torch/CUDA status |
| `GET /api/modalities` | modality/task/hyperparameter registry (drives the UI) |
| `POST /api/train` | multipart: `modality`, `task`, `config` (JSON), `files[]` → `{job_id}` |
| `GET /api/jobs` / `GET /api/jobs/{id}` | job list / detail (status, metrics) |
| `GET /api/jobs/{id}/logs?since=n` | incremental logs + metrics (polling) |
| `POST /api/jobs/{id}/cancel` · `DELETE /api/jobs/{id}` | cancel / delete a job |
| `GET /api/checkpoints` · `DELETE /api/checkpoints/{name}` | checkpoint management |
| `POST /api/prompt` | multipart: `checkpoint`, `prompt_text`, `values` (JSON), `options` (JSON), `file` → typed result (`text`/`label`/`images`/`number`) |

## Project structure

```
├── app.py                  # FastAPI server + REST API
├── train.py                # CLI training (same trainers as the UI)
├── config.py               # modality/task/hyperparameter registry
├── models/                 # PyTorch architectures (text/image/video/multimodal/tabular)
├── training/
│   ├── datasets.py         # loaders for every data format
│   ├── trainer.py          # training loops + checkpointing
│   └── jobs.py             # background job manager
├── inference/predict.py    # checkpoint loading + prompting
├── scripts/make_demo_data.py
├── static/                 # web UI (index.html, app.js, styles.css)
├── checkpoints/ uploads/ predictions/   # runtime artifacts (git-ignored)
└── demo_data/              # generated by scripts/make_demo_data.py (git-ignored)
```

## Notes & limits

- Models are deliberately **small** (they train on CPU in minutes) — perfect for learning, prototyping and small datasets. Scale up the hyperparameters (and data) for harder problems.
- Video files need `opencv-python-headless`; frame-folder clips work without it.
- Speech modalities (TTS/STT) are out of scope by design.
