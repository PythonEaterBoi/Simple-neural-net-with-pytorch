"""Central registry of supported modalities, tasks, hyperparameters and prompt specs.

The web UI builds its forms dynamically from this registry, so adding a new
modality/task here (plus a trainer + predictor) automatically exposes it in
the interface.
"""

MODALITIES = {
    "text": {
        "label": "Text",
        "icon": "📝",
        "description": "Language models and text classifiers (PyTorch Transformers).",
        "tasks": {
            "generation": {
                "label": "Text Generation",
                "description": "Train a small character-level GPT on your own corpus, then prompt it to generate text.",
                "data_help": "Upload a .txt file (or a .zip containing .txt files). All text is concatenated for training.",
                "accept": ".txt,.zip",
                "hyperparams": [
                    {"name": "block_size", "label": "Context length", "type": "int", "default": 64, "min": 8, "max": 256, "help": "Characters of context the model sees."},
                    {"name": "n_embd", "label": "Embedding size", "type": "int", "default": 128, "min": 32, "max": 512, "help": "Model width."},
                    {"name": "n_layer", "label": "Layers", "type": "int", "default": 4, "min": 1, "max": 12},
                    {"name": "n_head", "label": "Attention heads", "type": "int", "default": 4, "min": 1, "max": 16},
                    {"name": "dropout", "label": "Dropout", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 10, "min": 1, "max": 500},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 1, "max": 512},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.0003, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {
                    "kind": "text",
                    "placeholder": "Once upon a time...",
                    "options": [
                        {"name": "max_tokens", "label": "Max new tokens", "type": "int", "default": 120, "min": 1, "max": 1000},
                        {"name": "temperature", "label": "Temperature", "type": "float", "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05},
                        {"name": "top_k", "label": "Top-k", "type": "int", "default": 40, "min": 1, "max": 200},
                    ],
                },
            },
            "classification": {
                "label": "Text Classification",
                "description": "Train a Transformer classifier on labelled text (sentiment, topics, intent...).",
                "data_help": "Upload a CSV with a text column and a label column.",
                "accept": ".csv,.zip",
                "hyperparams": [
                    {"name": "text_col", "label": "Text column", "type": "text", "default": "text"},
                    {"name": "label_col", "label": "Label column", "type": "text", "default": "label"},
                    {"name": "max_len", "label": "Max tokens", "type": "int", "default": 64, "min": 8, "max": 512},
                    {"name": "n_embd", "label": "Embedding size", "type": "int", "default": 64, "min": 16, "max": 512},
                    {"name": "n_layer", "label": "Layers", "type": "int", "default": 2, "min": 1, "max": 8},
                    {"name": "n_head", "label": "Attention heads", "type": "int", "default": 4, "min": 1, "max": 16},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 15, "min": 1, "max": 500},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 32, "min": 1, "max": 512},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {"kind": "text", "placeholder": "Type text to classify..."},
            },
        },
    },
    "image": {
        "label": "Image",
        "icon": "🖼️",
        "description": "Image classifiers and class-conditional image generators.",
        "tasks": {
            "classification": {
                "label": "Image Classification",
                "description": "Train a CNN to classify images, then prompt it with new images.",
                "data_help": "Upload a .zip with one folder per class, e.g. cats/*.jpg, dogs/*.jpg.",
                "accept": ".zip",
                "hyperparams": [
                    {"name": "img_size", "label": "Image size", "type": "select", "default": 64, "options": [32, 64, 96, 128]},
                    {"name": "base_channels", "label": "Base channels", "type": "int", "default": 32, "min": 8, "max": 128},
                    {"name": "augment", "label": "Augmentation (flip/crop)", "type": "checkbox", "default": True},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 15, "min": 1, "max": 500},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 32, "min": 1, "max": 256},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {"kind": "image"},
            },
            "generation": {
                "label": "Image Generation (Diffusion)",
                "description": "Train a tiny class-conditional diffusion model, then prompt it with a class to generate images.",
                "data_help": "Upload a .zip with one folder per class (images are resized to 32x32).",
                "accept": ".zip",
                "hyperparams": [
                    {"name": "timesteps", "label": "Diffusion timesteps", "type": "int", "default": 150, "min": 20, "max": 500},
                    {"name": "base_channels", "label": "Base channels", "type": "int", "default": 32, "min": 8, "max": 128},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 30, "min": 1, "max": 1000},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 1, "max": 256},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.0002, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {
                    "kind": "class",
                    "options": [
                        {"name": "n_images", "label": "Images to generate", "type": "int", "default": 2, "min": 1, "max": 4},
                    ],
                },
            },
        },
    },
    "video": {
        "label": "Video",
        "icon": "🎬",
        "description": "Video classifiers (per-frame CNN + temporal pooling).",
        "tasks": {
            "classification": {
                "label": "Video Classification",
                "description": "Train on labelled clips (action / scene recognition), then prompt with new videos.",
                "data_help": "Upload a .zip with one folder per class containing video files (.mp4/.avi/.mov/.mkv) or frame-folders (a subfolder of .jpg/.png frames counts as one clip).",
                "accept": ".zip",
                "hyperparams": [
                    {"name": "num_frames", "label": "Frames per clip", "type": "int", "default": 8, "min": 2, "max": 32},
                    {"name": "img_size", "label": "Frame size", "type": "select", "default": 64, "options": [32, 64, 96, 128]},
                    {"name": "emb_dim", "label": "Frame embedding", "type": "int", "default": 128, "min": 32, "max": 512},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 15, "min": 1, "max": 500},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 8, "min": 1, "max": 64},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {"kind": "video"},
            },
        },
    },
    "multimodal": {
        "label": "Multimodal",
        "icon": "🔗",
        "description": "Models that combine vision and language.",
        "tasks": {
            "captioning": {
                "label": "Image Captioning",
                "description": "Train a CNN + Transformer to describe images, then prompt it with new images.",
                "data_help": "Upload a .zip containing an images folder and a CSV with image,caption columns.",
                "accept": ".zip",
                "hyperparams": [
                    {"name": "image_col", "label": "Image column", "type": "text", "default": "image"},
                    {"name": "caption_col", "label": "Caption column", "type": "text", "default": "caption"},
                    {"name": "img_size", "label": "Image size", "type": "select", "default": 64, "options": [32, 64, 96, 128]},
                    {"name": "max_len", "label": "Max caption tokens", "type": "int", "default": 24, "min": 4, "max": 128},
                    {"name": "n_embd", "label": "Embedding size", "type": "int", "default": 128, "min": 32, "max": 512},
                    {"name": "n_layer", "label": "Layers", "type": "int", "default": 2, "min": 1, "max": 8},
                    {"name": "n_head", "label": "Attention heads", "type": "int", "default": 4, "min": 1, "max": 16},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 20, "min": 1, "max": 500},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 32, "min": 1, "max": 256},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.01, "step": 0.0001},
                ],
                "prompt": {"kind": "image"},
            },
        },
    },
    "tabular": {
        "label": "Tabular",
        "icon": "📊",
        "description": "MLP models for CSV / spreadsheet data (classification or regression).",
        "tasks": {
            "predict": {
                "label": "Tabular Prediction",
                "description": "Train an MLP on CSV features, then prompt it with feature values.",
                "data_help": "Upload a CSV with numeric feature columns and a target column.",
                "accept": ".csv,.zip",
                "hyperparams": [
                    {"name": "target_col", "label": "Target column", "type": "text", "default": "label"},
                    {"name": "problem_type", "label": "Problem type", "type": "select", "default": "auto", "options": ["auto", "classification", "regression"]},
                    {"name": "hidden_size", "label": "Hidden size", "type": "int", "default": 64, "min": 8, "max": 512},
                    {"name": "n_layers", "label": "Hidden layers", "type": "int", "default": 2, "min": 1, "max": 8},
                    {"name": "dropout", "label": "Dropout", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
                    {"name": "epochs", "label": "Epochs", "type": "int", "default": 50, "min": 1, "max": 2000},
                    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 32, "min": 1, "max": 1024},
                    {"name": "lr", "label": "Learning rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.05, "step": 0.0001},
                ],
                "prompt": {"kind": "form"},
            },
        },
    },
}


def get_task(modality: str, task: str) -> dict:
    """Return the registry entry for a modality/task pair (raises KeyError)."""
    return MODALITIES[modality]["tasks"][task]


def default_config(modality: str, task: str) -> dict:
    """Return the default hyperparameter values for a modality/task pair."""
    return {h["name"]: h["default"] for h in get_task(modality, task)["hyperparams"]}
