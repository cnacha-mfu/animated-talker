"""Download and manage model weights from HuggingFace Hub."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "animated_talker" / "models"

# (repo_id, filename, subfolder in local cache)
HF_MODELS: dict[str, tuple[str, str, str]] = {
    "musetalk_config": (
        "TMElyralab/MuseTalk",
        "musetalkV15/musetalk.json",
        "musetalk",
    ),
    "musetalk_unet": (
        "TMElyralab/MuseTalk",
        "musetalkV15/unet.pth",
        "musetalk",
    ),
    "sd_vae_config": (
        "stabilityai/sd-vae-ft-mse",
        "config.json",
        "sd-vae-ft-mse",
    ),
    "sd_vae_weights": (
        "stabilityai/sd-vae-ft-mse",
        "diffusion_pytorch_model.safetensors",
        "sd-vae-ft-mse",
    ),
    "whisper_tiny": (
        "openai/whisper-tiny",
        "model.safetensors",
        "whisper-tiny",
    ),
    "whisper_config": (
        "openai/whisper-tiny",
        "config.json",
        "whisper-tiny",
    ),
    "whisper_preprocessor": (
        "openai/whisper-tiny",
        "preprocessor_config.json",
        "whisper-tiny",
    ),
    "face_parse_bisenet": (
        "ManyOtherFunctions/face-parse-bisent",
        "79999_iter.pth",
        "face-parse-bisenet",
    ),
    "face_parse_resnet": (
        "ManyOtherFunctions/face-parse-bisent",
        "resnet18-5c106cde.pth",
        "face-parse-bisenet",
    ),
}


def get_model_dir() -> Path:
    """Return the root model directory, respecting ANIMATED_TALKER_MODEL_DIR env var."""
    env_dir = os.environ.get("ANIMATED_TALKER_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_MODEL_DIR


def get_model_path(model_key: str, model_dir: Path | None = None) -> Path:
    """Return the expected local path for a model file."""
    if model_dir is None:
        model_dir = get_model_dir()
    _, filename, subfolder = HF_MODELS[model_key]
    basename = Path(filename).name
    return model_dir / subfolder / basename


def download_model(model_key: str, model_dir: Path | None = None) -> Path:
    """Download a single model file if not already present. Returns local path."""
    if model_dir is None:
        model_dir = get_model_dir()

    local_path = get_model_path(model_key, model_dir)
    if local_path.exists():
        logger.debug("Model %s already exists at %s", model_key, local_path)
        return local_path

    repo_id, filename, subfolder = HF_MODELS[model_key]
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from %s ...", model_key, repo_id)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir / subfolder,
        local_dir_use_symlinks=False,
    )

    # hf_hub_download may place the file in a nested path matching the repo structure.
    # Move it to our expected flat location if needed.
    downloaded_path = Path(downloaded)
    if downloaded_path != local_path and downloaded_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.rename(local_path)

    logger.info("Downloaded %s to %s", model_key, local_path)
    return local_path


def ensure_models(
    model_dir: Path | None = None,
    include_postprocess: bool = False,
) -> dict[str, Path]:
    """Download all required models. Returns dict of model_key -> local_path."""
    if model_dir is None:
        model_dir = get_model_dir()

    paths = {}
    for key in HF_MODELS:
        paths[key] = download_model(key, model_dir)

    return paths


def verify_models(model_dir: Path | None = None) -> dict[str, bool]:
    """Check which models are present locally."""
    if model_dir is None:
        model_dir = get_model_dir()

    status = {}
    for key in HF_MODELS:
        path = get_model_path(key, model_dir)
        status[key] = path.exists()
    return status
