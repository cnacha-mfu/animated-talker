"""Download and manage model weights from HuggingFace Hub."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "animated_talker" / "models"

# MuseTalk models: (repo_id, filename, subfolder in local cache)
HF_MODELS_MUSETALK: dict[str, tuple[str, str, str]] = {
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

# SadTalker models (new-format V0.0.2 safetensors):
# (repo_id, filename, subfolder in local cache, repo_type)
HF_MODELS_SADTALKER: dict[str, tuple[str, str, str, str | None]] = {
    "sadtalker_v002_256": (
        "vinthony/SadTalker-V002",
        "SadTalker_V0.0.2_256.safetensors",
        "sadtalker",
        None,  # model repo (default)
    ),
    "sadtalker_v002_512": (
        "vinthony/SadTalker-V002",
        "SadTalker_V0.0.2_512.safetensors",
        "sadtalker",
        None,
    ),
    "sadtalker_mapping": (
        "vinthony/SadTalker-V002",
        "mapping_00109-model.pth.tar",
        "sadtalker",
        None,
    ),
}

# Backward-compatible alias used by existing tests
HF_MODELS = HF_MODELS_MUSETALK


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

    # Search in both registries
    registry = _find_registry(model_key)
    entry = registry[model_key]
    filename = entry[1]
    subfolder = entry[2]
    basename = Path(filename).name
    return model_dir / subfolder / basename


def _find_registry(model_key: str) -> dict:
    """Find which registry contains the given model_key."""
    if model_key in HF_MODELS_MUSETALK:
        return HF_MODELS_MUSETALK
    if model_key in HF_MODELS_SADTALKER:
        return HF_MODELS_SADTALKER
    raise KeyError(f"Unknown model key: {model_key}")


def download_model(model_key: str, model_dir: Path | None = None) -> Path:
    """Download a single model file if not already present. Returns local path."""
    if model_dir is None:
        model_dir = get_model_dir()

    local_path = get_model_path(model_key, model_dir)
    if local_path.exists():
        logger.debug("Model %s already exists at %s", model_key, local_path)
        return local_path

    registry = _find_registry(model_key)
    entry = registry[model_key]
    repo_id = entry[0]
    filename = entry[1]
    subfolder = entry[2]
    repo_type = entry[3] if len(entry) > 3 else None

    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from %s ...", model_key, repo_id)
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": model_dir / subfolder,
        "local_dir_use_symlinks": False,
    }
    if repo_type:
        kwargs["repo_type"] = repo_type

    downloaded = hf_hub_download(**kwargs)

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
    engine: str = "sadtalker",
) -> dict[str, Path]:
    """Download all required models for the given engine.

    Args:
        model_dir: Override model directory.
        engine: Engine name — "sadtalker" (default) or "musetalk".

    Returns:
        Dict of model_key -> local_path.
    """
    if model_dir is None:
        model_dir = get_model_dir()

    if engine == "sadtalker":
        registry = HF_MODELS_SADTALKER
    elif engine == "musetalk":
        registry = HF_MODELS_MUSETALK
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'sadtalker' or 'musetalk'.")

    paths = {}
    for key in registry:
        paths[key] = download_model(key, model_dir)

    return paths


def verify_models(
    model_dir: Path | None = None,
    engine: str | None = None,
) -> dict[str, bool]:
    """Check which models are present locally.

    If engine is None, checks both MuseTalk and SadTalker models.
    """
    if model_dir is None:
        model_dir = get_model_dir()

    if engine == "musetalk":
        registry = HF_MODELS_MUSETALK
    elif engine == "sadtalker":
        registry = HF_MODELS_SADTALKER
    else:
        # Check all models from both registries
        registry = {**HF_MODELS_MUSETALK, **HF_MODELS_SADTALKER}

    status = {}
    for key in registry:
        path = get_model_path(key, model_dir)
        status[key] = path.exists()
    return status
