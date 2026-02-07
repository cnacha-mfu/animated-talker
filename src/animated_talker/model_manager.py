"""Download and manage Ditto-TalkingHead model weights and repository."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "animated_talker" / "models"
DITTO_REPO_URL = "https://github.com/antgroup/ditto-talkinghead.git"
DITTO_HF_REPO = "digital-avatar/ditto-talkinghead"

# Ditto PyTorch model files to download from HuggingFace
# (HF filename relative to repo root, local subfolder)
DITTO_MODELS: dict[str, tuple[str, str]] = {
    "appearance_extractor": (
        "ditto_pytorch/appearance_extractor.pth",
        "ditto_pytorch",
    ),
    "decoder": (
        "ditto_pytorch/decoder.pth",
        "ditto_pytorch",
    ),
    "lmdm": (
        "ditto_pytorch/lmdm_v0.4_hubert.pth",
        "ditto_pytorch",
    ),
    "motion_extractor": (
        "ditto_pytorch/motion_extractor.pth",
        "ditto_pytorch",
    ),
    "stitch_network": (
        "ditto_pytorch/stitch_network.pth",
        "ditto_pytorch",
    ),
    "warp_network": (
        "ditto_pytorch/warp_network.pth",
        "ditto_pytorch",
    ),
    # ONNX models for audio and face detection
    "hubert_onnx": (
        "ditto_onnx/hubert_streaming_fix_kv.onnx",
        "ditto_onnx",
    ),
    "face_det_10g": (
        "ditto_onnx/det_10g.onnx",
        "ditto_onnx",
    ),
    "face_2d106det": (
        "ditto_onnx/2d106det.onnx",
        "ditto_onnx",
    ),
    "face_landmarker": (
        "ditto_onnx/face_landmarker.task",
        "ditto_onnx",
    ),
    "landmark203": (
        "ditto_onnx/landmark203.onnx",
        "ditto_onnx",
    ),
    # Config
    "cfg_pytorch": (
        "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        "ditto_cfg",
    ),
}


def get_model_dir() -> Path:
    """Return the root model directory, respecting ANIMATED_TALKER_MODEL_DIR env var."""
    env_dir = os.environ.get("ANIMATED_TALKER_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_MODEL_DIR


def get_ditto_repo_dir(model_dir: Path | None = None) -> Path:
    """Return the path where the Ditto repo is/should be cloned."""
    if model_dir is None:
        model_dir = get_model_dir()
    return model_dir / "ditto-talkinghead"


def get_model_path(model_key: str, model_dir: Path | None = None) -> Path:
    """Return the expected local path for a model file."""
    if model_dir is None:
        model_dir = get_model_dir()
    hf_filename, subfolder = DITTO_MODELS[model_key]
    basename = Path(hf_filename).name
    return model_dir / subfolder / basename


def clone_ditto_repo(model_dir: Path | None = None) -> Path:
    """Clone the Ditto-TalkingHead repository if not already present."""
    if model_dir is None:
        model_dir = get_model_dir()

    repo_dir = get_ditto_repo_dir(model_dir)
    if repo_dir.exists() and (repo_dir / "inference.py").exists():
        logger.debug("Ditto repo already cloned at %s", repo_dir)
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning Ditto-TalkingHead repo to %s ...", repo_dir)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", DITTO_REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone Ditto repo:\n{result.stderr}\n"
            f"Ensure git is installed and you have internet access."
        )

    logger.info("Ditto repo cloned successfully")
    return repo_dir


def download_model(model_key: str, model_dir: Path | None = None) -> Path:
    """Download a single model file if not already present. Returns local path."""
    from huggingface_hub import hf_hub_download

    if model_dir is None:
        model_dir = get_model_dir()

    local_path = get_model_path(model_key, model_dir)
    if local_path.exists():
        logger.debug("Model %s already exists at %s", model_key, local_path)
        return local_path

    hf_filename, subfolder = DITTO_MODELS[model_key]
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from %s ...", model_key, DITTO_HF_REPO)
    downloaded = hf_hub_download(
        repo_id=DITTO_HF_REPO,
        filename=hf_filename,
        local_dir=model_dir / "_hf_download",
        local_dir_use_symlinks=False,
    )

    # Move to our expected location
    downloaded_path = Path(downloaded)
    if downloaded_path != local_path and downloaded_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(downloaded_path), str(local_path))

    logger.info("Downloaded %s to %s", model_key, local_path)
    return local_path


def ensure_models(
    model_dir: Path | None = None,
    include_repo: bool = True,
) -> dict[str, Path]:
    """Download all required models and clone the Ditto repo.

    Returns dict of model_key -> local_path.
    """
    if model_dir is None:
        model_dir = get_model_dir()

    paths: dict[str, Path] = {}

    # Clone the Ditto repo (needed for inference code)
    if include_repo:
        repo_dir = clone_ditto_repo(model_dir)
        paths["ditto_repo"] = repo_dir

    # Download model weights
    for key in DITTO_MODELS:
        paths[key] = download_model(key, model_dir)

    return paths


def verify_models(model_dir: Path | None = None) -> dict[str, bool]:
    """Check which models are present locally."""
    if model_dir is None:
        model_dir = get_model_dir()

    status: dict[str, bool] = {}

    # Check repo
    repo_dir = get_ditto_repo_dir(model_dir)
    status["ditto_repo"] = repo_dir.exists() and (repo_dir / "inference.py").exists()

    # Check model files
    for key in DITTO_MODELS:
        path = get_model_path(key, model_dir)
        status[key] = path.exists()

    return status
