"""Setup and manage MuseTalk dependency and model weights."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DEFAULT_INSTALL_DIR = Path.home() / ".cache" / "animated_talker"


def get_install_dir() -> Path:
    env_dir = os.environ.get("ANIMATED_TALKER_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_INSTALL_DIR


def get_musetalk_dir() -> Path:
    return get_install_dir() / "MuseTalk"


def get_models_dir() -> Path:
    return get_musetalk_dir() / "models"


def ensure_musetalk_repo() -> Path:
    """Clone MuseTalk repo if not present. Returns path to repo."""
    musetalk_dir = get_musetalk_dir()
    if (musetalk_dir / "musetalk").is_dir():
        logger.info("MuseTalk repo found at %s", musetalk_dir)
        return musetalk_dir

    logger.info("Cloning MuseTalk repo...")
    musetalk_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/TMElyralab/MuseTalk.git", str(musetalk_dir)],
        check=True,
    )
    logger.info("MuseTalk cloned to %s", musetalk_dir)
    return musetalk_dir


def ensure_musetalk_deps() -> None:
    """Install MuseTalk Python dependencies."""
    musetalk_dir = get_musetalk_dir()
    req_file = musetalk_dir / "requirements.txt"
    if not req_file.exists():
        logger.warning("requirements.txt not found in MuseTalk repo")
        return

    logger.info("Installing MuseTalk dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        check=True,
    )

    # Install mmlab packages (required for DWPose landmark detection)
    logger.info("Installing mmlab packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-U", "openmim"],
        check=True,
    )
    for pkg in ["mmengine", "mmcv==2.0.1", "mmdet==3.1.0", "mmpose==1.1.0"]:
        subprocess.run(
            [sys.executable, "-m", "mim", "install", "-q", pkg],
            check=False,  # Don't fail if already installed
        )


# Model files to download: (repo_id, filename)
HF_DOWNLOADS: list[tuple[str, str]] = [
    # MuseTalk v1.5 UNet
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth"),
    # SD VAE
    ("stabilityai/sd-vae-ft-mse", "config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin"),
    # Whisper tiny
    ("openai/whisper-tiny", "config.json"),
    ("openai/whisper-tiny", "pytorch_model.bin"),
    ("openai/whisper-tiny", "preprocessor_config.json"),
    # DWPose
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth"),
    # Face parsing
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth"),
    ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth"),
]

# Map HuggingFace paths to local model subdirectory
LOCAL_DIR_MAP: dict[str, str] = {
    "TMElyralab/MuseTalk": "",  # Files already have musetalkV15/ prefix
    "stabilityai/sd-vae-ft-mse": "sd-vae",
    "openai/whisper-tiny": "whisper",
    "yzd-v/DWPose": "dwpose",
    "ManyOtherFunctions/face-parse-bisent": "face-parse-bisent",
}


def ensure_models() -> Path:
    """Download all required model weights into MuseTalk's models/ dir."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, filename in HF_DOWNLOADS:
        local_subdir = LOCAL_DIR_MAP[repo_id]
        if local_subdir:
            target_dir = models_dir / local_subdir
        else:
            target_dir = models_dir

        target_file = target_dir / Path(filename).name
        if target_file.exists():
            logger.debug("Already exists: %s", target_file)
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s/%s ...", repo_id, filename)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

        # hf_hub_download may nest files — move to expected flat location
        downloaded_path = Path(downloaded)
        if downloaded_path != target_file and downloaded_path.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_path), str(target_file))

    logger.info("All models downloaded to %s", models_dir)
    return models_dir


def ensure_musetalk_on_path() -> None:
    """Add MuseTalk repo to sys.path so we can import its modules."""
    musetalk_dir = str(get_musetalk_dir())
    if musetalk_dir not in sys.path:
        sys.path.insert(0, musetalk_dir)


def setup_all(install_deps: bool = False) -> Path:
    """Full setup: clone repo, optionally install deps, download models.

    Returns path to MuseTalk directory.
    """
    musetalk_dir = ensure_musetalk_repo()
    if install_deps:
        ensure_musetalk_deps()
    ensure_models()
    ensure_musetalk_on_path()
    return musetalk_dir


def verify_setup() -> dict[str, bool]:
    """Check what's installed and available."""
    musetalk_dir = get_musetalk_dir()
    models_dir = get_models_dir()

    status = {
        "musetalk_repo": (musetalk_dir / "musetalk").is_dir(),
        "unet_weights": (models_dir / "musetalkV15" / "unet.pth").exists(),
        "unet_config": (models_dir / "musetalkV15" / "musetalk.json").exists(),
        "vae_config": (models_dir / "sd-vae" / "config.json").exists(),
        "vae_weights": (models_dir / "sd-vae" / "diffusion_pytorch_model.bin").exists(),
        "whisper_model": (models_dir / "whisper" / "pytorch_model.bin").exists(),
        "dwpose": (models_dir / "dwpose" / "dw-ll_ucoco_384.pth").exists(),
        "face_parse": (models_dir / "face-parse-bisent" / "79999_iter.pth").exists(),
    }
    return status
