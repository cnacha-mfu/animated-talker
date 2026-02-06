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

    # Patch MuseTalk for PyTorch 2.6+ compatibility
    _patch_musetalk_torch_load(musetalk_dir)

    return musetalk_dir


def _patch_musetalk_torch_load(musetalk_dir: Path) -> None:
    """Patch torch.load() calls in MuseTalk to use weights_only=False.

    PyTorch 2.6 changed the default to weights_only=True, which breaks
    loading legacy .tar format checkpoints (resnet18, bisenet).
    """
    files_to_patch = [
        musetalk_dir / "musetalk" / "utils" / "face_parsing" / "resnet.py",
        musetalk_dir / "musetalk" / "utils" / "face_parsing" / "model.py",
    ]
    for fpath in files_to_patch:
        if not fpath.exists():
            continue
        text = fpath.read_text()
        if "weights_only" not in text and "torch.load(" in text:
            patched = text.replace(
                "torch.load(model_path)",
                "torch.load(model_path, weights_only=False)",
            )
            if patched != text:
                fpath.write_text(patched)
                logger.info("Patched %s for PyTorch 2.6+ compatibility", fpath.name)


def ensure_musetalk_deps() -> None:
    """Install MuseTalk Python dependencies needed for inference.

    We install selectively instead of using MuseTalk's requirements.txt,
    which pins incompatible versions (tensorflow==2.12, numpy==1.23.5)
    that don't work on Python 3.12+.
    """
    # Only the packages MuseTalk needs for inference (not training/UI)
    inference_deps = [
        "diffusers",
        "accelerate",
        "transformers",
        "opencv-python",
        "librosa",
        "soundfile",
        "einops",
        "omegaconf",
        "ffmpeg-python",
        "moviepy",
        "imageio[ffmpeg]",
    ]

    logger.info("Installing MuseTalk inference dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + inference_deps,
        check=True,
    )

    # Install mmlab packages (required for DWPose landmark detection).
    # We avoid openmim which breaks setuptools/pkg_resources on Python 3.12.
    # Use pre-built mmcv wheels from OpenMMLab to avoid slow compilation.
    logger.info("Installing mmlab packages (mmengine, mmcv, mmdet, mmpose)...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mmengine"],
        check=True,
    )

    # Install mmcv with pre-built wheel (avoids 10+ min source compilation)
    _install_mmcv_prebuilt()

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mmdet>=3.1.0", "mmpose>=1.1.0"],
        check=True,
    )


def _install_mmcv_prebuilt() -> None:
    """Install mmcv using pre-built wheels from OpenMMLab's index.

    Auto-detects PyTorch and CUDA versions to find the right wheel.
    Falls back to source build if no pre-built wheel is available.
    """
    import torch

    torch_version = torch.__version__.split("+")[0]  # e.g. "2.9.0"
    # Shorten to major.minor for the wheel index URL
    tv_parts = torch_version.split(".")
    torch_short = f"{tv_parts[0]}.{tv_parts[1]}.0"  # e.g. "2.9.0"

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda.replace(".", "")  # e.g. "126"
        cu_tag = f"cu{cuda_version}"
    else:
        cu_tag = "cpu"

    wheel_index = (
        f"https://download.openmmlab.com/mmcv/dist/{cu_tag}/torch{torch_short}/index.html"
    )

    logger.info("Installing mmcv from pre-built wheel (%s, torch %s)...", cu_tag, torch_short)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "mmcv>=2.0.0", "-f", wheel_index],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        logger.info("mmcv installed from pre-built wheel")
        return

    # If no pre-built wheel, try without --find-links (source build)
    logger.warning(
        "No pre-built mmcv wheel found for %s/torch%s, falling back to pip install "
        "(this may take 10+ minutes to compile)...",
        cu_tag, torch_short,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mmcv>=2.0.0"],
        check=True,
    )


# Model files: (repo_id, hf_filename, local_path relative to models_dir)
MODEL_FILES: list[tuple[str, str, str]] = [
    # MuseTalk v1.5 UNet
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "musetalkV15/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "musetalkV15/unet.pth"),
    # SD VAE
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae/config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "sd-vae/diffusion_pytorch_model.bin"),
    # Whisper tiny
    ("openai/whisper-tiny", "config.json", "whisper/config.json"),
    ("openai/whisper-tiny", "pytorch_model.bin", "whisper/pytorch_model.bin"),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper/preprocessor_config.json"),
    # DWPose
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "dwpose/dw-ll_ucoco_384.pth"),
    # Face parsing
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "face-parse-bisent/79999_iter.pth"),
    ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth", "face-parse-bisent/resnet18-5c106cde.pth"),
]


def ensure_models() -> Path:
    """Download all required model weights into MuseTalk's models/ dir."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, hf_filename, local_path in MODEL_FILES:
        target_file = models_dir / local_path
        if target_file.exists():
            logger.debug("Already exists: %s", target_file)
            continue

        target_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s/%s ...", repo_id, hf_filename)

        # Download to HF cache, then copy to target location
        cached = hf_hub_download(repo_id=repo_id, filename=hf_filename)
        shutil.copy2(cached, target_file)
        logger.info("  → %s", target_file)

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
    # Always apply patches (idempotent) in case repo was cloned before patch existed
    _patch_musetalk_torch_load(musetalk_dir)
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
