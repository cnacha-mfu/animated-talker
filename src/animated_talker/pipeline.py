"""Main pipeline orchestrator — ties all stages together."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from animated_talker.model_manager import setup_all
from animated_talker.musetalk_engine import MuseTalkEngine

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to cuda/mps/cpu based on availability."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(
    image: str | Path,
    audio: str | Path,
    output: str | Path,
    fps: int = 25,
    bbox_shift: int = 0,
    batch_size: int = 8,
    use_float16: bool = True,
    device: str = "auto",
    extra_margin: int = 10,
    face_restore: bool = False,
    upscale: bool = False,
    upscale_factor: int = 4,
    restore_fidelity: float = 0.7,
    model_dir: str | Path | None = None,
    crf: int = 18,
) -> Path:
    """Generate a lip-synced talking avatar video.

    Takes a portrait image and an audio file, generates lip-synced frames
    using MuseTalk v1.5, and encodes them into a video with the original audio.

    Args:
        image: Path to portrait image (JPEG/PNG)
        audio: Path to audio file (MP3/WAV)
        output: Path for output video (MP4)
        fps: Video frame rate (default 25)
        bbox_shift: Jaw boundary adjustment (positive = more chin, default 0)
        batch_size: Frames per inference batch (default 8)
        use_float16: Use FP16 for inference (default True, auto-disabled on CPU)
        device: PyTorch device — "auto" (default), "cuda", "mps", or "cpu"
        extra_margin: Extra margin below jaw in pixels (default 10)
        face_restore: Apply CodeFormer face restoration (not yet supported)
        upscale: Apply Real-ESRGAN upscaling (not yet supported)
        upscale_factor: Upscale multiplier (default 4)
        restore_fidelity: CodeFormer fidelity weight 0.0-1.0 (default 0.7)
        model_dir: Custom model directory (currently unused, reserved)
        crf: Video quality 0-51 (default 18, visually lossless)

    Returns:
        Path to the output video file

    Example:
        >>> from animated_talker import generate
        >>> generate(image="avatar.png", audio="speech.mp3", output="result.mp4")
    """
    device = _resolve_device(device)

    # FP16 not supported on CPU
    if device == "cpu" and use_float16:
        logger.info("FP16 not supported on CPU, falling back to FP32")
        use_float16 = False

    logger.info("Using device: %s (float16=%s)", device, use_float16)

    # Ensure MuseTalk repo is cloned and models are downloaded
    logger.info("Setting up MuseTalk...")
    setup_all(install_deps=False)

    # Create engine and generate
    logger.info("Loading models...")
    engine = MuseTalkEngine(device=device, use_float16=use_float16)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = engine.generate(
        image_path=str(image),
        audio_path=str(audio),
        output_path=str(output),
        fps=fps,
        batch_size=batch_size,
        bbox_shift=bbox_shift,
        extra_margin=extra_margin,
    )

    if face_restore or upscale:
        logger.warning(
            "Post-processing (face_restore, upscale) is not yet supported "
            "with the MuseTalk wrapper. Skipping."
        )

    logger.info("Done! Output: %s", result)
    return Path(result)
