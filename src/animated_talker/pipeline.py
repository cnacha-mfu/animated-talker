"""Main pipeline orchestrator — ties all stages together."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from animated_talker.input_handler import validate_image, validate_audio
from animated_talker.model_manager import ensure_models, get_model_dir
from animated_talker.ditto_engine import DittoEngine

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
    emotion: int = 4,
    sampling_timesteps: int = 50,
    fade_in: int = -1,
    fade_out: int = -1,
    face_restore: bool = False,
    upscale: bool = False,
    upscale_factor: int = 4,
    restore_fidelity: float = 0.7,
    use_float16: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> Path:
    """Generate a lip-synced talking avatar video.

    Takes a portrait image and an audio file, generates lip-synced frames
    using Ditto-TalkingHead, and encodes them into a video with the original
    audio.

    Args:
        image: Path to portrait image (JPEG/PNG)
        audio: Path to audio file (MP3/WAV)
        output: Path for output video (MP4)
        fps: Video frame rate (default 25)
        emotion: Emotion style index 0-8 (default 4 = neutral)
        sampling_timesteps: Diffusion denoising steps (default 50, lower = faster)
        fade_in: Number of frames to fade in at start (-1 = disabled)
        fade_out: Number of frames to fade out at end (-1 = disabled)
        face_restore: Apply CodeFormer face restoration (requires postprocess extras)
        upscale: Apply Real-ESRGAN upscaling (requires postprocess extras)
        upscale_factor: Upscale multiplier (default 4)
        restore_fidelity: CodeFormer fidelity weight 0.0-1.0 (default 0.7)
        use_float16: Use FP16 for inference (default True, saves VRAM)
        device: PyTorch device — "auto" (default), "cuda", "mps", or "cpu"
        model_dir: Custom model directory (default ~/.cache/animated_talker/models)

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

    # Validate inputs exist
    validate_image(image)
    validate_audio(audio)

    output = Path(output)
    model_path = Path(model_dir) if model_dir else get_model_dir()

    # Step 1: Ensure model weights and Ditto repo are available
    logger.info("Checking model weights...")
    ensure_models(model_dir=model_path)

    # Step 2: Generate video using Ditto engine
    logger.info("Initializing Ditto engine...")
    engine = DittoEngine(
        model_dir=model_path,
        device=device,
        use_float16=use_float16,
    )

    # If post-processing requested, generate to temp path first
    if face_restore or upscale:
        import tempfile
        temp_output = Path(tempfile.mktemp(suffix=".mp4"))
        target_output = temp_output
    else:
        target_output = output

    result = engine.generate_video(
        image_path=image,
        audio_path=audio,
        output_path=target_output,
        fps=fps,
        emotion=emotion,
        sampling_timesteps=sampling_timesteps,
        fade_in=fade_in,
        fade_out=fade_out,
    )

    # Step 3: Optional post-processing
    if face_restore or upscale:
        from animated_talker.postprocess import postprocess_video

        postprocess_video(
            input_path=temp_output,
            output_path=output,
            audio_path=audio,
            face_restore=face_restore,
            upscale=upscale,
            device=device,
            restore_fidelity=restore_fidelity,
            upscale_factor=upscale_factor,
            fps=fps,
        )
        # Clean up temp file
        try:
            temp_output.unlink()
        except OSError:
            pass
        result = output

    logger.info("Done! Output: %s", result)
    return result
