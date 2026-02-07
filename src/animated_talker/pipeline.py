"""Main pipeline orchestrator — ties all stages together."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from animated_talker.engine import TalkingHeadEngine
from animated_talker.input_handler import load_audio, load_image
from animated_talker.model_manager import ensure_models, get_model_dir
from animated_talker.video_encoder import encode_video

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ("sadtalker", "musetalk")


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to cuda/mps/cpu based on availability."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _create_engine(
    engine_name: str,
    model_dir: Path,
    device: str,
    use_float16: bool,
    **engine_kwargs,
) -> TalkingHeadEngine:
    """Instantiate the requested engine."""
    if engine_name == "sadtalker":
        from animated_talker.sadtalker_engine import SadTalkerEngine

        return SadTalkerEngine(
            model_dir=model_dir,
            device=device,
            use_float16=use_float16,
            size=engine_kwargs.get("size", 256),
            preprocess=engine_kwargs.get("preprocess", "crop"),
            pose_style=engine_kwargs.get("pose_style", 0),
            expression_scale=engine_kwargs.get("expression_scale", 1.0),
            still_mode=engine_kwargs.get("still_mode", False),
            use_blink=engine_kwargs.get("use_blink", True),
        )

    if engine_name == "musetalk":
        from animated_talker.musetalk_engine import MuseTalkEngine

        return MuseTalkEngine(
            model_dir=model_dir,
            device=device,
            use_float16=use_float16,
            bbox_shift=engine_kwargs.get("bbox_shift", 0),
        )

    raise ValueError(
        f"Unknown engine: {engine_name!r}. Supported: {SUPPORTED_ENGINES}"
    )


def generate(
    image: str | Path,
    audio: str | Path,
    output: str | Path,
    engine: str = "sadtalker",
    fps: int = 25,
    # --- SadTalker-specific ---
    pose_style: int = 0,
    expression_scale: float = 1.0,
    still_mode: bool = False,
    use_blink: bool = True,
    size: int = 256,
    preprocess: str = "crop",
    # --- MuseTalk-specific ---
    bbox_shift: int = 0,
    # --- Post-processing ---
    face_restore: bool = False,
    upscale: bool = False,
    upscale_factor: int = 4,
    restore_fidelity: float = 0.7,
    # --- Common ---
    batch_size: int = 8,
    use_float16: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
    crf: int = 18,
) -> Path:
    """Generate a talking-head video with lip-sync and head motion.

    Takes a portrait image and an audio file, generates animated frames
    using the selected engine, and encodes them into a video.

    Args:
        image: Path to portrait image (JPEG/PNG).
        audio: Path to audio file (MP3/WAV).
        output: Path for output video (MP4).
        engine: Animation engine — "sadtalker" (default, head motion) or
            "musetalk" (lip-sync only, faster).
        fps: Video frame rate (default 25).
        pose_style: [SadTalker] Head motion style 0-45 (default 0).
        expression_scale: [SadTalker] Expression intensity 0.0-3.0 (default 1.0).
        still_mode: [SadTalker] Reduce background motion (default False).
        use_blink: [SadTalker] Enable eye blinks (default True).
        size: [SadTalker] Face region size, 256 or 512 (default 256).
        preprocess: [SadTalker] Face preprocess mode — "crop", "resize",
            "full", "extcrop", "extfull" (default "crop").
        bbox_shift: [MuseTalk] Jaw boundary adjustment in pixels (default 0).
        face_restore: Apply CodeFormer face restoration (requires postprocess extras).
        upscale: Apply Real-ESRGAN upscaling (requires postprocess extras).
        upscale_factor: Upscale multiplier (default 4).
        restore_fidelity: CodeFormer fidelity 0.0-1.0 (default 0.7).
        batch_size: Frames per inference batch (default 8).
        use_float16: Use FP16 inference (default True, auto-disabled on CPU).
        device: PyTorch device — "auto" (default), "cuda", "mps", or "cpu".
        model_dir: Custom model directory (default ~/.cache/animated_talker/models).
        crf: Video quality 0-51 (default 18, visually lossless).

    Returns:
        Path to the output video file.

    Example:
        >>> from animated_talker import generate
        >>> generate(image="avatar.png", audio="speech.mp3", output="result.mp4")
    """
    device = _resolve_device(device)

    # FP16 not supported on CPU
    if device == "cpu" and use_float16:
        logger.info("FP16 not supported on CPU, falling back to FP32")
        use_float16 = False

    logger.info("Using engine=%s, device=%s (float16=%s)", engine, device, use_float16)

    output = Path(output)
    model_path = Path(model_dir) if model_dir else get_model_dir()

    # Step 1: Ensure model weights are downloaded
    logger.info("Checking model weights...")
    ensure_models(model_dir=model_path, engine=engine)

    # Step 2: Load inputs
    logger.info("Loading inputs...")
    img = load_image(image)
    audio_wav, duration = load_audio(audio)

    logger.info("Image: %s, Audio: %.2fs", Path(image).name, duration)

    # Step 3: Create engine and generate frames
    engine_obj = _create_engine(
        engine_name=engine,
        model_dir=model_path,
        device=device,
        use_float16=use_float16,
        # SadTalker params
        pose_style=pose_style,
        expression_scale=expression_scale,
        still_mode=still_mode,
        use_blink=use_blink,
        size=size,
        preprocess=preprocess,
        # MuseTalk params
        bbox_shift=bbox_shift,
    )
    frames = engine_obj.generate_frames(
        image=img,
        audio=audio_wav,
        fps=fps,
        batch_size=batch_size,
    )

    # Step 4: Optional post-processing
    if face_restore or upscale:
        from animated_talker.postprocess import restore_frames

        frames = restore_frames(
            frames,
            face_restore=face_restore,
            upscale=upscale,
            device=device,
            restore_fidelity=restore_fidelity,
            upscale_factor=upscale_factor,
        )

    # Step 5: Encode video with audio
    logger.info("Encoding video...")
    result = encode_video(
        frames=frames,
        audio_path=audio,
        output_path=output,
        fps=fps,
        crf=crf,
    )

    logger.info("Done! Output: %s", result)
    return result
