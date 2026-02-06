"""Main pipeline orchestrator — ties all stages together."""

from __future__ import annotations

import logging
from pathlib import Path

from animated_talker.input_handler import detect_face, load_audio, load_image
from animated_talker.model_manager import ensure_models, get_model_dir
from animated_talker.musetalk_engine import MuseTalkEngine
from animated_talker.video_encoder import encode_video

logger = logging.getLogger(__name__)


def generate(
    image: str | Path,
    audio: str | Path,
    output: str | Path,
    fps: int = 25,
    bbox_shift: int = 0,
    face_restore: bool = False,
    upscale: bool = False,
    upscale_factor: int = 4,
    restore_fidelity: float = 0.7,
    batch_size: int = 8,
    use_float16: bool = True,
    device: str = "cuda",
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
        face_restore: Apply CodeFormer face restoration (requires postprocess extras)
        upscale: Apply Real-ESRGAN 4x upscaling (requires postprocess extras)
        upscale_factor: Upscale multiplier (default 4)
        restore_fidelity: CodeFormer fidelity weight 0.0-1.0 (default 0.7)
        batch_size: Frames per inference batch (default 8)
        use_float16: Use FP16 for inference (default True, saves VRAM)
        device: PyTorch device (default "cuda")
        model_dir: Custom model directory (default ~/.cache/animated_talker/models)
        crf: Video quality 0-51 (default 18, visually lossless)

    Returns:
        Path to the output video file

    Example:
        >>> from animated_talker import generate
        >>> generate(image="avatar.png", audio="speech.mp3", output="result.mp4")
    """
    output = Path(output)
    model_path = Path(model_dir) if model_dir else get_model_dir()

    # Step 1: Ensure model weights are downloaded
    logger.info("Checking model weights...")
    ensure_models(model_dir=model_path)

    # Step 2: Load and preprocess inputs
    logger.info("Loading inputs...")
    img = load_image(image)
    face_crop, bbox = detect_face(img, bbox_shift=bbox_shift)
    audio_wav, duration = load_audio(audio)

    logger.info("Image: %s, Audio: %.2fs", Path(image).name, duration)

    # Step 3: Generate lip-synced frames
    engine = MuseTalkEngine(
        model_dir=model_path,
        device=device,
        use_float16=use_float16,
    )
    frames = engine.generate_frames(
        face_crop=face_crop,
        original_frame=img,
        bbox=bbox,
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
