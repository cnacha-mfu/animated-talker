"""FFmpeg-based video encoding with audio muxing."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def check_ffmpeg() -> str:
    """Check that FFmpeg is available and return its path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found on PATH. Install FFmpeg >= 5.0: "
            "https://ffmpeg.org/download.html"
        )
    return ffmpeg


def encode_video(
    frames: list[np.ndarray],
    audio_path: str | Path,
    output_path: str | Path,
    fps: int = 25,
    crf: int = 18,
) -> Path:
    """Encode frames to H.264 MP4 and mux with audio.

    Args:
        frames: List of BGR numpy arrays (all same shape)
        audio_path: Path to audio file (MP3/WAV)
        output_path: Path for output MP4 file
        fps: Frame rate
        crf: Constant Rate Factor (0-51, lower = better quality, 18 = visually lossless)

    Returns:
        Path to the output video file
    """
    ffmpeg = check_ffmpeg()
    output_path = Path(output_path)
    audio_path = Path(audio_path)

    if len(frames) == 0:
        raise ValueError("No frames to encode")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    logger.info(
        "Encoding %d frames (%dx%d) at %d FPS, CRF=%d",
        len(frames), w, h, fps, crf,
    )

    # Use pipe to feed frames directly to FFmpeg (avoids writing temp PNGs)
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",  # Video from stdin
        "-i", str(audio_path),  # Audio input
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",  # Match duration to shortest stream
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.debug("FFmpeg command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Write frames to FFmpeg stdin
    for frame in frames:
        proc.stdin.write(frame.tobytes())

    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg encoding failed (exit {proc.returncode}):\n{error_msg}")

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video saved to %s (%.1f MB)", output_path, file_size)
    return output_path


def encode_video_from_dir(
    frame_dir: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    fps: int = 25,
    crf: int = 18,
    pattern: str = "%06d.png",
) -> Path:
    """Encode video from a directory of numbered frame images.

    Alternative to pipe-based encoding for debugging or large frame counts.
    """
    ffmpeg = check_ffmpeg()
    output_path = Path(output_path)
    frame_dir = Path(frame_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / pattern),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed:\n{result.stderr}")

    return output_path
