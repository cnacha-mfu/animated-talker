"""Validate input image and audio files for the lip-sync pipeline.

Ditto-TalkingHead handles face detection and audio processing internally,
so this module only validates that input files exist and are readable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import librosa
import numpy as np

logger = logging.getLogger(__name__)


def validate_image(image_path: str | Path) -> None:
    """Validate that the image file exists and is readable."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    logger.info("Validated image %s — shape %s", path.name, img.shape)


def validate_audio(audio_path: str | Path) -> None:
    """Validate that the audio file exists and is readable."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Quick load check
    waveform, sr = librosa.load(str(path), sr=16000, mono=True, duration=1.0)
    if len(waveform) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    logger.info("Validated audio %s", path.name)


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image as BGR numpy array. Raises if file missing or unreadable."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    logger.info("Loaded image %s — shape %s", path.name, img.shape)
    return img


def load_audio(audio_path: str | Path, sr: int = 16000) -> tuple[np.ndarray, float]:
    """Load audio file and resample to target rate.

    Returns (waveform, duration_seconds). Waveform is float32 mono at target sr.
    Supports MP3, WAV, FLAC, and other formats via librosa/soundfile.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, _ = librosa.load(str(path), sr=sr, mono=True)
    duration = len(waveform) / sr

    logger.info(
        "Loaded audio %s — duration=%.2fs, sr=%d, samples=%d",
        path.name,
        duration,
        sr,
        len(waveform),
    )
    return waveform, duration
