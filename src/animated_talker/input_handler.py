"""Load and preprocess input image and audio for the lip-sync pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import librosa
import numpy as np

logger = logging.getLogger(__name__)


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


def detect_face(
    image: np.ndarray,
    bbox_shift: int = 0,
    extra_margin: int = 10,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Detect face and return (cropped_256x256, bbox_coords).

    Uses face_alignment for 68-landmark detection, consistent with MuseTalk.
    bbox_shift adjusts the jaw boundary (positive = lower crop, more chin).
    extra_margin adds pixels below the jaw.
    """
    import face_alignment

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
    )

    # Convert BGR to RGB for face_alignment
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks_list = fa.get_landmarks(rgb)

    if landmarks_list is None or len(landmarks_list) == 0:
        raise ValueError(
            "No face detected in image. Ensure the image contains a clear, "
            "front-facing face."
        )

    landmarks = landmarks_list[0]  # Use first detected face

    # Compute bounding box from landmarks (jaw + forehead region)
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x1 = int(np.min(x_coords))
    y1 = int(np.min(y_coords))
    x2 = int(np.max(x_coords))
    y2 = int(np.max(y_coords))

    # Add margin around the face
    h, w = image.shape[:2]
    face_h = y2 - y1
    face_w = x2 - x1
    margin_x = int(face_w * 0.2)
    margin_y_top = int(face_h * 0.4)  # More margin above for forehead
    margin_y_bottom = int(face_h * 0.1) + extra_margin + bbox_shift

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y_top)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y_bottom)

    bbox = (x1, y1, x2, y2)

    # Crop and resize to 256x256
    crop = image[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    logger.info("Face detected — bbox=(%d,%d,%d,%d)", *bbox)
    return crop_resized, bbox


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
