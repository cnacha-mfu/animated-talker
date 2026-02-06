"""Tests for input_handler module."""

import numpy as np
import pytest


def test_load_image_missing_file():
    from animated_talker.input_handler import load_image

    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.png")


def test_load_audio_missing_file():
    from animated_talker.input_handler import load_audio

    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/path/audio.mp3")


def test_load_image_returns_ndarray(tmp_path):
    """Test loading a valid image file."""
    import cv2

    # Create a simple test image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 50:200] = 255  # White square
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    from animated_talker.input_handler import load_image

    result = load_image(str(img_path))
    assert isinstance(result, np.ndarray)
    assert result.shape == (256, 256, 3)


def test_load_audio_returns_waveform(tmp_path):
    """Test loading a valid WAV file."""
    import soundfile as sf

    # Create a simple test audio (1 second of silence)
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), audio, sr)

    from animated_talker.input_handler import load_audio

    waveform, duration = load_audio(str(audio_path))
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert abs(duration - 1.0) < 0.1
