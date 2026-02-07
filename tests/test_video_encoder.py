"""Tests for video_encoder module."""

import shutil

import numpy as np
import pytest


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg not installed",
)
def test_encode_video_creates_file(tmp_path):
    """Test that encode_video produces an MP4 file."""
    import soundfile as sf

    from animated_talker.video_encoder import encode_video

    # Create test frames (10 frames, 256x256)
    frames = [
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        for _ in range(10)
    ]

    # Create test audio
    sr = 16000
    duration = 10 / 25  # Match 10 frames at 25 fps
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), audio, sr)

    output_path = tmp_path / "output.mp4"
    result = encode_video(frames, str(audio_path), str(output_path), fps=25)

    assert result.exists()
    assert result.stat().st_size > 0


def test_check_ffmpeg():
    """Test FFmpeg detection."""
    from animated_talker.video_encoder import check_ffmpeg

    if shutil.which("ffmpeg"):
        path = check_ffmpeg()
        assert path is not None
    else:
        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            check_ffmpeg()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg not installed",
)
def test_encode_video_empty_frames(tmp_path):
    """Test that empty frame list raises ValueError."""
    from animated_talker.video_encoder import encode_video

    with pytest.raises(ValueError, match="No frames"):
        encode_video([], str(tmp_path / "audio.wav"), str(tmp_path / "out.mp4"))
