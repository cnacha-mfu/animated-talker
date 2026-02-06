"""Tests for the pipeline module."""

import pytest


def test_generate_missing_image():
    """Test that missing image raises FileNotFoundError."""
    from animated_talker.pipeline import generate

    with pytest.raises(FileNotFoundError):
        generate(
            image="/nonexistent/avatar.png",
            audio="/nonexistent/audio.mp3",
            output="/tmp/output.mp4",
        )


def test_generate_import():
    """Test that generate is importable from the package root."""
    from animated_talker import generate

    assert callable(generate)
