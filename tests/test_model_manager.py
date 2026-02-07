"""Tests for model_manager module."""

from pathlib import Path

from animated_talker.model_manager import (
    DITTO_MODELS,
    get_model_dir,
    get_model_path,
    get_ditto_repo_dir,
    verify_models,
)


def test_get_model_dir_default():
    """Test default model directory."""
    model_dir = get_model_dir()
    assert isinstance(model_dir, Path)
    assert "animated_talker" in str(model_dir)


def test_get_model_dir_from_env(monkeypatch, tmp_path):
    """Test model directory from environment variable."""
    custom_dir = tmp_path / "custom_models"
    monkeypatch.setenv("ANIMATED_TALKER_MODEL_DIR", str(custom_dir))
    assert get_model_dir() == custom_dir


def test_get_model_path():
    """Test model path construction."""
    for key in DITTO_MODELS:
        path = get_model_path(key)
        assert isinstance(path, Path)
        assert path.name  # Has a filename


def test_get_ditto_repo_dir():
    """Test Ditto repo directory path."""
    repo_dir = get_ditto_repo_dir()
    assert isinstance(repo_dir, Path)
    assert "ditto-talkinghead" in str(repo_dir)


def test_verify_models_empty_dir(tmp_path):
    """Test verification with no models downloaded."""
    status = verify_models(model_dir=tmp_path)
    assert all(v is False for v in status.values())
    # +1 for ditto_repo check
    assert len(status) == len(DITTO_MODELS) + 1
