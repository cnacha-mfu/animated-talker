"""Tests for model_manager module."""

from pathlib import Path

from animated_talker.model_manager import (
    HF_MODELS,
    get_model_dir,
    get_model_path,
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
    for key in HF_MODELS:
        path = get_model_path(key)
        assert isinstance(path, Path)
        assert path.name  # Has a filename


def test_verify_models_empty_dir(tmp_path):
    """Test verification with no models downloaded."""
    status = verify_models(model_dir=tmp_path)
    assert all(v is False for v in status.values())
    assert len(status) == len(HF_MODELS)


def test_sadtalker_model_keys():
    """Test that SadTalker model keys are present."""
    expected_keys = {
        "sadtalker_checkpoint",
        "mapping_full",
        "mapping_still",
        "face_recon",
        "audio2exp",
        "audio2pose",
    }
    assert set(HF_MODELS.keys()) == expected_keys


def test_model_paths_in_sadtalker_subdir():
    """Test that all model paths are under the sadtalker subdirectory."""
    for key in HF_MODELS:
        path = get_model_path(key)
        assert "sadtalker" in str(path)
