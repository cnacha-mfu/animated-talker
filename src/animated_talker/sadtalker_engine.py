"""SadTalker talking-head animation engine.

Generates realistic talking-head videos with natural head motion and
lip-sync using 3D Morphable Models (3DMM) and a face-vid2vid renderer.

Reference: https://github.com/OpenTalker/SadTalker (CVPR 2023)
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch

from animated_talker.engine import TalkingHeadEngine
from animated_talker.model_manager import get_model_dir

logger = logging.getLogger(__name__)

# Pinned commit for reproducibility
_SADTALKER_GITHUB_ZIP = (
    "https://github.com/OpenTalker/SadTalker/archive/refs/heads/main.zip"
)
_SADTALKER_EXTRACT_PREFIX = "SadTalker-main"


def _ensure_sadtalker_source(model_dir: Path) -> Path:
    """Download and extract SadTalker source if not already present.

    Returns the root directory of the extracted SadTalker source.
    """
    src_dir = model_dir / "sadtalker_src"
    marker = src_dir / ".downloaded"

    if marker.exists():
        root = src_dir / _SADTALKER_EXTRACT_PREFIX
        if root.is_dir():
            return root

    src_dir.mkdir(parents=True, exist_ok=True)
    zip_path = src_dir / "sadtalker.zip"

    logger.info("Downloading SadTalker source from GitHub...")
    urllib.request.urlretrieve(_SADTALKER_GITHUB_ZIP, str(zip_path))

    logger.info("Extracting SadTalker source...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(src_dir))

    zip_path.unlink()
    marker.touch()

    root = src_dir / _SADTALKER_EXTRACT_PREFIX
    if not root.is_dir():
        raise RuntimeError(
            f"Expected SadTalker source at {root} but directory not found"
        )
    return root


class SadTalkerEngine(TalkingHeadEngine):
    """SadTalker engine — 3DMM head motion + lip-sync from audio.

    Architecture (3 stages):
      1. 3DMM coefficient extraction from the source image
      2. Audio → expression coefficients (ExpNet) + head pose (PoseVAE)
      3. Coefficients → video frames (face-vid2vid renderer + MappingNet)
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        use_float16: bool = True,
        size: int = 256,
        preprocess: str = "crop",
        pose_style: int = 0,
        expression_scale: float = 1.0,
        still_mode: bool = False,
        use_blink: bool = True,
    ):
        self.device = device
        self.use_float16 = use_float16
        self.model_dir = Path(model_dir) if model_dir else get_model_dir()
        self.size = size
        self.preprocess = preprocess
        self.pose_style = pose_style
        self.expression_scale = expression_scale
        self.still_mode = still_mode
        self.use_blink = use_blink

        self._sadtalker_root: Path | None = None
        self._preprocess_model = None
        self._audio_to_coeff = None
        self._animate_from_coeff = None

        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Download source, add to sys.path, and load models."""
        self._sadtalker_root = _ensure_sadtalker_source(self.model_dir)

        root_str = str(self._sadtalker_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # SadTalker internal imports (available after sys.path update)
        from src.facerender.animate import AnimateFromCoeff  # noqa: E402
        from src.test_audio2coeff import Audio2Coeff  # noqa: E402
        from src.utils.init_path import init_path  # noqa: E402
        from src.utils.preprocess import CropAndExtract  # noqa: E402

        checkpoint_dir = str(self.model_dir / "sadtalker")
        config_dir = str(self._sadtalker_root / "src" / "config")

        sadtalker_paths = init_path(
            checkpoint_dir,
            config_dir,
            size=self.size,
            old_version=False,
            preprocess=self.preprocess,
        )

        logger.info("Loading SadTalker models...")
        self._preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        self._audio_to_coeff = Audio2Coeff(sadtalker_paths, self.device)
        self._animate_from_coeff = AnimateFromCoeff(sadtalker_paths, self.device)
        self._sadtalker_paths = sadtalker_paths
        logger.info("SadTalker models loaded successfully")

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------

    def generate_frames(
        self,
        image: np.ndarray,
        audio: np.ndarray,
        fps: int = 25,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate talking-head frames with natural head motion.

        SadTalker produces per-frame head rotation/translation from audio
        via its PoseVAE, and expression coefficients via ExpNet, then
        renders through a face-vid2vid generator.

        Keyword Args:
            pose_style: Head motion style 0-45 (default: instance default).
            expression_scale: Expression intensity 0.0-3.0.
            still_mode: Minimize background motion for full-body shots.
            use_blink: Enable automatic eye blinks.
            batch_size: Renderer batch size (default 2).
        """
        from src.generate_batch import get_data  # noqa: E402
        from src.generate_facerender_batch import get_facerender_data  # noqa: E402

        pose_style = kwargs.get("pose_style", self.pose_style)
        expression_scale = kwargs.get("expression_scale", self.expression_scale)
        still_mode = kwargs.get("still_mode", self.still_mode)
        batch_size = kwargs.get("batch_size", 2)

        with tempfile.TemporaryDirectory(prefix="sadtalker_") as tmpdir:
            tmpdir = Path(tmpdir)

            # SadTalker expects file paths — write inputs to temp files
            image_path = str(tmpdir / "input.png")
            cv2.imwrite(image_path, image)

            audio_path = str(tmpdir / "input.wav")
            sf.write(audio_path, audio, 16000, subtype="PCM_16")

            save_dir = str(tmpdir / "output")
            os.makedirs(save_dir, exist_ok=True)

            first_frame_dir = str(tmpdir / "first_frame")
            os.makedirs(first_frame_dir, exist_ok=True)

            # Stage 1: 3DMM coefficient extraction from source image
            logger.info("Extracting 3DMM coefficients...")
            first_coeff_path, crop_pic_path, crop_info = (
                self._preprocess_model.generate(
                    image_path,
                    first_frame_dir,
                    self.preprocess,
                    source_image_flag=True,
                    pic_size=self.size,
                )
            )

            if first_coeff_path is None:
                raise ValueError(
                    "No face detected in image. Ensure the image contains "
                    "a clear, front-facing face."
                )

            # Stage 2: Audio → motion coefficients (ExpNet + PoseVAE)
            logger.info("Generating motion coefficients from audio...")
            batch = get_data(
                first_coeff_path,
                audio_path,
                self.device,
                ref_eyeblink_coeff_path=None,
                still=still_mode,
            )
            coeff_path = self._audio_to_coeff.generate(
                batch,
                save_dir,
                pose_style=pose_style,
                ref_pose_coeff_path=None,
            )

            # Stage 3: Coefficients → rendered frames (face-vid2vid)
            logger.info("Rendering animated frames...")
            data = get_facerender_data(
                coeff_path,
                crop_pic_path,
                first_coeff_path,
                audio_path,
                batch_size=batch_size,
                input_yaw_list=None,
                input_pitch_list=None,
                input_roll_list=None,
                expression_scale=expression_scale,
                still_mode=still_mode,
                preprocess=self.preprocess,
                size=self.size,
            )

            result_path = self._animate_from_coeff.generate(
                data,
                save_dir,
                image_path,
                crop_info,
                enhancer=None,
                background_enhancer=None,
                preprocess=self.preprocess,
                img_size=self.size,
            )

            # Read rendered frames back from the temp video
            frames = _read_video_frames(result_path)

        logger.info("Generated %d frames with SadTalker", len(frames))
        return frames


def _read_video_frames(video_path: str) -> list[np.ndarray]:
    """Read all BGR frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return frames
