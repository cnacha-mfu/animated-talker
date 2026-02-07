"""Ditto-TalkingHead inference engine for audio-driven talking head synthesis.

Wraps the Ditto StreamSDK pipeline to generate lip-synced video from a
source portrait image and an audio file.

Reference: https://github.com/antgroup/ditto-talkinghead
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

from animated_talker.model_manager import get_ditto_repo_dir, get_model_dir

logger = logging.getLogger(__name__)


class DittoEngine:
    """Ditto-TalkingHead inference engine.

    Loads the Ditto pipeline (HuBERT audio encoder, LMDM diffusion model,
    appearance/motion/warp/decode networks) and generates lip-synced videos
    from a single portrait image and audio file.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        use_float16: bool = True,
    ):
        self.device = device
        self.use_float16 = use_float16
        self.model_dir = Path(model_dir) if model_dir else get_model_dir()
        self._sdk = None

        self._init_sdk()

    def _init_sdk(self) -> None:
        """Initialize the Ditto StreamSDK."""
        repo_dir = get_ditto_repo_dir(self.model_dir)
        if not repo_dir.exists():
            raise RuntimeError(
                f"Ditto repo not found at {repo_dir}. "
                "Run 'animated-talker download-models' first."
            )

        # Add Ditto repo to Python path so its modules can be imported
        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        logger.info("Loading Ditto models from %s ...", self.model_dir)

        from stream_pipeline_offline import StreamSDK

        cfg_pkl = str(self.model_dir / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl")
        data_root = str(self.model_dir / "ditto_pytorch")

        self._sdk = StreamSDK(cfg_pkl, data_root)
        logger.info("Ditto StreamSDK initialized")

    def generate_video(
        self,
        image_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        fps: int = 25,
        emotion: int = 4,
        sampling_timesteps: int = 50,
        fade_in: int = -1,
        fade_out: int = -1,
    ) -> Path:
        """Generate a lip-synced talking head video.

        Args:
            image_path: Path to source portrait image (JPEG/PNG)
            audio_path: Path to audio file (WAV/MP3)
            output_path: Path for output video (MP4)
            fps: Video frame rate (default 25, Ditto's native rate)
            emotion: Emotion style index (default 4 = neutral)
            sampling_timesteps: Diffusion sampling steps (default 50)
            fade_in: Number of frames to fade in (-1 = disabled)
            fade_out: Number of frames to fade out (-1 = disabled)

        Returns:
            Path to the output video file
        """
        if self._sdk is None:
            raise RuntimeError("Ditto SDK not initialized")

        image_path = Path(image_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio to determine duration and frame count
        audio, sr = librosa.core.load(str(audio_path), sr=16000)
        num_frames = math.ceil(len(audio) / 16000 * fps)
        duration = len(audio) / 16000

        logger.info(
            "Generating %d frames at %d FPS (%.2fs audio)",
            num_frames, fps, duration,
        )

        # Setup the SDK with source image
        setup_kwargs = {
            "emo": emotion,
            "sampling_timesteps": sampling_timesteps,
        }
        self._sdk.setup(
            str(image_path),
            str(output_path),
            **setup_kwargs,
        )

        # Configure frame count and fade
        self._sdk.setup_Nd(
            N_d=num_frames,
            fade_in=fade_in,
            fade_out=fade_out,
        )

        # Run inference (offline mode — process full audio at once)
        aud_feat = self._sdk.wav2feat.wav2feat(audio)
        self._sdk.audio2motion_queue.put(aud_feat)
        self._sdk.close()

        # Mux audio with the generated silent video
        tmp_video = self._sdk.tmp_output_path
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg not found on PATH. Install FFmpeg >= 5.0: "
                "https://ffmpeg.org/download.html"
            )

        cmd = [
            ffmpeg_path,
            "-loglevel", "error",
            "-y",
            "-i", str(tmp_video),
            "-i", str(audio_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        logger.debug("FFmpeg mux command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio mux failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Clean up temp file
        try:
            os.remove(tmp_video)
        except OSError:
            pass

        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info("Video saved to %s (%.1f MB)", output_path, file_size)
        return output_path
