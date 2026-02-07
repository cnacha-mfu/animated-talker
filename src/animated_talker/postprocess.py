"""Optional post-processing: face restoration and super-resolution.

These modules require the 'postprocess' optional dependency group:
    pip install animated-talker[postprocess]
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceRestorer:
    """CodeFormer-based face restoration for artifact removal.

    Enhances face details and removes artifacts from generated frames.
    Requires: pip install basicsr gfpgan
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        fidelity: float = 0.7,
    ):
        self.fidelity = fidelity
        self.device = device
        self._restorer = None
        self._model_path = model_path
        self._init_restorer()

    def _init_restorer(self) -> None:
        """Initialize the CodeFormer restorer."""
        try:
            from basicsr.utils import img2tensor, tensor2img
            from basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch
            import torch
        except ImportError:
            raise ImportError(
                "CodeFormer requires additional dependencies. Install with:\n"
                "  pip install animated-talker[postprocess]"
            )

        self._img2tensor = img2tensor
        self._tensor2img = tensor2img
        self._torch = torch

        net = CodeFormerArch(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

        if self._model_path:
            ckpt = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)))

        net = net.to(self.device).eval()
        self._net = net
        logger.info("CodeFormer face restorer initialized")

    def restore(self, frame: np.ndarray) -> np.ndarray:
        """Apply face restoration to a single frame.

        Args:
            frame: BGR image (any size)

        Returns:
            Restored BGR image (same size)
        """
        import torch

        h, w = frame.shape[:2]

        # CodeFormer expects 512x512 aligned face
        face = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        face_t = self._img2tensor(face / 255.0, bgr2rgb=True, float32=True)
        face_t = face_t.unsqueeze(0).to(self.device)

        # Normalize to [-1, 1]
        face_t = (face_t - 0.5) / 0.5

        with torch.no_grad():
            output = self._net(face_t, w=self.fidelity, adain=True)[0]
            output = (output * 0.5 + 0.5).clamp(0, 1)
            restored = self._tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(0, 1))

        restored = (restored * 255).clip(0, 255).astype(np.uint8)
        if (h, w) != (512, 512):
            restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return restored


class Upscaler:
    """Real-ESRGAN super-resolution upscaler.

    Upscales frames by 4x for higher output resolution.
    Requires: pip install realesrgan basicsr
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        scale: int = 4,
    ):
        self.scale = scale
        self.device = device
        self._upsampler = None
        self._model_path = model_path
        self._init_upscaler()

    def _init_upscaler(self) -> None:
        """Initialize the Real-ESRGAN upscaler."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "Real-ESRGAN requires additional dependencies. Install with:\n"
                "  pip install animated-talker[postprocess]"
            )

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )

        self._upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self._model_path) if self._model_path else None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self.device != "cpu",
        )
        logger.info("Real-ESRGAN upscaler initialized (scale=%d)", self.scale)

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame.

        Args:
            frame: BGR image

        Returns:
            Upscaled BGR image (scale * original dimensions)
        """
        output, _ = self._upsampler.enhance(frame, outscale=self.scale)
        return output


def restore_frames(
    frames: list[np.ndarray],
    face_restore: bool = False,
    upscale: bool = False,
    device: str = "cuda",
    restore_fidelity: float = 0.7,
    upscale_factor: int = 4,
    codeformer_path: str | Path | None = None,
    realesrgan_path: str | Path | None = None,
) -> list[np.ndarray]:
    """Apply optional post-processing to a list of frames.

    Order: face restoration first, then upscaling.
    """
    if not face_restore and not upscale:
        return frames

    restorer = None
    upscaler = None

    if face_restore:
        restorer = FaceRestorer(
            model_path=codeformer_path,
            device=device,
            fidelity=restore_fidelity,
        )

    if upscale:
        upscaler = Upscaler(
            model_path=realesrgan_path,
            device=device,
            scale=upscale_factor,
        )

    processed = []
    for i, frame in enumerate(frames):
        if restorer:
            frame = restorer.restore(frame)
        if upscaler:
            frame = upscaler.upscale(frame)
        processed.append(frame)

        if (i + 1) % 50 == 0:
            logger.info("Post-processed %d/%d frames", i + 1, len(frames))

    logger.info("Post-processing complete (%d frames)", len(processed))
    return processed


def postprocess_video(
    input_path: str | Path,
    output_path: str | Path,
    audio_path: str | Path,
    face_restore: bool = False,
    upscale: bool = False,
    device: str = "cuda",
    restore_fidelity: float = 0.7,
    upscale_factor: int = 4,
    fps: int = 25,
) -> Path:
    """Post-process an existing video: extract frames, enhance, re-encode.

    Args:
        input_path: Path to input video (MP4)
        output_path: Path for output video (MP4)
        audio_path: Path to original audio for re-muxing
        face_restore: Apply CodeFormer face restoration
        upscale: Apply Real-ESRGAN upscaling
        device: PyTorch device
        restore_fidelity: CodeFormer fidelity weight
        upscale_factor: Upscale multiplier
        fps: Video frame rate

    Returns:
        Path to the output video file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Extract frames from video
    logger.info("Extracting frames from %s for post-processing...", input_path.name)
    cap = cv2.VideoCapture(str(input_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {input_path}")

    logger.info("Extracted %d frames", len(frames))

    # Apply post-processing
    frames = restore_frames(
        frames,
        face_restore=face_restore,
        upscale=upscale,
        device=device,
        restore_fidelity=restore_fidelity,
        upscale_factor=upscale_factor,
    )

    # Re-encode with FFmpeg
    from animated_talker.video_encoder import encode_video

    encode_video(
        frames=frames,
        audio_path=audio_path,
        output_path=output_path,
        fps=fps,
    )

    return output_path
