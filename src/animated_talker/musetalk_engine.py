"""Thin wrapper around MuseTalk v1.5 inference pipeline.

Uses MuseTalk's own code for model loading, audio processing,
face detection, UNet inference, and face blending.
"""

from __future__ import annotations

import copy
import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from animated_talker.model_manager import ensure_musetalk_on_path, get_models_dir

logger = logging.getLogger(__name__)


class MuseTalkEngine:
    """MuseTalk v1.5 inference engine.

    Wraps MuseTalk's actual inference code for reliable lip-sync generation.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_float16: bool = True,
    ):
        self.device = torch.device(device)
        self.use_float16 = use_float16 and device != "cpu"

        # Ensure MuseTalk is importable
        ensure_musetalk_on_path()

        self._load_models()

    def _load_models(self) -> None:
        """Load all models using MuseTalk's own loaders."""
        models_dir = get_models_dir()

        # Change to MuseTalk dir so relative paths work
        from animated_talker.model_manager import get_musetalk_dir
        original_cwd = os.getcwd()
        os.chdir(str(get_musetalk_dir()))

        try:
            from musetalk.utils.utils import load_all_model
            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.utils.face_parsing import FaceParsing
            from transformers import WhisperModel

            logger.info("Loading MuseTalk models...")

            # Load VAE, UNet, PositionalEncoding
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(models_dir / "musetalkV15" / "unet.pth"),
                vae_type="sd-vae",
                unet_config=str(models_dir / "musetalkV15" / "musetalk.json"),
                device=self.device,
            )

            if self.use_float16:
                self.pe = self.pe.half().to(self.device)
                self.vae.vae = self.vae.vae.half().to(self.device)
                self.unet.model = self.unet.model.half().to(self.device)
            else:
                self.pe = self.pe.float().to(self.device)
                self.vae.vae = self.vae.vae.float().to(self.device)
                self.unet.model = self.unet.model.float().to(self.device)

            # Audio processor (Whisper-based)
            self.audio_processor = AudioProcessor(
                feature_extractor_path=str(models_dir / "whisper")
            )

            # Whisper model for feature extraction
            weight_dtype = self.unet.model.dtype
            self.whisper = WhisperModel.from_pretrained(
                str(models_dir / "whisper")
            ).to(device=self.device, dtype=weight_dtype).eval()
            self.whisper.requires_grad_(False)
            self.weight_dtype = weight_dtype

            # Face parsing for blending
            self.face_parser = FaceParsing()

            # Timestep tensor (single-step, not diffusion)
            self.timesteps = torch.tensor([0], device=self.device)

            logger.info("All models loaded successfully")
        finally:
            os.chdir(original_cwd)

    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25,
        batch_size: int = 8,
        bbox_shift: int = 0,
        extra_margin: int = 10,
    ) -> str:
        """Generate lip-synced video from image + audio.

        Uses MuseTalk's full pipeline: face detection, audio features,
        UNet inference, face blending, and FFmpeg encoding.

        Args:
            image_path: Path to portrait image (JPEG/PNG)
            audio_path: Path to audio file (MP3/WAV)
            output_path: Path for output video (MP4)
            fps: Video frame rate (default 25)
            batch_size: Frames per batch (default 8)
            bbox_shift: Face bbox vertical shift (default 0)
            extra_margin: Extra margin below jaw (default 10)

        Returns:
            Path to output video
        """
        from animated_talker.model_manager import get_musetalk_dir

        original_cwd = os.getcwd()
        os.chdir(str(get_musetalk_dir()))

        try:
            return self._run_inference(
                image_path, audio_path, output_path,
                fps, batch_size, bbox_shift, extra_margin,
            )
        finally:
            os.chdir(original_cwd)

    def _run_inference(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int,
        batch_size: int,
        bbox_shift: int,
        extra_margin: int,
    ) -> str:
        from musetalk.utils.utils import datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
        from musetalk.utils.blending import get_image

        # Create temp directories
        tmp_dir = tempfile.mkdtemp(prefix="animated_talker_")
        frames_dir = os.path.join(tmp_dir, "frames")
        result_dir = os.path.join(tmp_dir, "results")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        try:
            # Step 1: Create input frames from single image
            logger.info("Preparing input frames...")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Determine frame count from audio duration
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio_data) / sr
            num_frames = int(duration * fps)
            logger.info("Audio: %.2fs → %d frames at %d FPS", duration, num_frames, fps)

            # Write frames (MuseTalk expects numbered PNGs)
            for i in range(num_frames):
                cv2.imwrite(os.path.join(frames_dir, f"{i:08d}.png"), image)

            input_img_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

            # Step 2: Extract audio features
            logger.info("Extracting audio features...")
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                audio_path
            )
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )

            # Step 3: Detect face landmarks and bounding boxes
            logger.info("Detecting faces...")
            coord_list, frame_list = get_landmark_and_bbox(
                input_img_list, upperbondrange=bbox_shift
            )

            # Step 4: Encode cropped faces to VAE latents
            logger.info("Encoding face latents...")
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + extra_margin, frame.shape[0])
                crop = cv2.resize(
                    frame[y1:y2, x1:x2], (256, 256),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                input_latent_list.append(self.vae.get_latents_for_unet(crop))

            if not input_latent_list:
                raise ValueError("No face detected in the image")

            # Step 5: Create cycled lists
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Step 6: Batch inference through UNet
            logger.info("Running UNet inference (%d chunks, batch_size=%d)...",
                        len(whisper_chunks), batch_size)
            gen = datagen(
                whisper_chunks, input_latent_list_cycle,
                batch_size=batch_size, device=self.device,
            )
            res_frame_list = []
            for whisper_batch, latent_batch in tqdm(gen, desc="Generating"):
                audio_feature_batch = self.pe(whisper_batch)
                pred_latents = self.unet.model(
                    latent_batch.to(dtype=self.unet.model.dtype),
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                for frame in self.vae.decode_latents(pred_latents):
                    res_frame_list.append(frame)

            # Step 7: Blend generated faces onto original frames
            logger.info("Blending %d frames...", len(res_frame_list))
            for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + extra_margin, ori_frame.shape[0])
                res_frame = cv2.resize(
                    res_frame.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                )
                combine = get_image(
                    ori_frame, res_frame, [x1, y1, x2, y2],
                    mode="jaw", fp=self.face_parser,
                )
                cv2.imwrite(os.path.join(result_dir, f"{i:08d}.png"), combine)

            # Step 8: Encode video with FFmpeg
            logger.info("Encoding video...")
            temp_video = os.path.join(tmp_dir, "temp.mp4")
            os.system(
                f'ffmpeg -y -v warning -r {fps} -f image2 '
                f'-i "{result_dir}/%08d.png" '
                f'-vcodec libx264 -vf format=yuv420p -crf 18 "{temp_video}"'
            )
            os.system(
                f'ffmpeg -y -v warning -i "{audio_path}" -i "{temp_video}" '
                f'-c:v copy -c:a aac -shortest "{output_path}"'
            )

            logger.info("Output saved to %s", output_path)
            return output_path

        finally:
            # Cleanup temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)
