"""MuseTalk v1.5 lip-sync inference engine.

This module vendors the core inference logic from MuseTalk, wrapping the
VAE, UNet, Whisper feature extraction, and face compositing into a single
class that takes a face crop + audio and produces lip-synced frames.

Reference: https://github.com/TMElyralab/MuseTalk
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

from animated_talker.model_manager import get_model_dir, get_model_path

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for audio features."""

    def __init__(self, d_model: int = 384, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FaceParser:
    """BiSeNet-based face parsing for compositing masks.

    Generates semantic segmentation masks to blend the generated mouth
    region seamlessly with the original face.
    """

    def __init__(self, bisenet_path: str, resnet_path: str, device: str = "cuda"):
        self.device = device
        self._load_bisenet(bisenet_path, resnet_path)

    def _load_bisenet(self, bisenet_path: str, resnet_path: str) -> None:
        """Load BiSeNet face parsing model."""
        from animated_talker._bisenet import BiSeNet

        self.model = BiSeNet(n_classes=19)

        # Remove ResNet fc layer (not used in segmentation, not in checkpoint)
        if hasattr(self.model.cp.resnet, "fc"):
            del self.model.cp.resnet.fc
        if hasattr(self.model.cp.resnet, "avgpool"):
            del self.model.cp.resnet.avgpool

        state_dict = torch.load(bisenet_path, map_location=self.device, weights_only=False)

        # Load with strict=False — some checkpoints omit SpatialPath weights
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("BiSeNet missing keys (will use init weights): %d keys", len(missing))
            logger.debug("Missing: %s", missing[:5])
        if unexpected:
            logger.warning("BiSeNet unexpected keys (ignored): %d keys", len(unexpected))
            logger.debug("Unexpected: %s", unexpected[:5])

        self.model.to(self.device)
        self.model.eval()

    def parse(self, face_image: np.ndarray) -> np.ndarray:
        """Parse a face image and return semantic label map.

        Args:
            face_image: BGR image of shape (H, W, 3)

        Returns:
            Label map of shape (H, W) with class indices 0-18
        """
        h, w = face_image.shape[:2]
        img = cv2.resize(face_image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
        return parsing

    def get_mouth_mask(
        self,
        face_image: np.ndarray,
        upper_boundary_ratio: float = 0.5,
        blur_kernel: int = 21,
    ) -> np.ndarray:
        """Generate a soft mask for the lower face / mouth region.

        Returns a float32 mask of shape (H, W) in range [0, 1].
        """
        parsing = self.parse(face_image)
        h, w = parsing.shape

        # Face parsing labels: 1=skin, 11=upper_lip, 12=lower_lip, 13=teeth/mouth
        # Create mask for lower face region
        mouth_labels = {1, 11, 12, 13}
        mask = np.isin(parsing, list(mouth_labels)).astype(np.float32)

        # Zero out upper portion of face
        cutoff = int(h * upper_boundary_ratio)
        mask[:cutoff] = 0

        # Gaussian blur for smooth edges
        if blur_kernel > 0:
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

        return mask


class MuseTalkEngine:
    """MuseTalk v1.5 inference engine for lip-sync frame generation.

    Loads the VAE, UNet, Whisper model, and face parser, then generates
    lip-synced frames from a face crop and audio waveform.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        use_float16: bool = True,
    ):
        self.device = device
        self.dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32
        self.model_dir = Path(model_dir) if model_dir else get_model_dir()

        self._load_models()

    def _load_models(self) -> None:
        """Load all required models."""
        logger.info("Loading MuseTalk models...")

        # Load VAE
        vae_dir = self.model_dir / "sd-vae-ft-mse"
        self.vae = AutoencoderKL.from_pretrained(str(vae_dir))
        self.vae = self.vae.to(self.device, dtype=self.dtype)
        self.vae.eval()
        logger.info("VAE loaded")

        # Load UNet
        musetalk_dir = self.model_dir / "musetalk"
        config_path = musetalk_dir / "musetalk.json"
        weights_path = musetalk_dir / "unet.pth"

        self.unet = UNet2DConditionModel.from_config(str(config_path))
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self.unet.load_state_dict(state_dict)
        self.unet = self.unet.to(self.device, dtype=self.dtype)
        self.unet.eval()
        logger.info("UNet loaded")

        # Load positional encoding
        self.pe = PositionalEncoding(d_model=384).to(self.device, dtype=self.dtype)

        # Load Whisper for audio feature extraction
        whisper_dir = self.model_dir / "whisper-tiny"
        self.whisper_processor = WhisperProcessor.from_pretrained(str(whisper_dir))
        self.whisper_model = WhisperModel.from_pretrained(str(whisper_dir))
        self.whisper_model = self.whisper_model.to(self.device, dtype=self.dtype)
        self.whisper_model.eval()
        logger.info("Whisper loaded")

        # Load face parser for compositing
        bisenet_path = get_model_path("face_parse_bisenet", self.model_dir)
        resnet_path = get_model_path("face_parse_resnet", self.model_dir)
        self.face_parser = FaceParser(
            str(bisenet_path), str(resnet_path), device=self.device
        )
        logger.info("Face parser loaded")

        logger.info("All models loaded successfully")

    def extract_audio_features(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> torch.Tensor:
        """Extract Whisper encoder embeddings from audio.

        Args:
            audio: Float32 waveform at target sample rate
            sr: Sample rate (default 16000 for Whisper)

        Returns:
            Tensor of shape (num_frames, feature_dim) where feature_dim=384
        """
        # Process audio through Whisper
        inputs = self.whisper_processor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            encoder_output = self.whisper_model.encoder(input_features)
            features = encoder_output.last_hidden_state  # (1, T, 384)

        return features.squeeze(0)  # (T, 384)

    def features_to_chunks(
        self,
        features: torch.Tensor,
        num_frames: int,
        fps: int = 25,
        padding_left: int = 2,
        padding_right: int = 2,
    ) -> list[torch.Tensor]:
        """Map Whisper features to per-frame audio chunks.

        Whisper features are at ~50 FPS. We map them to video frames at
        the target FPS using a sliding window.

        Args:
            features: (T_audio, 384) Whisper encoder features
            num_frames: Number of video frames to generate
            fps: Target video frame rate
            padding_left: Number of context frames to include before
            padding_right: Number of context frames to include after

        Returns:
            List of tensors, one per video frame, each (window_size, 384)
        """
        T_audio = features.shape[0]

        # Whisper features are at 50 FPS (30 seconds audio -> 1500 features)
        whisper_fps = 50.0
        chunk_size = padding_left + 1 + padding_right

        chunks = []
        for i in range(num_frames):
            # Map video frame index to whisper feature index
            center = int(i * whisper_fps / fps)
            start = max(0, center - padding_left)
            end = min(T_audio, center + padding_right + 1)

            chunk = features[start:end]

            # Pad if necessary
            if chunk.shape[0] < chunk_size:
                pad_size = chunk_size - chunk.shape[0]
                padding = torch.zeros(pad_size, features.shape[1], device=features.device, dtype=features.dtype)
                chunk = torch.cat([chunk, padding], dim=0)

            chunks.append(chunk[:chunk_size])

        return chunks

    @torch.no_grad()
    def generate_frames(
        self,
        face_crop: np.ndarray,
        original_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        audio: np.ndarray,
        fps: int = 25,
        batch_size: int = 8,
    ) -> list[np.ndarray]:
        """Generate lip-synced frames from a face crop and audio.

        Args:
            face_crop: 256x256 BGR face crop
            original_frame: Full-resolution BGR original image
            bbox: (x1, y1, x2, y2) bounding box of face in original
            audio: Float32 mono waveform at 16kHz
            fps: Target video frame rate
            batch_size: Number of frames to process together

        Returns:
            List of full-resolution BGR frames with lip-synced face composited
        """
        duration = len(audio) / 16000
        num_frames = int(duration * fps)

        if num_frames == 0:
            raise ValueError("Audio too short to generate any frames")

        logger.info("Generating %d frames at %d FPS (%.2fs)", num_frames, fps, duration)

        # Encode reference face to latent
        ref_image = self._preprocess_face(face_crop)
        ref_latent = self.vae.encode(ref_image).latent_dist.sample()
        ref_latent = ref_latent * self.vae.config.scaling_factor

        # Create masked face latent (mask bottom half)
        masked_face = face_crop.copy()
        masked_face[128:, :, :] = 0  # Zero bottom half (mouth region)
        masked_image = self._preprocess_face(masked_face)
        masked_latent = self.vae.encode(masked_image).latent_dist.sample()
        masked_latent = masked_latent * self.vae.config.scaling_factor

        # Extract audio features
        audio_features = self.extract_audio_features(audio)
        audio_chunks = self.features_to_chunks(audio_features, num_frames, fps)

        # Get face parsing mask for compositing
        mouth_mask = self.face_parser.get_mouth_mask(face_crop)

        # Generate frames in batches
        generated_crops = []
        for batch_start in tqdm(
            range(0, num_frames, batch_size),
            desc="Generating frames",
            unit="batch",
        ):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_chunks = audio_chunks[batch_start:batch_end]
            current_batch_size = len(batch_chunks)

            # Stack audio chunks and apply positional encoding
            audio_batch = torch.stack(batch_chunks)  # (B, window, 384)
            audio_batch = self.pe(audio_batch)

            # Repeat latents for batch
            ref_batch = ref_latent.repeat(current_batch_size, 1, 1, 1)
            masked_batch = masked_latent.repeat(current_batch_size, 1, 1, 1)

            # Concatenate masked + reference latents channel-wise
            latent_input = torch.cat([masked_batch, ref_batch], dim=1)

            # UNet forward pass (single-step, not diffusion)
            timestep = torch.zeros(current_batch_size, device=self.device, dtype=torch.long)
            pred_latent = self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=audio_batch,
            ).sample

            # Decode predicted latent to pixel space
            pred_latent = pred_latent / self.vae.config.scaling_factor
            decoded = self.vae.decode(pred_latent).sample

            # Convert to numpy images
            for j in range(current_batch_size):
                img = decoded[j].cpu().float().numpy()
                img = (img * 0.5 + 0.5).clip(0, 1)
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                generated_crops.append(img)

        # Composite generated faces onto original frame
        logger.info("Compositing %d frames...", len(generated_crops))
        output_frames = []
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = y2 - y1, x2 - x1

        for gen_crop in generated_crops:
            frame = original_frame.copy()

            # Resize generated crop to match original bbox size
            gen_resized = cv2.resize(gen_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

            # Resize mask to match
            mask_resized = cv2.resize(mouth_mask, (crop_w, crop_h))
            mask_3ch = np.stack([mask_resized] * 3, axis=-1)

            # Blend: generated face in mouth region, original elsewhere
            original_crop = frame[y1:y2, x1:x2].astype(np.float32)
            gen_float = gen_resized.astype(np.float32)
            blended = original_crop * (1 - mask_3ch) + gen_float * mask_3ch
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)

            output_frames.append(frame)

        logger.info("Frame generation complete")
        return output_frames

    def _preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Convert a 256x256 BGR face image to model input tensor.

        Returns tensor of shape (1, 3, 256, 256) normalized to [-1, 1].
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)
