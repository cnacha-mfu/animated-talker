"""SadTalker inference engine — audio-driven talking head with head motion.

This module wraps all SadTalker components (face reconstruction, audio-to-
expression, audio-to-pose, and face-vid2vid rendering) into a single class
that takes a face crop + audio and produces animated frames with head movement.

Reference: https://github.com/OpenTalker/SadTalker (CVPR 2023)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm

from animated_talker.model_manager import get_model_dir, get_model_path

logger = logging.getLogger(__name__)

# 3DMM coefficient indices (257-dim output from face reconstruction net)
_ID_END = 80
_EXP_END = 144  # 80:144 = 64 dims
_TEX_END = 224
_ANGLE_END = 227  # 224:227 = 3 dims
_GAMMA_END = 254
_TRANS_END = 257  # 254:257 = 3 dims

# Mel spectrogram parameters (matching SadTalker's audio processing)
_MEL_STEP_SIZE = 16
_SAMPLE_RATE = 16000
_N_MELS = 80


class FaceReconNet(nn.Module):
    """3D face reconstruction network (ResNet50-based).

    Extracts 257-dim 3DMM coefficients from a 224x224 face image.
    Based on Deep3DFaceRecon_pytorch.
    """

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        # Keep everything except the final FC
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Separate FC heads for each coefficient group
        self.final_layers = nn.ModuleList([
            nn.Linear(2048, 80),   # identity
            nn.Linear(2048, 64),   # expression
            nn.Linear(2048, 80),   # texture
            nn.Linear(2048, 3),    # angle
            nn.Linear(2048, 27),   # gamma
            nn.Linear(2048, 3),    # translation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return torch.cat([layer(x) for layer in self.final_layers], dim=1)


class SadTalkerEngine:
    """SadTalker inference engine for audio-driven talking head animation.

    Generates animated frames with natural head movement and lip sync
    from a single face image and audio waveform.
    """

    # Default face-vid2vid config from SadTalker's facerender.yaml
    FACERENDER_CONFIG = {
        "common_params": {
            "num_kp": 15,
            "image_channel": 3,
            "feature_channel": 32,
            "estimate_jacobian": False,
        },
        "kp_detector_params": {
            "temperature": 0.1,
            "block_expansion": 32,
            "max_features": 1024,
            "scale_factor": 0.25,
            "num_blocks": 5,
            "reshape_channel": 16384,
            "reshape_depth": 16,
        },
        "he_estimator_params": {
            "block_expansion": 64,
            "max_features": 2048,
            "num_bins": 66,
        },
        "generator_params": {
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
            "reshape_channel": 32,
            "reshape_depth": 16,
            "num_resblocks": 6,
            "estimate_occlusion_map": True,
            "dense_motion_params": {
                "block_expansion": 32,
                "max_features": 1024,
                "num_blocks": 5,
                "reshape_depth": 16,
                "compress": 4,
            },
        },
        "mapping_params": {
            "coeff_nc": 70,
            "descriptor_nc": 1024,
            "layer": 3,
            "num_kp": 15,
            "num_bins": 66,
        },
    }

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        use_float16: bool = True,
        still_mode: bool = False,
        expression_scale: float = 1.0,
        pose_style: int = 0,
    ):
        self.device = device
        self.dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32
        self.model_dir = Path(model_dir) if model_dir else get_model_dir()
        self.still_mode = still_mode
        self.expression_scale = expression_scale
        self.pose_style = pose_style

        self._load_models()

    def _load_models(self) -> None:
        """Load all SadTalker models."""
        from animated_talker._sadtalker.audio_models import (
            Audio2Exp, Audio2Pose, SimpleWrapperV2,
        )
        from animated_talker._sadtalker.renderer import (
            HEEstimator, KPDetector, MappingNet,
            OcclusionAwareSPADEGenerator,
        )

        logger.info("Loading SadTalker models...")
        cfg = self.FACERENDER_CONFIG
        common = cfg["common_params"]

        # --- Face reconstruction network ---
        self.face_recon = FaceReconNet()
        recon_path = get_model_path("face_recon", self.model_dir)
        recon_ckpt = torch.load(str(recon_path), map_location="cpu", weights_only=False)
        state = recon_ckpt.get("net_recon", recon_ckpt)
        self.face_recon.load_state_dict(state, strict=False)
        self.face_recon.to(self.device, dtype=self.dtype)
        self.face_recon.eval()
        logger.info("Face reconstruction network loaded")

        # --- Audio2Exp ---
        exp_net = SimpleWrapperV2()
        exp_path = get_model_path("audio2exp", self.model_dir)
        exp_ckpt = torch.load(str(exp_path), map_location="cpu", weights_only=False)
        exp_state = exp_ckpt.get("model", exp_ckpt)
        # Handle key prefix (checkpoint may have 'netG.' prefix)
        exp_state_clean = {}
        for k, v in exp_state.items():
            new_k = k.replace("netG.", "") if k.startswith("netG.") else k
            exp_state_clean[new_k] = v
        exp_net.load_state_dict(exp_state_clean, strict=False)
        self.audio2exp = Audio2Exp(exp_net, device=self.device)
        self.audio2exp.to(self.device, dtype=self.dtype)
        self.audio2exp.eval()
        logger.info("Audio2Exp loaded")

        # --- Audio2Pose ---
        self.audio2pose = Audio2Pose(device=self.device)
        pose_path = get_model_path("audio2pose", self.model_dir)
        pose_ckpt = torch.load(str(pose_path), map_location="cpu", weights_only=False)
        pose_state = pose_ckpt.get("model", pose_ckpt)
        self.audio2pose.load_state_dict(pose_state, strict=False)
        self.audio2pose.to(self.device, dtype=self.dtype)
        self.audio2pose.eval()
        logger.info("Audio2Pose loaded")

        # --- Face renderer components (from safetensors or individual files) ---
        self.kp_detector = KPDetector(
            **cfg["kp_detector_params"],
            **{k: common[k] for k in ["num_kp", "image_channel", "feature_channel"]},
            estimate_jacobian=common["estimate_jacobian"],
        )
        self.he_estimator = HEEstimator(
            **cfg["he_estimator_params"],
            **{k: common[k] for k in ["num_kp", "image_channel", "feature_channel"]},
            estimate_jacobian=common["estimate_jacobian"],
        )
        self.generator = OcclusionAwareSPADEGenerator(
            **cfg["generator_params"],
            **{k: common[k] for k in ["num_kp", "image_channel", "feature_channel"]},
            estimate_jacobian=common["estimate_jacobian"],
        )
        self.mapping = MappingNet(**cfg["mapping_params"])

        # Load renderer weights (safetensors or individual checkpoint)
        self._load_renderer_weights()

        # Load mapping weights
        mapping_key = "mapping_still" if self.still_mode else "mapping_full"
        mapping_path = get_model_path(mapping_key, self.model_dir)
        mapping_ckpt = torch.load(str(mapping_path), map_location="cpu", weights_only=False)
        mapping_state = mapping_ckpt.get("model", mapping_ckpt)
        self.mapping.load_state_dict(mapping_state, strict=False)

        # Move to device
        for mod in [self.kp_detector, self.he_estimator, self.generator, self.mapping]:
            mod.to(self.device, dtype=self.dtype)
            mod.eval()

        logger.info("Face renderer and mapping loaded")
        logger.info("All SadTalker models loaded successfully")

    def _load_renderer_weights(self) -> None:
        """Load face renderer weights from safetensors checkpoint."""
        renderer_path = get_model_path("sadtalker_checkpoint", self.model_dir)

        if str(renderer_path).endswith(".safetensors"):
            import safetensors.torch
            checkpoint = safetensors.torch.load_file(str(renderer_path))
        else:
            checkpoint = torch.load(str(renderer_path), map_location="cpu", weights_only=False)

        # Split checkpoint by prefix
        gen_state, kp_state, he_state = {}, {}, {}
        for k, v in checkpoint.items():
            if k.startswith("generator."):
                gen_state[k.replace("generator.", "", 1)] = v
            elif k.startswith("kp_extractor."):
                kp_state[k.replace("kp_extractor.", "", 1)] = v
            elif k.startswith("he_estimator."):
                he_state[k.replace("he_estimator.", "", 1)] = v

        self.generator.load_state_dict(gen_state, strict=False)
        self.kp_detector.load_state_dict(kp_state, strict=False)
        self.he_estimator.load_state_dict(he_state, strict=False)

    def _extract_3dmm(self, face_crop: np.ndarray) -> torch.Tensor:
        """Extract 3DMM coefficients from a 256x256 face crop.

        Returns tensor of shape (1, 257).
        """
        # Resize to 224x224 for the face reconstruction network
        face_224 = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(face_224, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            coeffs = self.face_recon(tensor)  # (1, 257)
        return coeffs

    def _audio_to_mel_chunks(
        self,
        audio: np.ndarray,
        fps: int = 25,
    ) -> tuple[np.ndarray, int]:
        """Convert audio waveform to mel spectrogram chunks.

        Returns (mel_chunks as np.ndarray of shape (T, 1, 80, 16), num_frames).
        """
        # Compute mel spectrogram using librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=_SAMPLE_RATE, n_mels=_N_MELS,
            hop_length=int(_SAMPLE_RATE / fps * 80 / _MEL_STEP_SIZE),
            win_length=int(_SAMPLE_RATE / fps * 80 / _MEL_STEP_SIZE * 4),
            n_fft=2048,
        )
        # Convert to log scale
        mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        mel_idx_multiplier = 80.0 / fps
        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + _MEL_STEP_SIZE > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - _MEL_STEP_SIZE:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + _MEL_STEP_SIZE])
            i += 1

        num_frames = len(mel_chunks)
        # Reshape to (T, 1, 80, 16)
        mel_array = np.stack(mel_chunks, axis=0)[:, np.newaxis, :, :]
        return mel_array, num_frames

    def _build_semantics(
        self,
        ref_coeff: torch.Tensor,
        exp_pred: torch.Tensor,
        pose_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build source and target semantic tensors for the MappingNet.

        Returns:
            source_semantics: (1, 70, T_ctx) - source coefficients (repeated)
            target_semantics: (1, T, 70) - per-frame target coefficients
        """
        # Extract source components
        ref_exp = ref_coeff[:, _ID_END:_EXP_END]  # (1, 64)
        ref_angle = ref_coeff[:, _TEX_END:_ANGLE_END]  # (1, 3)
        ref_trans = ref_coeff[:, _GAMMA_END:_TRANS_END]  # (1, 3)
        ref_semantic = torch.cat([ref_exp, ref_angle, ref_trans], dim=1)  # (1, 70)

        # Source semantics: repeat to form temporal context for Conv1d
        # MappingNet uses Conv1d with kernel=7 and 3 layers with dilation=3
        # Minimum temporal length: 7 + 3*6 = 25, use 27 for safety
        T_ctx = 27
        source_semantics = ref_semantic.unsqueeze(2).repeat(1, 1, T_ctx)  # (1, 70, 27)

        # Target semantics: combine generated expression + pose per frame
        num_frames = exp_pred.shape[1]
        target_list = []
        for t in range(num_frames):
            exp_t = ref_exp + self.expression_scale * exp_pred[:, t, :]  # (1, 64)
            pose_t = pose_pred[:, t, :]  # (1, 6) = angle(3) + trans(3)
            semantic_t = torch.cat([exp_t, pose_t], dim=1)  # (1, 70)
            target_list.append(semantic_t)

        target_semantics = torch.stack(target_list, dim=1)  # (1, T, 70)

        # Each target frame also needs temporal context for Conv1d
        # Expand each frame to (1, 70, T_ctx) for per-frame mapping
        target_expanded = []
        for t in range(num_frames):
            frame = target_semantics[:, t, :].unsqueeze(2).repeat(1, 1, T_ctx)
            target_expanded.append(frame)

        return source_semantics, target_expanded

    @torch.no_grad()
    def generate_frames(
        self,
        face_crop: np.ndarray,
        original_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        audio: np.ndarray,
        fps: int = 25,
        batch_size: int = 4,
    ) -> list[np.ndarray]:
        """Generate animated frames with head motion and lip sync.

        Args:
            face_crop: 256x256 BGR face crop
            original_frame: Full-resolution BGR original image
            bbox: (x1, y1, x2, y2) bounding box of face in original
            audio: Float32 mono waveform at 16kHz
            fps: Target video frame rate
            batch_size: Not used (kept for API compatibility), rendering is per-frame

        Returns:
            List of full-resolution BGR frames with animated face composited
        """
        from animated_talker._sadtalker.animation import (
            keypoint_transformation, headpose_pred_to_degree,
        )
        from animated_talker._sadtalker.renderer import MappingNet

        duration = len(audio) / _SAMPLE_RATE
        logger.info("Generating frames at %d FPS (%.2fs audio)", fps, duration)

        # Step 1: Extract 3DMM coefficients from source face
        logger.info("Extracting 3DMM coefficients...")
        ref_coeff = self._extract_3dmm(face_crop)  # (1, 257)

        # Step 2: Process audio to mel chunks
        logger.info("Processing audio...")
        mel_chunks, num_frames = self._audio_to_mel_chunks(audio, fps)
        if num_frames == 0:
            raise ValueError("Audio too short to generate any frames")
        logger.info("Will generate %d frames", num_frames)

        # Step 3: Generate expression coefficients from audio
        logger.info("Generating expression from audio...")
        mel_tensor = torch.FloatTensor(mel_chunks).unsqueeze(0).to(self.device, dtype=self.dtype)
        # Build reference for Audio2Exp: (1, T, 70) repeated ref coefficients
        ref_exp = ref_coeff[:, _ID_END:_EXP_END]  # (1, 64)
        ref_angle = ref_coeff[:, _TEX_END:_ANGLE_END]
        ref_trans = ref_coeff[:, _GAMMA_END:_TRANS_END]
        ref_semantic = torch.cat([ref_exp, ref_angle, ref_trans], dim=1)  # (1, 70)
        ref_repeated = ref_semantic.unsqueeze(1).repeat(1, num_frames, 1)  # (1, T, 70)

        ratio = torch.ones(1, num_frames, 1, device=self.device, dtype=self.dtype)
        exp_batch = {
            "indiv_mels": mel_tensor,
            "ref": ref_repeated,
            "ratio_gt": ratio,
        }
        exp_result = self.audio2exp.test(exp_batch)
        exp_pred = exp_result["exp_coeff_pred"]  # (1, T, 64)

        # Step 4: Generate head pose from audio
        logger.info("Generating head pose from audio...")
        pose_batch = {
            "indiv_mels": mel_tensor,
            "ref": ref_semantic.unsqueeze(1),  # (1, 1, 70)
            "class": torch.LongTensor([self.pose_style]).to(self.device),
            "num_frames": num_frames,
        }
        pose_result = self.audio2pose.test(pose_batch)
        pose_pred = pose_result["pose_pred"]  # (1, T+1, 6) or (1, T, 6)
        # Trim to match num_frames
        pose_pred = pose_pred[:, :num_frames, :]

        # Step 5: Build source/target semantics
        logger.info("Building animation semantics...")
        source_semantics, target_expanded = self._build_semantics(
            ref_coeff, exp_pred, pose_pred,
        )

        # Step 6: Render animated frames
        logger.info("Rendering %d frames...", num_frames)
        # Prepare source image tensor
        face_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(face_256, cv2.COLOR_BGR2RGB)
        src_tensor = torch.from_numpy(rgb).float() / 255.0
        src_tensor = src_tensor.permute(2, 0, 1).unsqueeze(0)
        src_tensor = src_tensor.to(self.device, dtype=self.dtype)

        # Get canonical keypoints and source keypoints
        kp_canonical = self.kp_detector(src_tensor)
        he_source = self.mapping(source_semantics)

        from animated_talker._sadtalker.animation import keypoint_transformation
        kp_source = keypoint_transformation(kp_canonical, he_source)

        generated_crops = []
        for t in tqdm(range(num_frames), desc="Rendering frames"):
            target_frame = target_expanded[t]  # (1, 70, 27)
            he_driving = self.mapping(target_frame)
            kp_driving = keypoint_transformation(kp_canonical, he_driving)

            out = self.generator(src_tensor, kp_source=kp_source, kp_driving=kp_driving)
            pred = out["prediction"]  # (1, 3, 256, 256)

            # Convert to numpy BGR
            img = pred[0].cpu().float().numpy()
            img = (img.clip(0, 1) * 255).astype(np.uint8)
            img = img.transpose(1, 2, 0)  # (H, W, 3) RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            generated_crops.append(img)

        # Step 7: Composite generated faces onto original frame
        logger.info("Compositing %d frames...", len(generated_crops))
        output_frames = []
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = y2 - y1, x2 - x1

        for gen_crop in generated_crops:
            frame = original_frame.copy()
            gen_resized = cv2.resize(gen_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
            frame[y1:y2, x1:x2] = gen_resized
            output_frames.append(frame)

        logger.info("Frame generation complete")
        return output_frames
