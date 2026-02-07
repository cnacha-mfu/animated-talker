"""Animation utilities for SadTalker face rendering.

Converts 3DMM coefficients to keypoint transformations and generates
animated frame sequences.

Reference: https://github.com/OpenTalker/SadTalker
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def headpose_pred_to_degree(pred: torch.Tensor) -> torch.Tensor:
    """Convert softmax bin prediction to angle in degrees."""
    device = pred.device
    idx_tensor = torch.arange(66, dtype=torch.float32, device=device)
    pred = F.softmax(pred, dim=-1)
    degree = torch.sum(pred * idx_tensor, dim=1) * 3 - 99
    return degree


def get_rotation_matrix(yaw: torch.Tensor, pitch: torch.Tensor,
                        roll: torch.Tensor) -> torch.Tensor:
    """Build 3x3 rotation matrix from Euler angles (degrees)."""
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
        torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch),
    ], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([
        torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw),
    ], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([
        torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
        torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
        torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll),
    ], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)
    return rot_mat


def keypoint_transformation(kp_canonical: dict, he: dict,
                            wo_exp: bool = False) -> dict:
    """Apply head pose rotation, translation, and expression to keypoints.

    Args:
        kp_canonical: dict with 'value' tensor (B, K, 3) — canonical keypoints
        he: dict with 'yaw', 'pitch', 'roll' (bin logits), 't' (B,3), 'exp' (B, K*3)
        wo_exp: if True, zero out expression deformation
    """
    kp = kp_canonical["value"]  # (B, K, 3)
    yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if "yaw_in" in he:
        yaw = he["yaw_in"]
    if "pitch_in" in he:
        pitch = he["pitch_in"]
    if "roll_in" in he:
        roll = he["roll_in"]

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (B, 3, 3)

    t, exp = he["t"], he["exp"]
    if wo_exp:
        exp = exp * 0

    # Keypoint rotation
    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)

    # Keypoint translation (zero out x and z to keep face centered)
    t = t.clone()
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # Add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {"value": kp_transformed}


@torch.no_grad()
def make_animation(
    source_image: torch.Tensor,
    source_semantics: torch.Tensor,
    target_semantics: torch.Tensor,
    generator: torch.nn.Module,
    kp_detector: torch.nn.Module,
    mapping: torch.nn.Module,
    yaw_c_seq: torch.Tensor | None = None,
    pitch_c_seq: torch.Tensor | None = None,
    roll_c_seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate animated frames from source image and driving coefficients.

    Args:
        source_image: (B, 3, H, W) source face image
        source_semantics: (B, coeff_nc, T_ctx) source 3DMM coefficients
        target_semantics: (B, T, coeff_nc) target coefficient sequence
        generator: OcclusionAwareSPADEGenerator
        kp_detector: KPDetector
        mapping: MappingNet
        yaw_c_seq, pitch_c_seq, roll_c_seq: optional pose override sequences

    Returns:
        (B, T, 3, H, W) generated frame tensor
    """
    predictions = []

    kp_canonical = kp_detector(source_image)
    he_source = mapping(source_semantics)
    kp_source = keypoint_transformation(kp_canonical, he_source)

    for frame_idx in tqdm(range(target_semantics.shape[1]), desc="Rendering"):
        target_semantics_frame = target_semantics[:, frame_idx]
        he_driving = mapping(target_semantics_frame)
        if yaw_c_seq is not None:
            he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
        if pitch_c_seq is not None:
            he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
        if roll_c_seq is not None:
            he_driving["roll_in"] = roll_c_seq[:, frame_idx]

        kp_driving = keypoint_transformation(kp_canonical, he_driving)
        out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
        predictions.append(out["prediction"])

    return torch.stack(predictions, dim=1)
