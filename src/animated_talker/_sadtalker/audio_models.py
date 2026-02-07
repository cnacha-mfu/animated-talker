"""Audio-to-expression and audio-to-pose networks from SadTalker.

Contains:
- SimpleWrapperV2 (ExpNet): mel spectrogram → expression coefficient deltas
- AudioEncoder: mel spectrogram → audio embedding (Wav2Lip-style)
- CVAE: conditional VAE for pose generation
- Audio2Exp: wrapper for expression inference
- Audio2Pose: wrapper for pose inference

Reference: https://github.com/OpenTalker/SadTalker
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Conv2d(nn.Module):
    """Conv2d → BatchNorm → ReLU with optional residual connection."""

    def __init__(self, cin, cout, kernel_size, stride, padding,
                 residual=False, use_act=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            out += x
        if self.use_act:
            return self.act(out)
        return out


# ---------------------------------------------------------------------------
# Audio2Exp: SimpleWrapperV2 (ExpNet)
# ---------------------------------------------------------------------------

class SimpleWrapperV2(nn.Module):
    """ExpNet — maps mel spectrogram to expression coefficient deltas.

    Input:  mel (B*T, 1, 80, 16), ref expression (B, T, 64), ratio (B, T)
    Output: expression deltas (B, T, 64)
    """

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.mapping1 = nn.Linear(512 + 64 + 1, 64)
        nn.init.constant_(self.mapping1.bias, 0.0)

    def forward(self, x: torch.Tensor, ref: torch.Tensor,
                ratio: torch.Tensor) -> torch.Tensor:
        x = self.audio_encoder(x).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)
        ratio = ratio.reshape(x.size(0), -1)
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1))
        out = y.reshape(ref.shape[0], ref.shape[1], -1)
        return out


class Audio2Exp(nn.Module):
    """Wrapper that runs SimpleWrapperV2 in sliding-window batches."""

    def __init__(self, netG: SimpleWrapperV2, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.netG = netG.to(device)

    @torch.no_grad()
    def test(self, batch: dict) -> dict:
        mel_input = batch["indiv_mels"]  # (B, T, 1, 80, 16)
        T = mel_input.shape[1]
        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), desc="Audio2Exp"):
            current_mel = mel_input[:, i:i + 10]
            ref = batch["ref"][:, :, :64][:, i:i + 10]
            ratio = batch["ratio_gt"][:, i:i + 10]
            audiox = current_mel.view(-1, 1, 80, 16)
            pred = self.netG(audiox, ref, ratio)
            exp_coeff_pred.append(pred)

        return {"exp_coeff_pred": torch.cat(exp_coeff_pred, dim=1)}


# ---------------------------------------------------------------------------
# Audio2Pose building blocks
# ---------------------------------------------------------------------------

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class ResUnet(nn.Module):
    """Residual U-Net for pose embedding encoding/decoding."""

    def __init__(self, channel: int = 1, filters: list[int] | None = None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=(2, 1), padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=(2, 1), padding=1)
        self.bridge = ResidualConv(filters[2], filters[3], stride=(2, 1), padding=1)

        self.upsample_1 = Upsample(filters[3], filters[3], kernel=(2, 1), stride=(2, 1))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=(2, 1), stride=(2, 1))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=(2, 1), stride=(2, 1))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.bridge(x3)

        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        return self.output_layer(x10)


# ---------------------------------------------------------------------------
# Audio2Pose: AudioEncoder + CVAE
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """Wav2Lip-style audio encoder: mel spectrogram → 512-dim embedding."""

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences: torch.Tensor) -> torch.Tensor:
        # audio_sequences: (B, T, 1, 80, 16)
        B = audio_sequences.size(0)
        audio_sequences = torch.cat(
            [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0,
        )
        audio_embedding = self.audio_encoder(audio_sequences)
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))
        return audio_embedding.squeeze(-1).squeeze(-1)  # (B, T, 512)


class ENCODER(nn.Module):
    """CVAE encoder for pose estimation."""

    def __init__(self, layer_sizes: list[int], latent_size: int,
                 num_classes: int, audio_emb_in_size: int,
                 audio_emb_out_size: int, seq_len: int):
        super().__init__()
        self.resunet = ResUnet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.MLP = nn.Sequential()
        layer_sizes = list(layer_sizes)  # copy
        layer_sizes[0] += latent_size + seq_len * audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(f"L{i}", nn.Linear(in_size, out_size))
            self.MLP.add_module(f"A{i}", nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_logvar = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)
        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch: dict) -> dict:
        class_id = batch["class"]
        pose_motion_gt = batch["pose_motion_gt"]
        ref = batch["ref"]
        bs = pose_motion_gt.shape[0]
        audio_in = batch["audio_emb"]

        pose_emb = self.resunet(pose_motion_gt.unsqueeze(1))
        pose_emb = pose_emb.reshape(bs, -1)

        audio_out = self.linear_audio(audio_in)
        audio_out = audio_out.reshape(bs, -1)

        class_bias = self.classbias[class_id]
        x_in = torch.cat([ref, pose_emb, audio_out, class_bias], dim=-1)
        x_out = self.MLP(x_in)

        mu = self.linear_means(x_out)
        logvar = self.linear_logvar(x_out)

        batch.update({"mu": mu, "logvar": logvar})
        return batch


class DECODER(nn.Module):
    """CVAE decoder for pose generation."""

    def __init__(self, layer_sizes: list[int], latent_size: int,
                 num_classes: int, audio_emb_in_size: int,
                 audio_emb_out_size: int, seq_len: int):
        super().__init__()
        self.resunet = ResUnet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.MLP = nn.Sequential()
        input_size = latent_size + seq_len * audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(
            zip([input_size] + list(layer_sizes[:-1]), layer_sizes)
        ):
            self.MLP.add_module(f"L{i}", nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(f"A{i}", nn.ReLU())
            else:
                self.MLP.add_module("sigmoid", nn.Sigmoid())

        self.pose_linear = nn.Linear(6, 6)
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)
        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch: dict) -> dict:
        z = batch["z"]
        bs = z.shape[0]
        class_id = batch["class"]
        ref = batch["ref"]
        audio_in = batch["audio_emb"]

        audio_out = self.linear_audio(audio_in)
        audio_out = audio_out.reshape([bs, -1])
        class_bias = self.classbias[class_id]

        z = z + class_bias
        x_in = torch.cat([ref, z, audio_out], dim=-1)
        x_out = self.MLP(x_in)
        x_out = x_out.reshape((bs, self.seq_len, -1))

        pose_emb = self.resunet(x_out.unsqueeze(1))
        pose_motion_pred = self.pose_linear(pose_emb.squeeze(1))

        batch.update({"pose_motion_pred": pose_motion_pred})
        return batch


class CVAE(nn.Module):
    """Conditional VAE for head pose generation from audio."""

    def __init__(
        self,
        encoder_layer_sizes: list[int] | None = None,
        decoder_layer_sizes: list[int] | None = None,
        latent_size: int = 64,
        num_classes: int = 46,
        audio_emb_in_size: int = 512,
        audio_emb_out_size: int = 6,
        seq_len: int = 32,
    ):
        super().__init__()
        if encoder_layer_sizes is None:
            encoder_layer_sizes = [192, 128]
        if decoder_layer_sizes is None:
            decoder_layer_sizes = [128, 192]

        self.latent_size = latent_size
        self.encoder = ENCODER(
            encoder_layer_sizes, latent_size, num_classes,
            audio_emb_in_size, audio_emb_out_size, seq_len,
        )
        self.decoder = DECODER(
            decoder_layer_sizes, latent_size, num_classes,
            audio_emb_in_size, audio_emb_out_size, seq_len,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: dict) -> dict:
        batch = self.encoder(batch)
        z = self.reparameterize(batch["mu"], batch["logvar"])
        batch["z"] = z
        return self.decoder(batch)

    def test(self, batch: dict) -> dict:
        return self.decoder(batch)


class Audio2Pose(nn.Module):
    """Audio-to-pose network: generates head motion from audio.

    Uses an audio encoder (Wav2Lip-style) and a CVAE to generate
    6-DOF head pose sequences (yaw, pitch, roll, tx, ty, tz).
    """

    # Default CVAE config from SadTalker's auido2pose.yaml
    SEQ_LEN = 32
    LATENT_SIZE = 64

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.seq_len = self.SEQ_LEN
        self.latent_dim = self.LATENT_SIZE
        self.device = device

        self.audio_encoder = AudioEncoder()
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE(
            encoder_layer_sizes=[192, 128],
            decoder_layer_sizes=[128, 192],
            latent_size=self.LATENT_SIZE,
            num_classes=46,
            audio_emb_in_size=512,
            audio_emb_out_size=6,
            seq_len=self.SEQ_LEN,
        )

    @torch.no_grad()
    def test(self, x: dict) -> dict:
        batch: dict = {}
        ref = x["ref"]  # (B, 1, 70)
        batch["ref"] = x["ref"][:, 0, -6:]  # (B, 6)
        batch["class"] = x["class"]
        bs = ref.shape[0]

        indiv_mels = x["indiv_mels"]  # (B, T, 1, 80, 16)
        indiv_mels_use = indiv_mels[:, 1:]
        num_frames = int(x["num_frames"]) - 1

        div = num_frames // self.seq_len
        re = num_frames % self.seq_len
        pose_motion_pred_list = [
            torch.zeros(
                batch["ref"].unsqueeze(1).shape,
                dtype=batch["ref"].dtype,
                device=batch["ref"].device,
            )
        ]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch["z"] = z
            audio_emb = self.audio_encoder(
                indiv_mels_use[:, i * self.seq_len:(i + 1) * self.seq_len, :, :, :]
            )
            batch["audio_emb"] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch["pose_motion_pred"])

        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch["z"] = z
            audio_emb = self.audio_encoder(
                indiv_mels_use[:, -1 * self.seq_len:, :, :, :]
            )
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len - audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1)
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1)
            batch["audio_emb"] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch["pose_motion_pred"][:, -1 * re:, :])

        pose_motion_pred = torch.cat(pose_motion_pred_list, dim=1)
        batch["pose_motion_pred"] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred
        batch["pose_pred"] = pose_pred
        return batch
