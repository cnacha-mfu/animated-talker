"""Face renderer components from SadTalker's face-vid2vid pipeline.

Contains all building blocks for the face animation renderer:
- Utility blocks (ResBlock2d/3d, UpBlock, DownBlock, SameBlock, Hourglass, etc.)
- KPDetector: canonical keypoint detector
- HEEstimator: head pose + expression estimator
- DenseMotionNetwork: sparse-to-dense motion estimation
- OcclusionAwareSPADEGenerator: image generator with occlusion handling
- MappingNet: 3DMM coefficients → keypoint parameters

Reference: https://github.com/OpenTalker/SadTalker
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def kp2gaussian(kp: dict, spatial_size: tuple, kp_variance: float) -> torch.Tensor:
    """Transform keypoints into gaussian-like heatmap representation."""
    mean = kp["value"]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size: tuple, type: torch.dtype) -> torch.Tensor:
    """Create a meshgrid [-1, 1] of given 3D spatial size."""
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    z = 2 * (z / (d - 1)) - 1
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    return meshed


# ---------------------------------------------------------------------------
# 2D building blocks
# ---------------------------------------------------------------------------

class ResBottleneck(nn.Module):
    def __init__(self, in_features: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_features // 4, in_features // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_features // 4, in_features, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_features // 4, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features // 4, affine=True)
        self.norm3 = nn.BatchNorm2d(in_features, affine=True)
        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_features, in_features, kernel_size=1, stride=stride)
            self.norm4 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.stride != 1:
            x = self.norm4(self.skip(x))
        out += x
        return F.relu(out)


class ResBlock2d(nn.Module):
    def __init__(self, in_features: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    def __init__(self, in_features: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_features, in_features, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int = 3, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        return F.relu(out)


class UpBlock3d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int = 3, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        return F.relu(out)


class DownBlock2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int = 3, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return self.pool(out)


class DownBlock3d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int = 3, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return self.pool(out)


class SameBlock2d(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int = 1,
                 kernel_size: int = 3, padding: int = 1, lrelu: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.ac = nn.LeakyReLU() if lrelu else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        return self.ac(out)


# ---------------------------------------------------------------------------
# Hourglass encoder-decoder (3D)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, block_expansion: int, in_features: int,
                 num_blocks: int = 3, max_features: int = 256):
        super().__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(
                in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))),
                kernel_size=3, padding=1,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    def __init__(self, block_expansion: int, in_features: int,
                 num_blocks: int = 3, max_features: int = 256):
        super().__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
        self.conv = nn.Conv3d(self.out_filters, self.out_filters, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        return F.relu(out)


class Hourglass(nn.Module):
    def __init__(self, block_expansion: int, in_features: int,
                 num_blocks: int = 3, max_features: int = 256):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class KPHourglass(nn.Module):
    def __init__(self, block_expansion: int, in_features: int,
                 reshape_features: int, reshape_depth: int,
                 num_blocks: int = 3, max_features: int = 256):
        super().__init__()
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module(
                f"down{i}",
                DownBlock2d(
                    in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3, padding=1,
                ),
            )
        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_filters, reshape_features, kernel_size=1)
        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_f = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_f = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module(f"up{i}", UpBlock3d(in_f, out_f, kernel_size=3, padding=1))
        self.reshape_depth = reshape_depth
        self.out_filters = out_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c // self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)
        return out


class AntiAliasInterpolation2d(nn.Module):
    """Band-limited downsampling for better signal preservation."""

    def __init__(self, channels: int, scale: float):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        kernel = torch.ones(1)
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale
        self.int_inv_scale = int(1 / scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return input
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]
        return out


# ---------------------------------------------------------------------------
# SPADE normalization for the SPADE generator
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    def __init__(self, norm_nc: int, label_nc: int):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin: int, fout: int, norm_G: str, label_nc: int,
                 use_se: bool = False, dilation: int = 1):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if "spectral" in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x: torch.Tensor, seg1: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        return x_s + dx

    def shortcut(self, x: torch.Tensor, seg1: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg1))
        return x

    def actvn(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 2e-1)


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = "spadespectralinstance"
        label_nc = 256
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# KPDetector
# ---------------------------------------------------------------------------

class KPDetector(nn.Module):
    """Detect canonical 3D keypoints from source image."""

    def __init__(self, block_expansion: int, feature_channel: int,
                 num_kp: int, image_channel: int, max_features: int,
                 reshape_channel: int, reshape_depth: int,
                 num_blocks: int, temperature: float,
                 estimate_jacobian: bool = False,
                 scale_factor: float = 1.0,
                 single_jacobian_map: bool = False):
        super().__init__()
        self.predictor = KPHourglass(
            block_expansion, in_features=image_channel,
            max_features=max_features, reshape_features=reshape_channel,
            reshape_depth=reshape_depth, num_blocks=num_blocks,
        )
        self.kp = nn.Conv3d(self.predictor.out_filters, num_kp, kernel_size=3, padding=1)
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv3d(
                self.predictor.out_filters, 9 * self.num_jacobian_maps,
                kernel_size=3, padding=1,
            )
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float)
            )
        else:
            self.jacobian = None
        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap: torch.Tensor) -> dict:
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        return {"value": value}

    def forward(self, x: torch.Tensor) -> dict:
        if self.scale_factor != 1:
            x = self.down(x)
        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        out = self.gaussian2kp(heatmap)
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(
                final_shape[0], self.num_jacobian_maps, 9,
                final_shape[2], final_shape[3], final_shape[4],
            )
            heatmap = heatmap.unsqueeze(2)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out["jacobian"] = jacobian
        return out


# ---------------------------------------------------------------------------
# HEEstimator
# ---------------------------------------------------------------------------

class HEEstimator(nn.Module):
    """Estimate head pose (yaw/pitch/roll as bin distributions) and expression."""

    def __init__(self, block_expansion: int, feature_channel: int,
                 num_kp: int, image_channel: int, max_features: int,
                 num_bins: int = 66, estimate_jacobian: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channel, block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = nn.BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(block_expansion, 256, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(256, affine=True)
        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module(f"b1_{i}", ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(256, 512, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)
        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module(f"b3_{i}", ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(512, 1024, kernel_size=1)
        self.norm4 = nn.BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)
        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module(f"b5_{i}", ResBottleneck(in_features=1024, stride=1))

        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.norm5 = nn.BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)
        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module(f"b7_{i}", ResBottleneck(in_features=2048, stride=1))

        self.fc_roll = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)
        self.fc_t = nn.Linear(2048, 3)
        self.fc_exp = nn.Linear(2048, 3 * num_kp)

    def forward(self, x: torch.Tensor) -> dict:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.maxpool(out)
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.block1(out)
        out = F.relu(self.norm3(self.conv3(out)))
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.norm4(self.conv4(out)))
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.norm5(self.conv5(out)))
        out = self.block6(out)
        out = self.block7(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)
        # Note: SadTalker's original code swaps yaw/roll naming here
        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)
        return {"yaw": yaw, "pitch": pitch, "roll": roll, "t": t, "exp": exp}


# ---------------------------------------------------------------------------
# DenseMotionNetwork
# ---------------------------------------------------------------------------

class DenseMotionNetwork(nn.Module):
    """Predict dense optical flow from sparse keypoint motion."""

    def __init__(self, block_expansion: int, num_blocks: int,
                 max_features: int, num_kp: int, feature_channel: int,
                 reshape_depth: int, compress: int,
                 estimate_occlusion_map: bool = False):
        super().__init__()
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_kp + 1) * (compress + 1),
            max_features=max_features, num_blocks=num_blocks,
        )
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = nn.BatchNorm3d(compress, affine=True)
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(
                self.hourglass.out_filters * reshape_depth, 1, kernel_size=7, padding=3,
            )
        else:
            self.occlusion = None
        self.num_kp = num_kp

    def create_sparse_motions(self, feature: torch.Tensor,
                              kp_driving: dict, kp_source: dict) -> torch.Tensor:
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source["value"].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving["value"].view(bs, self.num_kp, 1, 1, 1, 3)
        if "jacobian" in kp_driving and kp_driving["jacobian"] is not None:
            jacobian = torch.matmul(kp_source["jacobian"], torch.inverse(kp_driving["jacobian"]))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
        driving_to_source = coordinate_grid + kp_source["value"].view(bs, self.num_kp, 1, 1, 1, 3)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_feature(self, feature: torch.Tensor,
                                sparse_motions: torch.Tensor) -> torch.Tensor:
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1, 1)
        feature_repeat = feature_repeat.view(bs * (self.num_kp + 1), -1, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), d, h, w, -1))
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, d, h, w))
        return sparse_deformed

    def create_heatmap_representations(self, feature: torch.Tensor,
                                       kp_driving: dict, kp_source: dict) -> torch.Tensor:
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def forward(self, feature: torch.Tensor,
                kp_driving: dict, kp_source: dict) -> dict:
        bs, _, d, h, w = feature.shape
        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)
        out_dict: dict = {}
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)
        input_ = torch.cat([heatmap, deformed_feature], dim=2)
        input_ = input_.view(bs, -1, d, h, w)
        prediction = self.hourglass(input_)
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict["mask"] = mask
        mask = mask.unsqueeze(2)
        zeros_mask = torch.zeros_like(mask)
        mask = torch.where(mask < 1e-3, zeros_mask, mask)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        out_dict["deformation"] = deformation
        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict["occlusion_map"] = occlusion_map
        return out_dict


# ---------------------------------------------------------------------------
# OcclusionAwareSPADEGenerator
# ---------------------------------------------------------------------------

class OcclusionAwareSPADEGenerator(nn.Module):
    """Face image generator with SPADE decoder and occlusion-aware warping."""

    def __init__(self, image_channel: int, feature_channel: int,
                 num_kp: int, block_expansion: int, max_features: int,
                 num_down_blocks: int, reshape_channel: int,
                 reshape_depth: int, num_resblocks: int,
                 estimate_occlusion_map: bool = False,
                 dense_motion_params: dict | None = None,
                 estimate_jacobian: bool = False):
        super().__init__()
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp, feature_channel=feature_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params,
            )
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(out_features, max_features, kernel_size=1, stride=1)
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module(f"3dr{i}", ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** num_down_blocks)
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(out_features, out_features, kernel_size=1, stride=1)
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel
        self.decoder = SPADEDecoder()

    def deform_input(self, inp: torch.Tensor, deformation: torch.Tensor) -> torch.Tensor:
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode="trilinear")
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image: torch.Tensor,
                kp_driving: dict, kp_source: dict) -> dict:
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)
        feature_3d = self.resblocks_3d(feature_3d)

        output_dict: dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source,
            )
            output_dict["mask"] = dense_motion["mask"]
            if "occlusion_map" in dense_motion:
                occlusion_map = dense_motion["occlusion_map"]
                output_dict["occlusion_map"] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion["deformation"]
            out = self.deform_input(feature_3d, deformation)
            bs, c, d, h, w = out.shape
            out = out.view(bs, c * d, h, w)
            out = self.third(out)
            out = self.fourth(out)
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode="bilinear")
                out = out * occlusion_map

        out = self.decoder(out)
        output_dict["prediction"] = out
        return output_dict


# ---------------------------------------------------------------------------
# MappingNet
# ---------------------------------------------------------------------------

class MappingNet(nn.Module):
    """Map 3DMM coefficients to head pose bins, translation, and expression."""

    def __init__(self, coeff_nc: int, descriptor_nc: int,
                 layer: int, num_kp: int, num_bins: int):
        super().__init__()
        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(
            nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True),
        )
        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3),
            )
            setattr(self, f"encoder{i}", net)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc
        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3 * num_kp)

    def forward(self, input_3dmm: torch.Tensor) -> dict:
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, f"encoder{i}")
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)
        return {"yaw": yaw, "pitch": pitch, "roll": roll, "t": t, "exp": exp}
