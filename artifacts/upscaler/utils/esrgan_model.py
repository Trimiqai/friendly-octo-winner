"""
Self-contained Real-ESRGAN model implementation using only PyTorch.
Architecture matches the official Real-ESRGAN repo exactly.
Weights loaded directly from official pretrained .pth files.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────

def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    out_channel = c * (scale ** 2)
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, out_channel, h // scale, w // scale)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture as used in Real-ESRGAN."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch
        if scale == 2:
            in_ch = num_in_ch * 4
        elif scale == 1:
            in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─────────────────────────────────────────────
# Inference wrapper with tiling
# ─────────────────────────────────────────────

class RealESRGANer:
    """
    Inference wrapper for Real-ESRGAN models.
    Supports tiled processing to handle large images without OOM.
    """

    def __init__(
        self,
        scale: int,
        model_path: str,
        model: RRDBNet,
        tile: int = 400,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False,
        device: str = "cpu",
    ):
        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.device = device

        state_dict = torch.load(model_path, map_location=torch.device(device))
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model.to(device)
        if half:
            self.model = self.model.half()
        logger.info(f"Loaded RealESRGANer scale={scale} on {device}")

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()  # BGR → RGB, copy to remove negative stride
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()
        return img

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        out = t.squeeze().float().cpu().clamp_(0, 1).numpy()
        out = np.transpose(out, (1, 2, 0))  # CHW → HWC
        out = out[:, :, ::-1]               # RGB → BGR
        return (out * 255.0).round().astype(np.uint8)

    @torch.no_grad()
    def _process_tile(self, tile_t: torch.Tensor) -> torch.Tensor:
        return self.model(tile_t)

    @torch.no_grad()
    def enhance(self, img: np.ndarray, outscale: int = None):
        h_in, w_in = img.shape[:2]
        if self.tile > 0:
            output = self._tile_process(img)
        else:
            img_t = self._to_tensor(img)
            output_t = self.model(img_t)
            output = self._to_numpy(output_t)

        if outscale is not None and outscale != self.scale:
            output = cv2.resize(
                output,
                (int(w_in * outscale), int(h_in * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output, None

    def _tile_process(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        tile = min(self.tile, h, w)
        tile_pad = self.tile_pad
        stride = tile - tile_pad * 2

        h_scaled = math.ceil(h * self.scale)
        w_scaled = math.ceil(w * self.scale)
        output = np.zeros((h_scaled, w_scaled, 3), dtype=np.float32)
        weight = np.zeros((h_scaled, w_scaled, 1), dtype=np.float32)

        tiles_h = math.ceil(h / stride)
        tiles_w = math.ceil(w / stride)
        total_tiles = tiles_h * tiles_w
        processed = 0

        for row in range(tiles_h):
            for col in range(tiles_w):
                x0 = max(0, col * stride - tile_pad)
                y0 = max(0, row * stride - tile_pad)
                x1 = min(w, x0 + tile)
                y1 = min(h, y0 + tile)

                tile_img = img[y0:y1, x0:x1]
                tile_t = self._to_tensor(tile_img)

                with torch.no_grad():
                    tile_out_t = self.model(tile_t)
                tile_out = self._to_numpy(tile_out_t).astype(np.float32)

                ox0 = x0 * self.scale
                oy0 = y0 * self.scale
                ox1 = ox0 + tile_out.shape[1]
                oy1 = oy0 + tile_out.shape[0]

                ox1 = min(ox1, w_scaled)
                oy1 = min(oy1, h_scaled)
                cw = ox1 - ox0
                ch = oy1 - oy0

                output[oy0:oy1, ox0:ox1] += tile_out[:ch, :cw]
                weight[oy0:oy1, ox0:ox1] += 1.0

                processed += 1
                if processed % 5 == 0 or processed == total_tiles:
                    logger.debug(f"  Tile {processed}/{total_tiles}")

        weight = np.clip(weight, 1e-8, None)
        output /= weight
        return output.clip(0, 255).astype(np.uint8)
