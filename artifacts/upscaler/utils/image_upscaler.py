import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

COMPRESSION_SETTINGS = {
    "high_quality": {"png_compress": 1, "jpg_quality": 95, "webp_quality": 95},
    "balanced":     {"png_compress": 6, "jpg_quality": 85, "webp_quality": 80},
    "small_size":   {"png_compress": 9, "jpg_quality": 60, "webp_quality": 60},
}


def upscale_image(
    input_path: str,
    output_path: str,
    scale: int,
    fmt: str,
    compression: str,
) -> str:
    from .models_manager import get_upscaler

    logger.info(f"Upscaling image: {input_path} → {scale}x, fmt={fmt}, compression={compression}")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4

    if has_alpha:
        alpha = img[:, :, 3]
        img_bgr = img[:, :, :3]
    else:
        img_bgr = img
        alpha = None

    upscaler = get_upscaler(scale)
    output_bgr, _ = upscaler.enhance(img_bgr, outscale=scale)

    settings = COMPRESSION_SETTINGS.get(compression.lower(), COMPRESSION_SETTINGS["balanced"])
    fmt_upper = fmt.upper()

    if has_alpha and alpha is not None:
        alpha_up = cv2.resize(
            alpha,
            (output_bgr.shape[1], output_bgr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if fmt_upper == "PNG":
        if has_alpha:
            out_rgba = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGBA)
            out_rgba[:, :, 3] = alpha_up
            pil_img = Image.fromarray(out_rgba)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        pil_img.save(output_path, format="PNG", compress_level=settings["png_compress"])

    elif fmt_upper in ("JPG", "JPEG"):
        pil_img = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        pil_img.save(output_path, format="JPEG", quality=settings["jpg_quality"], optimize=True)

    elif fmt_upper == "WEBP":
        if has_alpha:
            out_rgba = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGBA)
            out_rgba[:, :, 3] = alpha_up
            pil_img = Image.fromarray(out_rgba)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        pil_img.save(output_path, format="WEBP", quality=settings["webp_quality"])

    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    logger.info(f"Image upscaled → {output_path}")
    return output_path
