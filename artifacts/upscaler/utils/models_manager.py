import os
import sys
import logging
import subprocess
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
REPO_DIR = BASE_DIR / "Real-ESRGAN"

REAL_ESRGAN_REPO = "https://github.com/xinntao/Real-ESRGAN.git"
REAL_ESRGAN_TAG = "v0.3.0"

MODELS = {
    "x4": {
        "name": "RealESRGAN_x4plus",
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_feat": 64,
        "num_block": 23,
    },
    "x2": {
        "name": "RealESRGAN_x2plus",
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_feat": 64,
        "num_block": 23,
    },
}

_upscalers: dict = {}


def clone_repo():
    """Clone the official Real-ESRGAN repo at the latest stable release."""
    if REPO_DIR.exists():
        logger.info("Real-ESRGAN repo already exists, skipping clone.")
        return
    logger.info(f"Cloning Real-ESRGAN {REAL_ESRGAN_TAG} from GitHub...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REAL_ESRGAN_TAG,
             REAL_ESRGAN_REPO, str(REPO_DIR)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("Real-ESRGAN cloned successfully.")
        else:
            logger.warning(f"Git clone warning: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not clone Real-ESRGAN repo: {e}. Continuing without it.")


def download_models():
    """Download official pretrained model weights if not already cached."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, info in MODELS.items():
        dest = MODELS_DIR / info["file"]
        if dest.exists():
            logger.info(f"Model {info['file']} already cached, skipping download.")
            continue
        logger.info(f"Downloading {info['file']} from official Real-ESRGAN releases...")
        try:
            resp = requests.get(info["url"], stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (10 * 1024 * 1024) < 65536:
                        logger.info(f"  {info['file']}: {downloaded / total * 100:.1f}%")
            logger.info(f"Downloaded {info['file']} ({downloaded / 1024 / 1024:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to download {info['file']}: {e}")
            if dest.exists():
                dest.unlink()


def load_upscalers():
    """Load pretrained models into memory using self-contained PyTorch implementation."""
    global _upscalers
    import torch
    from .esrgan_model import RRDBNet, RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Real-ESRGAN models on device: {device}")

    for key, info in MODELS.items():
        model_path = MODELS_DIR / info["file"]
        if not model_path.exists():
            logger.warning(f"Model {info['file']} not found, skipping {info['scale']}x upscaler.")
            continue

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=info["num_feat"],
            num_block=info["num_block"],
            num_grow_ch=32,
            scale=info["scale"],
        )

        upscaler = RealESRGANer(
            scale=info["scale"],
            model_path=str(model_path),
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )

        _upscalers[key] = upscaler
        logger.info(f"Loaded {info['name']} ({info['scale']}x upscaler).")


def get_upscaler(scale: int):
    key = f"x{scale}"
    if key not in _upscalers:
        raise RuntimeError(
            f"Upscaler for {scale}x is not available. "
            "Check that the model was downloaded successfully."
        )
    return _upscalers[key]


def setup():
    """Full setup: clone repo, download models, load into memory."""
    clone_repo()
    download_models()
    load_upscalers()
