---
name: Real-ESRGAN PyTorch inference
description: How to run Real-ESRGAN without basicsr/realesrgan PyPI packages (disk quota issue on Replit).
---

## Rule
Do not install `basicsr` or `realesrgan` via pip in this project. They pull in CUDA torch (~400 MB), exceeding the disk quota.

**Why:** `installLanguagePackages` uses uv but packages land in `.pythonlibs`. When pip resolves `basicsr`, it tries to reinstall torch with CUDA wheels (nvidia_cublas = 423 MB) even if CPU torch is already present, causing `Disk quota exceeded`.

## How to apply
Use `artifacts/upscaler/utils/esrgan_model.py` — a self-contained RRDBNet + RealESRGANer implementation in pure PyTorch that loads the official `.pth` weight files directly.

## Critical fix
`img[:, :, ::-1]` (BGR→RGB) creates a numpy view with negative strides. PyTorch rejects these.  
Always call `.copy()` immediately after: `img[:, :, ::-1].copy()`.

## Python environment
Workflow Python: `/home/runner/workspace/.pythonlibs/bin/python3`  
Packages live in: `/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages/`  
Do NOT use bare `pip3` — use `python3 -m pip` or `installLanguagePackages`.
