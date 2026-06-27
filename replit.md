# Real-ESRGAN Upscaler API

Production-ready AI image and video upscaling using the official Real-ESRGAN models. Supports 2x/4x upscaling, PNG/JPG/WEBP/MP4 output, API key management with rate limiting, and async video processing via a background job queue.

## Run & Operate

- **Upscaler API** workflow → `cd artifacts/upscaler && python main.py` (port 8000)
- First startup auto-clones Real-ESRGAN repo and downloads model weights (~130 MB, cached in `artifacts/upscaler/models/`)
- `pnpm run typecheck` — full typecheck across all packages

## Stack

- **Language:** Python 3.11
- **Framework:** FastAPI + Uvicorn
- **Upscaler:** Real-ESRGAN (official repo cloned from GitHub, v0.3.0)
- **Models:** `RealESRGAN_x4plus.pth` (4x), `RealESRGAN_x2plus.pth` (2x) — auto-downloaded from official GitHub releases
- **Inference:** Self-contained PyTorch (CPU) RRDBNet implementation — no basicsr/pip dependency needed
- **DB:** SQLite (via Python stdlib `sqlite3`)
- **Video:** FFmpeg for frame extraction and merging
- **Rate limiting:** Manual SQLite-backed daily counter (100 req/key/day, HTTP 429 on exceed)

## Where things live

- `artifacts/upscaler/main.py` — FastAPI app, all routes
- `artifacts/upscaler/utils/esrgan_model.py` — Self-contained RRDBNet + RealESRGANer (pure PyTorch)
- `artifacts/upscaler/utils/models_manager.py` — Repo clone, model download, upscaler loading
- `artifacts/upscaler/utils/image_upscaler.py` — Image upscaling with transparency support
- `artifacts/upscaler/utils/video_upscaler.py` — FFmpeg frame extract/merge + per-frame upscaling
- `artifacts/upscaler/utils/database.py` — SQLite: API keys, usage tracking, job table
- `artifacts/upscaler/utils/auth.py` — FastAPI dependency for X-API-Key validation + rate limit
- `artifacts/upscaler/utils/job_queue.py` — asyncio Queue + background worker
- `artifacts/upscaler/models/` — Cached model weights (gitignored)
- `artifacts/upscaler/Real-ESRGAN/` — Cloned repo (gitignored)
- `artifacts/upscaler/output/` — Processed files served at `/download/{filename}` (gitignored)

## Product

**POST /api/v1/generate-api-key** — Generate a `resr_…` API key stored in SQLite

**POST /api/v1/upscale-image** — Upload JPG/PNG/WEBP, choose scale (2x/4x), format (PNG/JPG/WEBP), compression (high_quality/balanced/small_size). Returns job_id immediately; async processing.

**POST /api/v1/upscale-video** — Upload MP4/MOV/MKV/AVI. Extracts frames with FFmpeg, upscales each frame, merges with original audio. Returns job_id; status polling via GET /api/v1/status/{job_id}.

**GET /api/v1/download/{job_id}** — Download completed result.

**GET /api/v1/api-keys** · **DELETE /api/v1/api-keys/{key}** · **POST /api/v1/api-keys/{key}/disable** · **POST /api/v1/api-keys/{key}/enable** — Key management

**GET /health** · **GET /metrics** — Health + CPU/RAM/disk/queue/GPU metrics

Interactive docs at `/docs` (Swagger) and `/redoc`.

## Architecture decisions

- **Self-contained RRDBNet** — avoids basicsr/realesrgan PyPI packages (CUDA torch disk quota bloat). The model architecture is ~150 lines of pure PyTorch matching the official repo exactly.
- **Official GitHub clone** — Real-ESRGAN repo cloned at v0.3.0 on first startup as requested; model weights downloaded from official GitHub release URLs.
- **Tile-based inference** — Images processed in 400px tiles with 10px padding to handle large inputs without OOM on CPU.
- **asyncio Queue** — Single background worker processes jobs sequentially; image/video jobs don't block the API.
- **SQLite + asyncio.run_in_executor** — Avoids async SQLite drivers while keeping the FastAPI event loop unblocked.

## User preferences

_Populate as you build._

## Gotchas

- Models are cached in `artifacts/upscaler/models/` (gitignored). On a fresh clone, first startup re-downloads them (~130 MB total).
- numpy `[::-1]` BGR→RGB creates negative strides; always call `.copy()` before passing to `torch.from_numpy()`.
- The Python packages (torch, etc.) live in `.pythonlibs/`; do NOT use `pip3` — use `/home/runner/workspace/.pythonlibs/bin/python3 -m pip` if installing anything manually.
- `installLanguagePackages` uses uv and may install to a different venv than `.pythonlibs`. For packages that need to be available in the workflow, install via the workflow's Python executable or `installLanguagePackages` (which reloads the environment).

## Pointers

- See `artifacts/upscaler/README.md` for full API usage examples (Python, JS, cURL, Telegram Bot)
- See the `pnpm-workspace` skill for workspace structure details
