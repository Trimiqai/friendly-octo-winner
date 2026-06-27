# Real-ESRGAN Upscaler API

Production-ready AI image and video upscaling using the official [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) models.

## Quick Start

The server auto-clones the Real-ESRGAN repo and downloads model weights on first startup.

### 1. Generate an API key

```bash
curl -X POST http://your-host/api/v1/generate-api-key?name=my-key
```

Response:
```json
{
  "success": true,
  "api_key": "resr_...",
  "daily_limit": 100
}
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/metrics` | CPU/RAM/disk/GPU stats |
| POST | `/api/v1/generate-api-key` | Generate API key |
| GET | `/api/v1/api-keys` | List all API keys |
| DELETE | `/api/v1/api-keys/{key}` | Delete API key |
| POST | `/api/v1/api-keys/{key}/disable` | Disable key |
| POST | `/api/v1/api-keys/{key}/enable` | Enable key |
| GET | `/api/v1/api-keys/{key}/usage` | Usage stats |
| POST | `/api/v1/upscale-image` | Upscale an image |
| POST | `/api/v1/upscale-video` | Upscale a video (async) |
| GET | `/api/v1/status/{job_id}` | Check job status |
| GET | `/api/v1/download/{job_id}` | Download result |

---

## Image Upscaling

```bash
curl -X POST http://your-host/api/v1/upscale-image \
  -H "X-API-Key: resr_YOUR_KEY" \
  -F "file=@photo.jpg" \
  -F "scale=4" \
  -F "output_format=PNG" \
  -F "compression=balanced"
```

**Parameters:**
- `scale`: `2` or `4`
- `output_format`: `PNG`, `JPG`, `WEBP`
- `compression`: `high_quality`, `balanced`, `small_size`

---

## Video Upscaling

```bash
curl -X POST http://your-host/api/v1/upscale-video \
  -H "X-API-Key: resr_YOUR_KEY" \
  -F "file=@clip.mp4" \
  -F "scale=2" \
  -F "compression=balanced"
```

Returns immediately with a `job_id`. Poll `/api/v1/status/{job_id}` for progress.

**Status values:** `queued` → `extracting_frames` → `upscaling` → `merging_video` → `completed`

---

## Language Examples

### Python

```python
import requests

API_KEY = "resr_YOUR_KEY"
BASE = "http://your-host"

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        f"{BASE}/api/v1/upscale-image",
        headers={"X-API-Key": API_KEY},
        files={"file": f},
        data={"scale": 4, "output_format": "PNG", "compression": "balanced"},
    )
job = resp.json()
print(job)
```

### JavaScript / Node.js

```js
const FormData = require("form-data");
const fs = require("fs");
const axios = require("axios");

const form = new FormData();
form.append("file", fs.createReadStream("photo.jpg"));
form.append("scale", "4");
form.append("output_format", "PNG");
form.append("compression", "balanced");

const res = await axios.post("http://your-host/api/v1/upscale-image", form, {
  headers: { ...form.getHeaders(), "X-API-Key": "resr_YOUR_KEY" },
});
console.log(res.data);
```

### cURL

```bash
curl -X POST http://your-host/api/v1/upscale-image \
  -H "X-API-Key: resr_YOUR_KEY" \
  -F "file=@image.png" \
  -F "scale=4" \
  -F "output_format=PNG" \
  -F "compression=high_quality"
```

### Telegram Bot (Python)

```python
import requests

async def upscale_telegram_photo(file_path: str, api_key: str, base_url: str) -> bytes:
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/api/v1/upscale-image",
            headers={"X-API-Key": api_key},
            files={"file": f},
            data={"scale": 4, "output_format": "PNG", "compression": "balanced"},
        )
    job = resp.json()
    # poll until done
    import time
    for _ in range(120):
        time.sleep(5)
        s = requests.get(
            f"{base_url}/api/v1/status/{job['job_id']}",
            headers={"X-API-Key": api_key},
        ).json()
        if s["status"] == "completed":
            dl = requests.get(
                f"{base_url}/api/v1/download/{job['job_id']}",
                headers={"X-API-Key": api_key},
            )
            return dl.content
    raise TimeoutError("Job timed out")
```

---

## Rate Limiting

- Default: **100 requests per API key per day**
- HTTP `429` returned when exceeded
- Check usage: `GET /api/v1/api-keys/{key}/usage`

---

## Auth

Every request (except key generation) requires:
```
X-API-Key: resr_YOUR_KEY
```
Returns HTTP `401` if missing or invalid.
