# Wan2.2 I2V – RunPod Serverless (Diffusers, base model)

[![Deploy on RunPod](https://runpod.io/button.svg)](https://runpod.io/console/hub/baoducng/wan22-I2V-runpod)

Serverless RunPod worker for **Wan2.2 Image-to-Video (A14B)** using the Diffusers library.
Accepts an image (base64 or URL) + prompt, returns `video_base64`.

---

## Input

```json
{
  "input": {
    "image_base64": "<base64-encoded image>",
    "prompt": "gentle breeze, natural light",
    "steps": 25,
    "cfg": 5.0,
    "length": 81,
    "seed": 42
  }
}
```

Or use `image_url` instead of `image_base64`:

```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "prompt": "gentle breeze, natural light"
  }
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `image_base64` | string | — | Base64 image (with or without data URI prefix) |
| `image_url` | string | — | HTTP/HTTPS URL to image |
| `prompt` | string | required | Text description of desired motion |
| `negative_prompt` | string | built-in | Override default negative prompt |
| `steps` | int | 25 | Inference steps |
| `cfg` | float | 5.0 | Guidance scale (applied to both transformers) |
| `length` | int | 81 | Number of frames (81 ≈ 5 s at 16 fps) |
| `seed` | int | 42 | Random seed |
| `clip_sec` | float | 5.0 | Duration in seconds (ignored if `length` is set) |

## Output

```json
{
  "video_base64": "<base64-encoded mp4>"
}
```

---

## Model

- **Wan-AI/Wan2.2-I2V-A14B-Diffusers** (≈28 GB fp16)
- Loaded from `/runpod-volume/models/Wan2.2-I2V-A14B-Diffusers`
- No LoRA — base model only

## Requirements

- GPU: RTX 4090 (24 GB) or A100 (40/80 GB)
- Persistent volume with model pre-downloaded at `/runpod-volume`
- CUDA 12.8

## Repository Structure

```
handler.py          # RunPod serverless handler (entrypoint)
preload_model.py    # One-time model downloader script
requirements.txt
Dockerfile
utils/
  video.py          # Pipeline load + generate_video()
  s3.py             # Unused (kept for reference)
  utllity.py        # Frame extraction helper
.runpod/
  hub.json
  tests.json
```

## Pre-downloading the model

If the volume is empty, run once on a GPU pod:

```bash
python preload_model.py
```

This downloads only the required Wan2.2 model files (~28 GB) to `/runpod-volume/models/Wan2.2-I2V-A14B-Diffusers`.
