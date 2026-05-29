# Frontend Handoff — Wan2.2 I2V Diffusers Endpoint

## New endpoint

| Field | Value |
|-------|-------|
| **Endpoint ID** | `d10xmm1c7cldff` |
| **API base** | `https://api.runpod.ai/v2/d10xmm1c7cldff` |
| **Hub page** | `https://console.runpod.io/serverless/user/endpoint/d10xmm1c7cldff` |
| **Model** | Wan2.2 I2V A14B — base model, no LoRA |
| **GPU** | 48 GB / 48 GB Pro / 80 GB |

---

## Required change in `lib/runpod/constants.ts`

```diff
- export const DEFAULT_ENDPOINT_ID = "q48ouwd6uk7671";
+ export const DEFAULT_ENDPOINT_ID = "d10xmm1c7cldff";

  export const DEFAULTS = {
    negative_prompt: "blurry, low quality, distorted, static, watermark, text",
    seed: 42,
-   cfg: 3.0,
-   steps: 14,
+   cfg: 5.0,
+   cfg2: 6.0,
+   steps: 40,
    width: 720,
    height: 1280,
    length: 81,
  } as const;
```

`cfg` and `steps` change because the old endpoint used Lightning LoRA (distilled, low-step) — the new one is the base model, which wants more steps and full guidance. `cfg2` is the low-noise (detail) expert of the A14B two-expert design; leaving it higher than `cfg` sharpens fine detail. These can be omitted — the handler defaults to `steps: 40`, `cfg: 5.0`, `cfg2: 6.0`.

---

## Input/output contract

### Input (unchanged shape — extra fields are silently ignored)

```ts
{
  image_base64?: string;   // base64 PNG/JPG, with or without data URI prefix
  image_url?: string;      // HTTP/HTTPS URL (one of the two required)
  prompt: string;
  negative_prompt?: string;
  steps?: number;          // default 40 (base model, non-distilled)
  cfg?: number;            // default 5.0  — high-noise (layout) expert
  cfg2?: number;           // default 6.0  — low-noise (detail) expert
  orientation?: "vertical" | "horizontal";  // omit = keep image's native aspect
  length?: number;         // frames (81 ≈ 5 s at 16 fps)
  clip_sec?: number;       // alternative to length; ignored when length is set
  seed?: number;
}
```

> `width` and `height` are accepted but **ignored** — the handler resizes the
> image to the model's native 720p area (~921,600 px), preserving aspect ratio
> and snapping to multiples of 16. Small uploads are upscaled to this area so
> they still generate at full resolution.
>
> `orientation: "vertical"` produces a 9:16 frame (720×1280) for Facebook /
> Reels by **center-cropping the source image before generation** — no
> letterboxing, full-resolution output. If the upload is already 9:16 it passes
> through uncropped. Same GPU cost as horizontal (identical pixel area).
> `"horizontal"` does the same for 16:9. Omit to keep the image's native aspect.
>
> `lora_pairs` is ignored — this endpoint has no LoRA.

### Output

```ts
{ video_base64: string }   // base64-encoded MP4
// or on error:
{ error: string }
```

`client.ts → streamVideo()` already checks `["video", "video_base64", ...]`
in order, so `video_base64` is picked up correctly with no code change.

---

## Cold start

First request on a cold worker downloads ~28 GB from HuggingFace (~80 s)
then loads the model (~3 min total). Subsequent requests on a warm worker
are fast. The frontend should show a loading state for up to **5 minutes**
on cold start.

To eliminate cold-start download: set env var `HF_HOME=/runpod-volume/hf_cache`
on the endpoint and attach a Network Volume — the model will be cached after
the first run.

---

## No other code changes needed

`client.ts`, `types.ts` — no changes required. The output key, auth flow,
polling pattern, and error types are identical.
