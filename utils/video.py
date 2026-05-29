import gc
import torch
import numpy as np
from PIL import Image
from typing import Optional

from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video

from torchao.quantization import quantize_
from torchao.quantization import Int8WeightOnlyConfig


# =========================
# Global config
# =========================
import os as _os
_LOCAL_MODEL = "/app/models/Wan2.2-I2V-A14B-Diffusers"
MODEL_ID = _LOCAL_MODEL if _os.path.isdir(_LOCAL_MODEL) else "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DEVICE = "cuda"

FIXED_FPS = 16
MAX_DIM = 1280
MIN_DIM = 480
MULTIPLE_OF = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 7720

# Wan2.2-A14B native generation area (720p). Cost is driven by pixel-area, so we
# always target this area regardless of input size — small images are upscaled,
# large ones downscaled — to keep the model at its trained resolution.
TARGET_AREA = 1280 * 720  # 921,600 px

# MP4 export quality (0-10, diffusers default 5). 9 keeps the encoder from
# softening the model output before Facebook re-compresses it on upload.
EXPORT_QUALITY = 9

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, motion artifacts, unstable motion, jitter, frame jitter, "
    "wobbling limbs, motion distortion, inconsistent movement, robotic movement, "
    "animation-like motion, awkward transitions, incorrect body mechanics, unnatural posing, "
    "off-balance poses, broken motion paths, frozen frames, duplicated frames, frame skipping, "
    "warped motion, stretching artifacts bad anatomy, incorrect proportions, deformed body, "
    "twisted torso, broken joints, dislocated limbs, distorted neck, unnatural spine curvature, "
    "malformed hands, extra fingers, missing fingers, fused fingers, distorted legs, extra limbs, "
    "collapsed feet, floating feet, foot sliding, foot jitter, backward walking, unnatural gait "
    "blurry details, ghosting, compression noise, jpeg artifacts, cartoon texture"
)

# =========================
# Global pipeline handle
# =========================
_PIPE = None


# =========================
# Utilities
# =========================
def get_num_frames(duration_seconds: float) -> int:
    frames = int(round(duration_seconds * FIXED_FPS))
    return int(np.clip(frames + 1, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))


def crop_to_aspect(image: Image.Image, aspect_w: int, aspect_h: int) -> Image.Image:
    """Center-crop `image` to the given aspect ratio (e.g. 9:16 for vertical).

    No-op when the image already matches the target ratio, so a pre-cropped
    upload is passed through untouched.
    """
    w, h = image.size
    target = aspect_w / aspect_h
    current = w / h
    if abs(current - target) < 1e-3:
        return image
    if current > target:
        # too wide -> trim the sides
        new_w = int(round(h * target))
        x0 = (w - new_w) // 2
        return image.crop((x0, 0, x0 + new_w, h))
    # too tall -> trim top/bottom
    new_h = int(round(w / target))
    y0 = (h - new_h) // 2
    return image.crop((0, y0, w, y0 + new_h))


def resize_image(image: Image.Image) -> Image.Image:
    """Resize to the model's native 720p pixel-area, preserving aspect ratio.

    Targets a fixed area (up- or down-scaling as needed) so small uploads still
    run at the model's trained resolution instead of below it. Dimensions are
    snapped to a multiple of 16 and floored at MIN_DIM.
    """
    w, h = image.size
    aspect = h / w
    height = int(round((TARGET_AREA * aspect) ** 0.5))
    width = int(round((TARGET_AREA / aspect) ** 0.5))
    width = max(MIN_DIM, (width // MULTIPLE_OF) * MULTIPLE_OF)
    height = max(MIN_DIM, (height // MULTIPLE_OF) * MULTIPLE_OF)
    return image.resize((width, height), Image.LANCZOS)


# =========================
# Pipeline loader (ONE TIME)
# =========================
def load_pipe():
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    torch.cuda.empty_cache()
    gc.collect()

    print("🚀 Loading Wan 2.2 transformers...")

    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    quantize_(transformer, Int8WeightOnlyConfig())

    transformer_2 = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
    )
    quantize_(transformer_2, Int8WeightOnlyConfig())

    print("🔒 Applying Int8 quantization to text encoder...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        transformer_2=transformer_2,
        torch_dtype=torch.bfloat16,
    )
    quantize_(pipe.text_encoder, Int8WeightOnlyConfig())

    pipe.to(DEVICE)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    _PIPE = pipe
    print("✅ Pipeline ready")

    return _PIPE


# =========================
# Video generation
# =========================
def generate_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_sec: float = 5.0,
    num_frames_override: Optional[int] = None,
    steps: int = 40,
    seed: int = 42,
    guidance_scale: float = 5.0,
    guidance_scale_2: float = 6.0,
    negative_prompt: Optional[str] = None,
    orientation: Optional[str] = None,
):
    pipe = load_pipe()

    torch.cuda.empty_cache()
    gc.collect()

    image = Image.open(image_path).convert("RGB")

    # Force a 9:16 frame for vertical (Facebook/Reels) by center-cropping the
    # source *before* generation — every generated pixel then lands in the final
    # video at full resolution. A pre-cropped 9:16 upload passes through as-is.
    if orientation == "vertical":
        image = crop_to_aspect(image, 9, 16)
    elif orientation == "horizontal":
        image = crop_to_aspect(image, 16, 9)

    image = resize_image(image)

    if num_frames_override is not None:
        num_frames = int(np.clip(num_frames_override, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
    else:
        num_frames = get_num_frames(duration_sec)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    out = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        height=image.height,
        width=image.width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        generator=generator,
    )

    frames = out.frames[0]
    del out
    torch.cuda.empty_cache()

    export_to_video(frames, output_path, fps=FIXED_FPS, quality=EXPORT_QUALITY)
    return output_path
