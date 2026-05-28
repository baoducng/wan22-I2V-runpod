import runpod
import uuid
import os
import base64
import logging
import time
import shutil
import torch
import requests
from pathlib import Path
from utils.video import load_pipe, generate_video

logging.basicConfig(level=logging.INFO)

SEED = 42

# Load pipeline once at worker startup (cached globally in video.py)
load_pipe()


def handler(event):
    workdir = None
    try:
        inp = event["input"]

        # ---- Image input ----
        workdir = Path("/tmp") / str(uuid.uuid4())
        workdir.mkdir(parents=True, exist_ok=True)
        input_img = workdir / "input.jpg"

        if inp.get("image_base64"):
            raw = inp["image_base64"]
            if "," in raw:
                raw = raw.split(",", 1)[1]
            with open(input_img, "wb") as f:
                f.write(base64.b64decode(raw))
        elif inp.get("image_url"):
            r = requests.get(inp["image_url"], timeout=30)
            r.raise_for_status()
            with open(input_img, "wb") as f:
                f.write(r.content)
        else:
            return {"error": "Must provide image_base64 or image_url"}

        # ---- Prompt ----
        prompt = inp.get("prompt") or ""
        # prompts array: chained multi-clip; web app always sends single prompt
        prompts = inp.get("prompts") or [prompt]
        if not prompts or not prompts[0]:
            return {"error": "Must provide prompt or prompts"}

        # ---- Generation params ----
        seed = inp.get("seed", SEED)
        steps = inp.get("steps", 25)
        guidance_scale = float(inp.get("cfg", 5.0))
        negative_prompt = inp.get("negative_prompt")

        # length (frames) takes precedence over clip_sec
        num_frames = int(inp["length"]) if "length" in inp else None
        clip_sec = float(inp.get("clip_sec", 5.0))

        # ---- Generate ----
        start = time.time()
        video_path = workdir / "output.mp4"

        generate_video(
            image_path=str(input_img),
            prompt=prompts[0],
            output_path=str(video_path),
            duration_sec=clip_sec,
            num_frames_override=num_frames,
            steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale,
            negative_prompt=negative_prompt,
        )

        logging.info(f"⏱️ Generation time: {time.time() - start:.2f}s")

        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        return {"video_base64": video_b64}

    except Exception as e:
        logging.exception("❌ Generation failed")
        return {"error": str(e)}

    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
