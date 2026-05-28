def ensure_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path

    WAN_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    BASE_MODEL_DIR = Path("/app/models/Wan2.2-I2V-A14B-Diffusers")
    WAN_SENTINEL = BASE_MODEL_DIR / "model_index.json"

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not WAN_SENTINEL.exists():
        print("⬇️ Downloading Wan 2.2 base model...")
        snapshot_download(
            repo_id=WAN_REPO_ID,
            repo_type="model",
            local_dir=str(BASE_MODEL_DIR),
            cache_dir="/app/models/.hf_cache",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "transformer/*",
                "transformer_2/*",
                "vae/*",
            ],
        )
    else:
        print("✅ Wan 2.2 base model already present")


if __name__ == "__main__":
    ensure_models()
