import os
import requests


def upload_video(local_path: str, key: str) -> str:
    token = os.environ["BLOB_READ_WRITE_TOKEN"]
    with open(local_path, "rb") as f:
        r = requests.put(
            f"https://blob.vercel-storage.com/{key}",
            headers={"Authorization": f"Bearer {token}", "x-content-type": "video/mp4"},
            data=f,
            timeout=300,
        )
    r.raise_for_status()
    return r.json()["url"]
