"""Download the configured model into the local models cache using token.txt.

Usage (PowerShell):
  python .\scripts\download_model.py

This script reads `config/settings.json` for `model_id`, reads `token.txt` or env vars
`HUGGINGFACE_HUB_TOKEN`/`HF_TOKEN`, and uses `huggingface_hub.snapshot_download` to fetch
the model into `models_cache/<safe_name(model_id)>`.
"""
from pathlib import Path
import json
import os
import sys


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def load_token(root: Path) -> str | None:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    candidates = [root / "config" / "hf_token.txt", root / "hf_token.txt", root / "token.txt"]
    for p in candidates:
        if p.is_file():
            try:
                t = p.read_text(encoding="utf-8").strip()
                if t:
                    return t
            except Exception:
                pass
    return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "settings.json"
    if not cfg_path.is_file():
        print("config/settings.json not found", file=sys.stderr)
        return 2

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_id = data.get("model_id") or data.get("model")
    if not model_id:
        print("No model_id found in config/settings.json", file=sys.stderr)
        return 2

    models_root = root / "models_cache"
    target_dir = models_root / safe_name(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    token = load_token(root)
    if token:
        print("Using Hugging Face token from environment or token file")
    else:
        print("No HF token found; public models only will be downloadable (may fail for private models)")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print("Please install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 3

    print(f"Downloading model {model_id} into {target_dir} ...")
    try:
        snapshot_download(repo_id=model_id, local_dir=str(target_dir), local_dir_use_symlinks=False, token=token, resume_download=True)
        print("Download complete.")
        return 0
    except Exception as exc:
        print("Failed to download model:", exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
