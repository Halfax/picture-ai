"""Pre-fetch the LTX-Video diffusers weights into picture-ai's model cache.

The Lightricks/LTX-Video repo is ~254 GB because it carries every released
checkpoint as a single .safetensors at the repo root. `from_pretrained` only
reads the diffusers-format subfolders, so we fetch exactly those (~28.5 GB).

Uses `cache_dir` (not `local_dir`) so the layout matches what
PipelineManager._model_dir_for_id + from_pretrained expect.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "Lightricks/LTX-Video"
CACHE = ROOT / "models_cache" / REPO.replace("/", "__")

ALLOW = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    token = None
    tok_file = ROOT / "token.txt"
    if tok_file.exists():
        token = tok_file.read_text(encoding="utf-8").strip() or None
    print(f"Fetching {REPO} diffusers weights -> {CACHE}", flush=True)
    path = snapshot_download(
        repo_id=REPO,
        cache_dir=str(CACHE),
        allow_patterns=ALLOW,
        token=token,
        max_workers=8,
    )
    print(f"OK: {path}", flush=True)
    sys.exit(0)
