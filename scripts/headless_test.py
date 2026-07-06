"""Minimal headless generation smoke test — drives PipelineManager directly,
no Tkinter GUI. Run from the picture-ai project root with the venv python:

    venv\\Scripts\\python.exe scripts\\headless_test.py
"""
import logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so `picture_ai` imports when run from scripts/

from picture_ai.pipeline_manager import PipelineManager

pm = PipelineManager(models_root=ROOT / "models_cache")
pm.ensure_pipeline("RunDiffusion/Juggernaut-XL-v9", None)   # loads from models_cache offline
img = pm.generate_image(
    prompt="a single red apple on a wooden table, photo, sharp focus",
    negative_prompt="blurry, lowres",
    width=768, height=768, steps=8, guidance_scale=3.0, seed=7,
)
out = ROOT / "outputs" / "headless_test.png"
out.parent.mkdir(exist_ok=True)
img.save(out)
print("SAVED", out, img.size)
