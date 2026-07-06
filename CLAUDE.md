# picture-ai (main / RTX 5080 build) — project-local notes for AI assistants

Cross-host homelab rules live in `projects/DECISIONS.md`. Read those first.
This file captures conventions specific to *this* project that aren't worth
promoting to DECISIONS.

## Two branches — don't confuse them

picture-ai has two divergent tracks:

- **`main`** (this one) — a **Tkinter desktop app** using the `diffusers`
  library on **CUDA**, targeting the UYScuti **RTX 5080 (16 GB)**.
- **`betelgeuse`** — a FastAPI **web service** using a **stable-diffusion.cpp
  Vulkan** backend on the Betelgeuse Strix Halo box. Completely different
  architecture; see that branch's own `CLAUDE.md`.

Code does not port verbatim between them. The `betelgeuse` branch's model
registry *metadata* was ported into `main`'s `model_catalog.py`; its Vulkan
backend, sd-server and q8_0 quantization were not (different substrate).

## Headless / programmatic use (no GUI) — VERIFIED 2026-07-06, don't re-derive

There is **no server/API** — but you don't need the Tkinter GUI to generate.
Drive `PipelineManager` directly. This exact recipe was verified on UYScuti /
RTX 5080 (Juggernaut-XL-v9, 768², 8 steps, generated in seconds):

```python
import sys
from pathlib import Path
ROOT = Path(r"C:\Users\arhal_iz5093n\Desktop\projects\picture-ai")
sys.path.insert(0, str(ROOT))                        # needed when run from scripts/
from picture_ai.pipeline_manager import PipelineManager, REF_MODE_IMG2IMG

pm = PipelineManager(models_root=ROOT / "models_cache")
pm.ensure_pipeline("RunDiffusion/Juggernaut-XL-v9", None)   # loads from cache, OFFLINE, CUDA
img = pm.generate_image(                              # -> PIL.Image
    prompt="...", negative_prompt="...",
    width=1024, height=1024, steps=25, guidance_scale=5.0, seed=7,
    # img2img (SDXL only): pass an init image and how much to keep
    # reference_images=[r"path\to\init.png"], ref_mode=REF_MODE_IMG2IMG, strength=0.5,
)
img.save(ROOT / "outputs" / "foo.png")
```

Run with the **venv python directly** — the console-script wrappers (`pip.exe`,
etc.) are stale (venv was made under the old name `picutreai`):

```
venv\Scripts\python.exe scripts\your_script.py
```

- **Locally cached models** (offline; pass the HF id, it maps to
  `models_cache/<id with "/"→"__">`): `RunDiffusion/Juggernaut-XL-v9`,
  `RunDiffusion/Juggernaut-XI-v11`, `SG161222/RealVisXL_V5.0`,
  `stabilityai/stable-diffusion-xl-base-1.0`, `John6666/lustify-sdxl-...`
  (uncensored), `runwayml/stable-diffusion-v1-5`,
  `stabilityai/stable-diffusion-3.5-medium`.
- **img2img / reference** is SDXL-only: `reference_images=[path]` +
  `ref_mode=REF_MODE_IMG2IMG` + `strength` (~0.3 keeps the init, ~0.8 mostly
  new). Ideal for texture projection (feed a rendered view, refine to photoreal).
- **SDXL native res is 1024** — 512 looks bad; use 768–1024. 5080/16 GB fits
  SDXL fp16 with headroom. ~25 steps @ CFG 5 for quality, 8 steps for a smoke test.
- Reusable smoke test lives at `scripts/headless_test.py`.

## Architecture as it stands

- **`pipeline_manager.py` is family-aware.** `_create_pipeline` reads the
  catalog `family` and dispatches:
  - **sdxl** — `StableDiffusionXLPipeline` (+ Img2Img / Inpaint built from
    shared components). The full feature set lives here: compel prompt
    weighting, IP-Adapter, LoRA fuse, HiRes Fix, the DPM++ sampler swap.
  - **sd3** — `StableDiffusion3Pipeline`, fp16, text2img only.
  - **flux** — `FluxPipeline`, NF4-quantized (transformer + T5) via
    `PipelineQuantizationConfig`, `enable_model_cpu_offload()`, text2img.
- **All SDXL-only machinery is gated behind `self._current_family`.** The
  SDXL path is unchanged from before the multi-family work — if you touch
  `generate_image`, keep the `_current_family in ("sd3","flux")` early-out
  ahead of the reference-image / compel / hires branches.
- **`model_catalog.py`** — `MODEL_CATALOG` keyed by HF repo id; each entry
  has `family`, `tag`, `blurb`, `vram_base_gb`, `single_file`, `gated`, and
  a `Recommended` block. `catalog_get()` returns an SDXL fallback for
  uncatalogued ids. `app.py` and `pipeline_manager.py` both read it.
- **SDXL runs plain PyTorch 2.x SDPA + channels_last — no slicing.**
  Attention slicing and VAE slicing/tiling are deliberately NOT enabled on
  the SDXL path (removed 2026-07-03): 16 GB fits SDXL fp16 with headroom,
  slicing on top of SDPA only slows inference, and attention slicing broke
  IP-Adapter processor loading (the old disable-slicing workaround in
  `_ensure_ip_adapter_loaded` went with it). The DiT path (SD3/Flux) keeps
  VAE slicing/tiling — those loads are tight. Don't re-add slicing to SDXL.
- **LoRA changes swap in place.** `ensure_pipeline` detects same-model /
  different-LoRA and goes through `_swap_lora` (`unfuse_lora` +
  `unload_lora_weights` + re-apply) instead of a full disk reload; a failed
  swap falls back to the reload path. `image_cache/` is capped at the 500
  newest PNGs (pruned at startup).

## When editing this project

- **No web framework, no server.** This is a local Tkinter app — there is
  no trust boundary to reason about (unlike the `betelgeuse` branch / the
  Halfax-AI API in DECISIONS §19). Don't add a server.
- **SD3 / Flux are text2img-only here.** compel, IP-Adapter, reference
  images, HiRes Fix and LoRA are SDXL-only. If you extend SD3/Flux, add a
  family-appropriate path — do not route them through the SDXL helpers
  (`_get_or_create_img2img_pipeline` etc. construct SDXL-specific classes).
- **Settings** persist to `config/settings.json`; the custom model list to
  `models_cache/models.json`. New `UserSettings` fields need a default so
  old settings files still load (the store merges over dataclass defaults).

## Common pitfalls

- **Single-file vs diffusers-format checkpoints** — `from_pretrained` needs
  a multi-folder diffusers repo; Civitai-style one-file `.safetensors`
  checkpoints need `from_single_file`. The SDXL loader tries `from_pretrained`
  then falls back to `from_single_file` automatically, so a catalog
  `single_file` flag is a hint/optimization, not load-bearing.
- **Never `.to("cuda")` a quantized Flux pipeline.** With an NF4
  `quantization_config`, component placement is handled by
  `enable_model_cpu_offload()`. Calling `.to()` on quantized weights errors
  or defeats the offload. The SDXL/SD3 paths *do* call `.to(device)`.
- **Flux takes no negative prompt** — it is guidance-distilled. `_generate_dit_text2img`
  passes `negative_prompt` for sd3 only. Flux "CFG" in the UI is Flux's
  distilled guidance — keep it low (~3.5 dev, off for schnell).
- **Flux VAE needs dimensions that are multiples of 16**; the UI clamps to
  multiples of 8, so `_generate_dit_text2img` re-rounds for flux.
- **The DPM++ `_init_step_index` monkey-patch** (`_patch_dpm_init_step_index`)
  defends against an `IndexError` on the final step with karras sigmas on
  certain checkpoints (Lustify). See DECISIONS §7 for the schema/grammar
  cousin of this; the patch itself is documented in-code.
- **The bundled `venv` is stale.** It was created under the project's old
  name `picutreai`; the folder was later renamed to `picture-ai`.
  Console-script wrappers (`pip.exe`, ...) embed an absolute interpreter
  path and fail with "cannot find the file specified". `python -m pip`
  works; recreating the venv (`python -m venv venv`) is the clean fix.

## Pointer back

If you learn something durable that other hosts / other AI tools should
know, promote it to `betelgeuse-docs/DECISIONS.md` (DECISIONS §1) rather
than leaving it only here or in conversation memory.
