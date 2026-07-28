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

## Programmatic use — prefer `picture_ai.api` (added 2026-07-27)

**`api.py` is now the supported entry point**, not raw `PipelineManager`. It
owns model loading, family dispatch, legal frame counts and encoding, so a
caller states intent and nothing else:

```python
from picture_ai import api
img  = api.generate_image("a lantern in fog", steps=25)        # -> ImageResult
clip = api.generate_video("a lantern swinging in fog", seconds=4)  # -> Clip
clip.save("clip.mp4")          # NVENC h264/hevc/av1, libx264 fallback
```

Also `python -m picture_ai.cli {image,video,encode,models,doctor,serve}` and the
HTTP job API (`serve`). All three are the same engine — `cli` and `server` are
thin clients of `api`.

⚠ **The older recipe below still works and is NOT deprecated** — several
existing consumers use it (`arhutils/pictureai_*.py`, `ashes-of-grace/art/*.py`).
`PipelineManager`'s signature is unchanged and backward compatible. Prefer
`api` for anything new; there is no need to migrate working scripts.

### The raw PipelineManager recipe — VERIFIED 2026-07-06, don't re-derive

Verified on UYScuti / RTX 5080 (Juggernaut-XL-v9, 768², 8 steps, seconds):

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
  - **ltx** — `LTXPipeline`, bf16 + `enable_model_cpu_offload()`, **video**.
    Entered through `generate_video()`, NOT `generate_image()` (which raises
    for a video family rather than failing deep in the pipeline). Frame counts
    and dimensions are snapped to what the 3D VAE accepts (`VideoSpec`: 8k+1
    frames, /32 dims for LTX) — rounding to the NEAREST legal value, because
    flooring turns a request for 8 frames into 1, i.e. a still.
    🔴 **VAE tiling is load-bearing here, not an optimization.** Video latents
    are frames×H×W and the stack decodes to pixels *after* the final sampling
    step, so without tiling a run completes every step and then dies at the
    decode. Same failure shape as DECISIONS §22 rule A on Betelgeuse's iGPU,
    different substrate.
  - **Fetching LTX:** `scripts/fetch_ltx.py`. Do NOT `snapshot_download` the
    whole repo — `Lightricks/LTX-Video` keeps every released checkpoint at its
    root and totals **~254 GB**; only the diffusers subfolders (~28.5 GB) are
    used by `from_pretrained`.
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

- **There IS a server now — the old "don't add one" rule was overturned
  2026-07-27, by Andrew, deliberately.** The previous rule read *"No web
  framework, no server... Don't add a server."* Its reasoning was that a local
  Tkinter app has no trust boundary to reason about. That reasoning was sound
  for a GUI-only app and is now simply out of date: the app was **UI-only with
  no callable surface**, so every other consumer — a script, another project,
  another host on the fleet — had to reimplement model loading and family
  rules by driving `PipelineManager` internals. The fix is `api.py` (the
  library), `cli.py` (the shell), `server.py` (the network).
  **This is a decision, not drift — don't "restore" the no-server rule.**
- **The server is stdlib `http.server`, and that part is still a WON'T.**
  🚫 **DELIBERATELY NOT DONE: no FastAPI / uvicorn / Flask dependency.** A
  loopback job queue does not need a web framework, and this is a desktop app
  whose dependency set is already heavy (torch, diffusers, transformers).
  If you find yourself wanting FastAPI, the question to answer first is what
  the framework buys that ~200 lines of `BaseHTTPRequestHandler` doesn't.
- 🔴 **The HTTP API has NO authentication, and the default bind is loopback.**
  On loopback the trust boundary is "processes on this box". Binding elsewhere
  (`--host`) puts unauthenticated GPU-consuming, disk-writing endpoints on the
  LAN/mesh, which is DECISIONS §19 territory and wants a firewall rule, not
  just the flag. The flag exists because the fleet may want it; the loopback
  default must not drift.
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

## Future / under consideration

- **Video generation — SHIPPED 2026-07-27**, no longer speculative. LTX
  text2video + NVENC encoding + UI + API/CLI/HTTP. `projects/VIDEO-IDEAS.md`
  still holds the reasoning and the *engine-side* half (HalfaxForge cutscene
  playback), which remains undecided.
- **Video features not built** (CAN'T-yet, not WON'T — nobody has needed them):
  image2video (`LTXImageToVideoPipeline` is already imported by
  `_lazy_import_ltx_pipelines` and unused), video2video, the LTX latent
  upscalers, and other video families (Wan, CogVideoX, HunyuanVideo). Adding a
  family means a catalog entry with a `VideoSpec` plus a branch in
  `_create_pipeline`; `generate_video()` itself is family-agnostic.
- 🚫 **DELIBERATELY NOT DONE: no second video model was added "for coverage".**
  Wan-1.3B would fit the 5080 easily, but an unverified catalog entry is worse
  than none — it claims support nobody ran. Add one when it is going to be
  used, and generate with it before committing the entry.

## Pointer back

If you learn something durable that other hosts / other AI tools should
know, promote it to `betelgeuse-docs/DECISIONS.md` (DECISIONS §1) rather
than leaving it only here or in conversation memory.
