# Halfax Image Generator

Generates **images and video** from text prompts via the `diffusers` library,
tuned for a 16 GB GPU such as the **RTX 5080**. It runs four model families:

- **SDXL** — diffusers-format repos *and* single-file Civitai-style
  checkpoints (Pony, DreamShaper, ...). Full feature set.
- **Stable Diffusion 3.5 Medium** — `StableDiffusion3Pipeline` (text2img).
- **Flux** dev / schnell — `FluxPipeline`, loaded **4-bit (NF4) quantized**
  with CPU offload so it fits 16 GB (text2img).
- **LTX-Video** — `LTXPipeline`, bf16 with CPU offload (**text2video**).

There are four ways to drive it, all on the same engine:

| Surface | Use it for |
|---|---|
| **Tkinter GUI** (`python main.py`) | interactive work; video gets an in-app animated preview with scrub + play |
| **Python API** (`from picture_ai import api`) | scripts and other projects |
| **CLI** (`python -m picture_ai.cli`) | shell / batch / other languages |
| **HTTP API** (`python -m picture_ai.cli serve`) | long-running jobs, other machines on the fleet |

## Quick start

```bash
# NVIDIA GPU (recommended for RTX 5080 / 16 GB VRAM)
install_cuda.bat          # creates venv, installs CUDA PyTorch + deps
venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

> **Note on this venv:** the bundled `venv` was created under the project's
> old name (`picutreai`). Console-script wrappers (`pip.exe`, etc.) embed an
> absolute interpreter path and are broken by the rename — use
> `python -m pip` instead of bare `pip`. A clean fix is recreating the venv:
> `python -m venv venv`.

To pre-download the configured model:

```powershell
python .\scripts\download_model.py
```

Models are saved into `models_cache/<model_id_safe_name>/`.

## Models

The curated catalog lives in `picture_ai/model_catalog.py` — each entry
carries a family, a one-line usage hint, a VRAM estimate and recommended
sampling settings (surfaced by the **Recommended settings** button).

| Model | Family | Notes |
|---|---|---|
| SDXL Base 1.0 | sdxl | Baseline, biggest LoRA ecosystem |
| RealVisXL V5.0 | sdxl | Photoreal finetune |
| Juggernaut XL v9 / XI v11 | sdxl | Photoreal all-rounders |
| Lustify SDXL | sdxl | Uncensored (NSFW) |
| DreamShaper XL | sdxl | Versatile artistic |
| SDXL Turbo | sdxl | 1–4 steps, CFG 1.0 — fast iteration |
| Pony Diffusion V6 XL | sdxl | Uncensored; single-file checkpoint; needs `score_*` prompt tags |
| Stable Diffusion 3.5 Medium | sd3 | Gated — accept the license on Hugging Face |
| Flux.1 dev / schnell | flux | Gated; NF4-quantized to fit 16 GB |
| LTX-Video | ltx | **Video.** bf16 + CPU offload; 704×480 native-ish, 24 fps |

> **Fetching LTX:** don't `snapshot_download` the whole repo — `Lightricks/LTX-Video`
> keeps every released checkpoint at its root and totals **~254 GB**. Only the
> diffusers subfolders (~28.5 GB) are needed:
> `venv\Scripts\python.exe scripts\fetch_ltx.py`.

You can also paste any compatible model id into the picker. Unknown ids are
assumed to be SDXL; if a repo isn't diffusers-format the loader
automatically falls back to single-file loading.

## Driving it without the GUI

```python
from picture_ai import api

img  = api.generate_image("a lantern in fog")           # -> ImageResult
img.save("still.png")

clip = api.generate_video("a lantern swinging in fog, slow dolly in", seconds=4)
clip.save("clip.mp4")            # NVENC-encoded
clip.thumbnail().save("t.png")   # frame 0
clip.save_frames("frames/")      # the PNGs, if you want them
```

The API owns model loading, family rules, legal frame counts and encoding, so
callers don't reimplement them. Consecutive calls reuse the loaded model.

```powershell
venv\Scripts\python.exe -m picture_ai.cli doctor          # GPU / encoders / models
venv\Scripts\python.exe -m picture_ai.cli models --kind video
venv\Scripts\python.exe -m picture_ai.cli video "..." --seconds 4 --out clip.mp4
venv\Scripts\python.exe -m picture_ai.cli encode frames\ out.mp4 --fps 24
venv\Scripts\python.exe -m picture_ai.cli serve           # HTTP API on 127.0.0.1:8770
```

### HTTP API

Generation takes minutes, so it is a **job API** — submit, poll, fetch:

```
GET    /healthz              worker + queue health (not just "am I up")
GET    /v1/models[?kind=]    catalog
POST   /v1/images            {"prompt": "...", ...}          -> 202 {job_id}
POST   /v1/videos            {"prompt": "...", "seconds": 4} -> 202 {job_id}
GET    /v1/jobs/<id>         state + step/of progress
GET    /v1/jobs/<id>/file    the artifact
DELETE /v1/jobs/<id>         cancel (only before it starts)
```

One GPU means one worker thread and a FIFO queue. **There is no authentication
— the default bind is loopback.** `--host` exists, but exposing it puts
unauthenticated GPU-consuming endpoints on the network; treat that as a
firewall decision, not a flag.

## Features

- Four model families (SDXL / SD3.5 / Flux / LTX-Video) — see above.
- **Text-to-video** (LTX): length in seconds or frames, snapped to a count the
  model's 3D VAE accepts; H.264/HEVC/**AV1** encoding via NVENC with an
  automatic libx264 fallback.
- **In-app video preview** — play/pause, scrub, and a frame counter; frames are
  pre-scaled once so playback holds the model's fps.
- Python API, CLI and a local HTTP job API on the same engine.
- **Single-file checkpoint loading** — `.safetensors` checkpoints (HF repo,
  URL, or local path) load via `from_single_file`.
- Text-to-image for all families; **img2img** and **reference images**
  (up to 3, blended) for SDXL.
- **Prompt weighting** — `(token:1.3)` syntax and **>77-token prompts** via
  `compel` (SDXL). Long prompts genuinely reach the model: compel's SDXL path
  has an upstream bug (`EmbeddingsProviderMulti` has no `empty_z`) that used to
  make `_encode_prompts` fall back to a *silently truncated* prompt whenever the
  positive and negative prompts had different chunk counts. That is patched —
  `scripts/test_compel_longprompt.py` proves a 101-token prompt produces 154
  conditioning rows rather than 77.
- Selectable **samplers** — DPM++ 2M Karras (default), DPM++ SDE Karras,
  Euler a, UniPC (SDXL).
- **HiRes Fix** — optional two-pass upscale + low-denoise img2img refine
  (SDXL).
- **Quality booster** toggle — appends masterpiece/quality terms.
- Style presets (Photoreal portrait, Raw photo, Editorial film, Cinematic,
  NSFW photoreal, Anime, Fantasy art).
- Optional **LoRA** with a preset picklist that auto-downloads on first use
  (SDXL).
- **A1111-compatible PNG metadata** embedded on save — re-readable by
  Automatic1111 / ComfyUI.
- VRAM/speed optimizations applied automatically: fp16 + channels_last
  (SDXL), NF4 + VAE slicing/tiling + model CPU offload (Flux/SD3).
- Upscale 2x (Lanczos) and **AI Upscale** (SD x4 upscaler).
- Image preview, auto-caching (capped at the 500 newest), save as
  PNG/JPEG, dark / light mode.

## Requirements

- **Python 3.10+** (the bundled venv is 3.11).
- A modern NVIDIA GPU. **16 GB VRAM** (RTX 5080) is the design target;
  SDXL works from ~8 GB.
- Internet access on first run to download weights from Hugging Face.
- SD3.5 and Flux are **gated** — accept their licenses on Hugging Face and
  put a valid token in `token.txt`.

`requirements.txt`:

```text
transformers==4.57.1
accelerate==1.12.0
safetensors==0.7.0
numpy
pillow
diffusers[torch]==0.35.2
peft==0.18.1
huggingface-hub>=0.34.0,<1.0
compel>=2.0.2,<3.0
bitsandbytes>=0.45.0       # Flux NF4 quantization
sentencepiece>=0.2.0       # T5 tokenizer for SD3.5 / Flux
protobuf>=4.25
```

For NVIDIA GPUs, `install_cuda.bat` installs PyTorch with CUDA support.
Then `python -m pip install -r requirements.txt`. (Attention uses PyTorch
2.x SDPA — xFormers is not needed.)

## Using the GUI

1. **Prompt / Negative prompt** — descriptive text; style presets auto-fill.
   (Flux ignores the negative prompt — it is guidance-distilled.)
2. **Model** — pick from the dropdown or paste a model id. The hint line
   below shows the model's tag and usage note.
3. **Recommended settings** — fills in steps / CFG / sampler tuned for the
   selected model (e.g. Flux guidance ≈ 3.5, SDXL Turbo 4 steps at CFG 1).
4. **Steps / W×H / Seed / CFG / Sampler** — standard controls. Sizes are
   clamped 256–1536; Flux rounds to multiples of 16.
5. **Reference images** — SDXL only; up to 3, blended into an img2img init.
6. **Generate** — loads the model if needed (switching families reloads).
7. **Upscale 2x / AI Upscale / Save Image**.
8. **Video** (LTX selected) — set **Length (s)**; the hint shows the frame count
   the model will actually use, since it snaps to a legal value (8k+1 for LTX).
   Pick a **codec** (`auto` prefers NVENC) and **quality** (CRF/CQ, lower is
   better). After generating, the clip **plays in the window** — pause, scrub,
   frame counter — then **Save Video** encodes it.

## Implementation Notes

- GUI: **Tkinter** + **ttk** in `picture_ai/app.py`. Selecting a video model
  swaps the mode: HiRes/LoRA/reference panels are removed and the video panel
  (length, codec, quality) takes their place, so the UI never offers a control
  the selected family will ignore.
- `picture_ai/api.py` — the callable surface (`Engine`, `generate_image`,
  `generate_video`, `encode`, `list_models`). `cli.py` and `server.py` are both
  thin clients of it; so is anything outside this repo.
- `picture_ai/video_export.py` — frames → video via a raw-RGB pipe into ffmpeg
  (bundled by `imageio-ffmpeg`, no system install). Deliberately not
  `diffusers.utils.export_to_video`, which silently falls back to the OpenCV
  writer when only `cv2` is present and gives no codec control.
- `picture_ai/pipeline_manager.py` is **family-aware** — `_create_pipeline`
  dispatches on the catalog `family`:
  - **sdxl** — `StableDiffusionXLPipeline` (+ Img2Img/Inpaint sharing GPU
    weights). compel, IP-Adapter, LoRA, HiRes Fix and the DPM++ sampler
    swap are SDXL-only, gated behind `_current_family`.
  - **sd3** — `StableDiffusion3Pipeline`, fp16, text2img.
  - **flux** — `FluxPipeline`, NF4-quantized transformer + T5 via
    `PipelineQuantizationConfig`, `enable_model_cpu_offload()`, text2img.
  - **ltx** — `LTXPipeline`, bf16 + `enable_model_cpu_offload()` (the T5-XXL
    encoder is the memory hog, not the transformer), text2video via
    `generate_video()`. **VAE tiling is mandatory**: video latents are
    frames×H×W and the whole stack decodes to pixels *after* the last sampling
    step, so an untiled decode dies at the very end of a successful run.
- `picture_ai/model_catalog.py` — the model catalog (family, tag, blurb,
  VRAM, recommended settings); `catalog_get()` returns an SDXL fallback for
  uncatalogued ids.
- Settings persist to `config/settings.json`; the custom model list to
  `models_cache/models.json`.

### LoRA presets & custom entries

- Built-in presets: Fabricated Reality, Objective Reality, Face Helper XL.
- Add custom presets via `config/lora_presets.json`:

  ```json
  [
    { "label": "My LoRA", "source": "username/my-lora", "weight_name": null, "scale": 0.8 }
  ]
  ```

- Presets download into `models_cache/loras/<safe-name>`. The downloader
  uses `token.txt`, `HUGGINGFACE_HUB_TOKEN`, or `HF_TOKEN`.
- LoRA applies to SDXL only — it is ignored for SD3 / Flux.

## Troubleshooting

- **`pip` "cannot find the file specified"** — the venv's console-script
  wrappers are stale (see the venv note above). Use `python -m pip`.
- **CUDA OOM** — reduce width/height (e.g. 768×768) or steps; close other
  GPU apps. Flux is tight on 16 GB even quantized — keep resolution modest.
- **SD3.5 / Flux won't load** — they are gated: accept the license on
  Hugging Face and ensure `token.txt` holds a valid token. Flux also needs
  `bitsandbytes` installed (without it, the load is unquantized and OOMs).
- **Very slow first Flux run** — expected: NF4 weights download and CPU
  offload streams components per step.
- **ImportError for `torch` / `diffusers` / `PIL`** — activate the venv and
  `python -m pip install -r requirements.txt`.
