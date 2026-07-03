# Halfax Image Generator

A Tkinter desktop app that generates images from text prompts via the
`diffusers` library, tuned for a 16 GB GPU such as the **RTX 5080**. It runs
three model families:

- **SDXL** — diffusers-format repos *and* single-file Civitai-style
  checkpoints (Pony, DreamShaper, ...). Full feature set.
- **Stable Diffusion 3.5 Medium** — `StableDiffusion3Pipeline` (text2img).
- **Flux** dev / schnell — `FluxPipeline`, loaded **4-bit (NF4) quantized**
  with CPU offload so it fits 16 GB (text2img).

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

You can also paste any compatible model id into the picker. Unknown ids are
assumed to be SDXL; if a repo isn't diffusers-format the loader
automatically falls back to single-file loading.

## Features

- Three model families (SDXL / SD3.5 / Flux) — see above.
- **Single-file checkpoint loading** — `.safetensors` checkpoints (HF repo,
  URL, or local path) load via `from_single_file`.
- Text-to-image for all families; **img2img** and **reference images**
  (up to 3, blended) for SDXL.
- **Prompt weighting** — `(token:1.3)` syntax and >77-token prompts via
  `compel` (SDXL).
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

## Implementation Notes

- GUI: **Tkinter** + **ttk** in `picture_ai/app.py`.
- `picture_ai/pipeline_manager.py` is **family-aware** — `_create_pipeline`
  dispatches on the catalog `family`:
  - **sdxl** — `StableDiffusionXLPipeline` (+ Img2Img/Inpaint sharing GPU
    weights). compel, IP-Adapter, LoRA, HiRes Fix and the DPM++ sampler
    swap are SDXL-only, gated behind `_current_family`.
  - **sd3** — `StableDiffusion3Pipeline`, fp16, text2img.
  - **flux** — `FluxPipeline`, NF4-quantized transformer + T5 via
    `PipelineQuantizationConfig`, `enable_model_cpu_offload()`, text2img.
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
