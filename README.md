# Halfax Image Generator

A Tkinter desktop app that uses **Stable Diffusion XL (SDXL)** via the `diffusers` library to generate images from text prompts, with optional reference-image conditioning (img2img).

## Quick start

```bash
# NVIDIA GPU (recommended for RTX 5080 / 16 GB VRAM)
install_cuda.bat          # creates venv, installs CUDA PyTorch + deps
venv\Scripts\activate
python main.py
```

To pre-download the configured model:

```powershell
python .\scripts\download_model.py
```

Model will be saved into `models_cache/<model_id_safe_name>`.

## Features

- Text-to-image and **img2img** generation with SDXL.
- **Reference images** (up to 3): images are blended and used as the init image for img2img with adjustable strength.
- Model ID dropdown (default models: `stabilityai/stable-diffusion-xl-base-1.0`, `SG161222/RealVisXL_V5.0`, `RunDiffusion/Juggernaut-XL-v9`).
- Adjustable inference steps (10–100, default 35).
- CFG scale slider (defaults to the realism-friendly 4–5 range).
- Adjustable output size (width/height) with size presets.
- Style presets (Photoreal portrait, Anime, Cinematic).
- Optional **LoRA** attachment with preset picklist (Fabricated Reality, Objective Reality, Face Helper XL) that auto-downloads weights on first use.
- SDXL optimizations (applied automatically):
  - **fp16** inference for ~50% less VRAM.
  - Attention slicing.
  - VAE slicing/tiling for lower VRAM usage.
  - Optional xFormers memory‑efficient attention (if installed).
  - **DPMSolver++ (Karras)** scheduler for sharper outputs.
- Upscale 2x (Lanczos) and **AI Upscale** (SD x4 upscaler pipeline).
- Image preview, auto-caching, and save as PNG/JPEG.
- Dark / light mode toggle.

## Requirements

- **Python 3.10+** (recommended).
- A modern NVIDIA GPU with **≥ 8 GB VRAM** is strongly recommended.
  - This project is tuned for SDXL fp16 and works very well with **16 GB VRAM GPUs** (e.g. RTX 5080).
- Internet access on first run to download model weights from Hugging Face.

Python dependencies are listed in `requirements.txt`:

```text
transformers==4.57.1
accelerate==1.12.0
safetensors==0.7.0
numpy
pillow
diffusers[torch]==0.35.2
peft==0.18.1
huggingface-hub>=0.34.0,<1.0
```

For NVIDIA GPUs, use `install_cuda.bat` which installs PyTorch with CUDA 12.6 support. Then:

```bash
pip install -r requirements.txt
```

(Optional but recommended for faster attention on GPU):

```bash
pip install xformers
```

## Running the App

```bash
python main.py
```

This opens the **Halfax Image Generator** window.

## Using the GUI

1. **Prompt / Negative prompt** — Enter descriptive text. Use style presets to auto-fill.
2. **Model** — Pick from the dropdown or paste any compatible SDXL model ID.
3. **Steps** — 10–100, default 35. 20–35 is usually enough.
4. **W×H** — Output resolution. Clamped 256–1536, multiples of 8. Use size presets for common SDXL resolutions.
5. **Reference images** — Add up to 3 images. They are blended equally and used as the init image for img2img generation. Adjust **Strength** (0.1 = subtle influence, 1.0 = fully replace with ref). Without references, pure text-to-image is used.
6. **Generate** — Loads the model (if needed) and generates. Progress bar tracks steps.
7. **Upscale 2x / AI Upscale** — Post-generation upscaling options.
8. **Save Image** — Save as PNG/JPEG.

## Implementation Notes

- GUI: **Tkinter** + **ttk** in `picture_ai/app.py`.
- Backend: `StableDiffusionXLPipeline` (text2img) and `StableDiffusionXLImg2ImgPipeline` (img2img, sharing GPU weights) from `diffusers`, managed in `picture_ai/pipeline_manager.py`.
- The pipeline:
  - Uses **fp16** precision by default.
  - Picks `cuda` if available, else `cpu`.
  - Applies attention slicing, VAE slicing/tiling, xFormers (if available), and DPMSolver++ (Karras).
  - Loads and **fuses LoRA** adapters when configured.
  - LoRA presets are pre-fetched on app launch in background threads.
- Settings are persisted to `config/settings.json`.

### LoRA presets & custom entries

- Built-in presets: Fabricated Reality, Objective Reality, Face Helper XL.
- Add custom presets via `config/lora_presets.json`:

  ```json
  [
    {
      "label": "My Favorite LoRA",
      "source": "username/my-awesome-lora",
      "weight_name": null,
      "scale": 0.8
    }
  ]
  ```

- Presets are downloaded into `models_cache/loras/<safe-name>`. The downloader uses `token.txt`, `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`, or `config/hf_token.txt`.

### Model caching

- Models are stored under `models_cache/`. Each model id gets its own subfolder.
- On Windows you might see a warning about symlinks. This does not affect functionality; suppress with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## Troubleshooting

- **Out of memory (CUDA OOM)**
  - Reduce width/height (e.g. 768×768).
  - Reduce steps.
  - Close other GPU-intensive applications.

- **LoRA download fails**
  - Ensure internet access and valid HF token for private repos.
  - Place a token in `token.txt` or `config/hf_token.txt`.
  - Manually place weights in `models_cache/loras/<safe-name>`.

- **Very slow generation**
  - Ensure CUDA PyTorch is installed (not CPU-only): `python -c "import torch; print(torch.cuda.is_available())"`.
  - Install xFormers: `pip install xformers`.
  - Lower steps or resolution.

- **ImportError for `torch`, `diffusers`, `PIL`, etc.**
  - Make sure your virtual environment is active.
  - Reinstall: `pip install -r requirements.txt`
