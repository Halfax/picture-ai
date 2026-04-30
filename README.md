# PictureAI – SDXL Image Generator GUI

A simple Tkinter desktop app that uses **Stable Diffusion XL (SDXL)** via the `diffusers` library to generate images from text prompts.

## Features

- Text prompt input for SDXL.
- Model ID field (default: `stabilityai/stable-diffusion-xl-base-1.0`).
- Adjustable inference steps (10–100, default 30).
- CFG scale slider (defaults to the realism-friendly 4–5 range).
- Adjustable output size (width/height) for the generated image.
- Optional **LoRA** attachment with a curated preset picklist (UltraRealistic, DetailTweaker, Pony Realism, etc.) that auto-downloads the weights the first time you select them.
- SDXL optimizations:
  - Attention slicing.
  - VAE slicing/tiling for lower VRAM usage.
  - Optional xFormers memory‑efficient attention (if `xformers` is installed).
- DPMSolver++ (Karras) scheduler pre-configured for sharper outputs.
- Image preview inside the window.
- Save generated images as PNG/JPEG.

## Requirements

- Python 3.10+ (recommended).
- A modern NVIDIA GPU with **≥ 8 GB VRAM** is strongly recommended.
  - This project is tuned for SDXL and works very well with 16 GB VRAM GPUs.
- Internet access on first run to download the SDXL model weights from Hugging Face.
- `peft` (0.18.1 or newer) and `huggingface-hub` are required for LoRA loading/downloading (already listed in `requirements.txt`). If you need to access private LoRA repos, run `huggingface-cli login` or create `config/hf_token.txt` containing a valid token.

Python dependencies are listed in `requirements.txt`:

```text
torch
transformers
accelerate
safetensors
pillow
diffusers[torch]
```

For best performance and compatibility, install PyTorch from the official instructions for your platform/GPU:

- https://pytorch.org/get-started/locally/

Then install the rest via:

```bash
pip install -r requirements.txt
```

(Optional but recommended for faster attention on GPU):

```bash
pip install xformers
```

## Running the App

From the project directory:

```bash
python main.py
```

This opens the **Picture AI Generator** window.

## Using the GUI

1. **Prompt**
   - Enter a descriptive text prompt (e.g. `a futuristic cityscape at night, ultra detailed`).

2. **Model**
   - Default: `stabilityai/stable-diffusion-xl-base-1.0`.
   - You can paste another compatible SDXL model ID from Hugging Face if you want.

3. **Steps**
   - Controls the number of denoising steps.
   - Range: 10–100, default 30.
   - Higher = more detail but slower. 20–35 is usually enough.

4. **W×H (Width × Height)**
   - Output image resolution.
   - Defaults to **1024 × 1024** (SDXL’s standard size).
   - Values are clamped between 256 and 1536 and adjusted to multiples of 8.
   - Examples:
     - 1024 × 1024 – high quality, heavier on VRAM.
     - 768 × 768 – lighter, still very good quality.
     - 512 × 512 – significantly lighter and faster.

5. **Generate**
   - Starts model loading (if not already loaded) and then image generation.
   - Model loading and generation run in background threads so the UI stays responsive.
   - Status messages at the bottom show:
     - Model loading progress.
     - When generation starts/finishes.
     - Any errors.

6. **Save Image**
   - After a successful generation, click **Save Image**.
   - Choose a path and format (PNG/JPEG).

## Implementation Notes

- GUI is built using **Tkinter** and **ttk** widgets in `picture_ai/app.py`.
- SDXL backend is implemented with `StableDiffusionXLPipeline` from `diffusers` managed via `picture_ai/pipeline_manager.py`.
- The app:
  - Picks `cuda` if available, else `cpu`.
  - Disables `safety_checker` for simplicity (do not use in production without proper safety measures).
  - Enables attention slicing, VAE slicing/tiling, and, if available, xFormers attention.
  - Switches to DPMSolver++ (Karras) scheduler automatically.
  - Loads LoRA adapters (source path or HF repo) when configured in the UI and fuses them before generation.
  - When you pick a built-in LoRA preset, the app downloads the weights into `models_cache/loras/<preset>` (using `huggingface-hub`), ensures `peft` is installed, and remembers the preset in user settings.

### LoRA presets & custom entries

- Built-in presets: UltraRealistic XL v2, Detail Tweaker XL, and Pony Realism v2.2.
- You can add your own presets by creating `config/lora_presets.json` with entries like:

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

- On app launch, all presets (built-in + custom) are downloaded in the background so they are ready when selected. Selecting a preset also triggers a foreground check; if the download fails (e.g., missing auth, bad ID), a dialog explains what happened. The downloader automatically uses `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`, or the optional `config/hf_token.txt` file so you can preconfigure credentials.
- All heavy work (model loading and generation) is performed on background threads and marshalled back to the GUI thread using `after()` to keep the interface responsive.

### Model caching

- Models are stored under a project-local `models_cache/` directory.
  - Each model id gets its own subfolder (e.g. `stabilityai__stable-diffusion-xl-base-1.0`).
  - Once a model has been downloaded into this folder, subsequent runs reuse it from there.
- Hugging Face’s global cache (e.g. `C:\Users\<you>\.cache\huggingface\hub`) may still be used internally by the library, but this app always tries to load from its own `models_cache` path.
- On Windows you might see a warning about symlinks not being supported by your filesystem for the Hugging Face cache. This does **not** affect functionality; it only means more disk space is used. You can suppress the warning by setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1` in your environment if you wish.

## Troubleshooting

- **Out of memory (CUDA OOM) errors**
  - Reduce width/height (e.g. 768×768 or 512×512).
  - Reduce steps.
  - Close other GPU-intensive applications.
  - Disable VAE tiling or LoRA if you are loading many adapters simultaneously.

- **LoRA download fails**
  - Make sure you have internet access and, if the LoRA requires permission, a logged-in Hugging Face CLI/token.
  - Confirm `huggingface-hub` is installed (included in `requirements.txt`).
  - Place a token in `config/hf_token.txt` or export `HUGGINGFACE_HUB_TOKEN` to avoid signing in every time.
  - You can also place weights manually inside `models_cache/loras/<safe-name>` and point the LoRA source to that folder.

- **Very slow generation**
  - Ensure you are using a GPU build of PyTorch (not CPU‑only).
  - Lower steps or resolution.
  - Confirm xFormers is installed (`pip install xformers`) to use the optimized attention path.

  - Check your internet connection.
  - Some Hugging Face models may require authentication; the default SDXL base model is public at the time of writing.

- **ImportError for `torch`, `diffusers`, `PIL`, etc.**
  - Make sure your virtual environment is active.
  - Reinstall dependencies:

    ```bash
    pip install -r requirements.txt
    ```
