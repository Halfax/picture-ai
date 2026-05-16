"""Curated model metadata for the RTX 5080 (16 GB) build of picture-ai.

Ported and extended from the `betelgeuse` branch's model registry
(config/models.yaml + registry.py). The betelgeuse box has 64 GB of unified
memory; this 16 GB card can't fit the big models at full precision, so the
Flux entries here are loaded NF4-quantized and SD3.5 Large / Chroma / HiDream
are deliberately omitted.

`family` drives which diffusers pipeline `pipeline_manager` builds:
    "sdxl" — StableDiffusionXL* pipelines (compel, IP-Adapter, LoRA, hires)
    "sd3"  — StableDiffusion3Pipeline      (text2img only)
    "flux" — FluxPipeline, NF4-quantized   (text2img only)

`single_file` marks an SDXL checkpoint distributed as one .safetensors file
rather than a diffusers-format repo; the loader uses `from_single_file` for
those (and falls back to it automatically if `from_pretrained` fails).

Entries are keyed by Hugging Face repo id so the catalog lines up with the
editable model picker and the persisted `model_id` setting; a repo id the
user typed that isn't here resolves to a generic SDXL fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recommended:
    """Suggested sampling settings, applied by the 'Recommended settings'
    button. `sampler` must be a key of pipeline_manager.SAMPLERS or it is
    left unchanged (and it is ignored entirely for sd3/flux, which keep
    their native flow-matching scheduler)."""

    steps: int = 35
    cfg_scale: float = 6.5
    sampler: str = "DPM++ 2M Karras"
    note: str = "SDXL: CFG 6-8, 30-40 steps, DPM++ 2M Karras."


@dataclass(frozen=True)
class ModelInfo:
    repo_id: str
    label: str
    family: str = "sdxl"               # sdxl | sd3 | flux
    tag: str = "custom"
    blurb: str = ""
    # Heuristic base VRAM (GiB) for a 1024x1024 / 30-step gen; the VRAM
    # estimator scales this by resolution and step count.
    vram_base_gb: float = 7.5
    single_file: bool = False          # SDXL only: load via from_single_file
    gated: bool = False                # HF repo requires accepting a license
    recommended: Recommended = field(default_factory=Recommended)


# Shared recommendation for the plain SDXL finetunes.
_SDXL_REC = Recommended()


MODEL_CATALOG: dict[str, ModelInfo] = {
    # ---- SDXL family, diffusers-format (from_pretrained) -------------------
    "stabilityai/stable-diffusion-xl-base-1.0": ModelInfo(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        label="SDXL Base 1.0",
        family="sdxl",
        tag="general-purpose · filtered",
        blurb="Baseline SDXL — reliable, with the largest LoRA ecosystem. A safe general-purpose default.",
        vram_base_gb=7.5,
    ),
    "SG161222/RealVisXL_V5.0": ModelInfo(
        repo_id="SG161222/RealVisXL_V5.0",
        label="RealVisXL V5.0",
        family="sdxl",
        tag="photoreal · filtered",
        blurb="Photoreal SDXL finetune — best for realistic people and scenes. Pairs well with the 'Raw photo (analog)' style.",
        vram_base_gb=8.0,
    ),
    "RunDiffusion/Juggernaut-XL-v9": ModelInfo(
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        label="Juggernaut XL v9",
        family="sdxl",
        tag="photoreal all-rounder · filtered",
        blurb="Versatile photoreal SDXL — a strong all-rounder across portraits, scenes and products.",
        vram_base_gb=8.5,
    ),
    "RunDiffusion/Juggernaut-XI-v11": ModelInfo(
        repo_id="RunDiffusion/Juggernaut-XI-v11",
        label="Juggernaut XI v11",
        family="sdxl",
        tag="photoreal, sharper · filtered",
        blurb="Newer Juggernaut — sharper detail and lighting than v9.",
        vram_base_gb=8.5,
    ),
    "John6666/lustify-sdxl-nsfwsfw-endgame-sdxl": ModelInfo(
        repo_id="John6666/lustify-sdxl-nsfwsfw-endgame-sdxl",
        label="Lustify SDXL",
        family="sdxl",
        tag="UNCENSORED (NSFW)",
        blurb="Uncensored SDXL finetune — small and fast. Pair with the 'NSFW photoreal' style preset.",
        vram_base_gb=8.5,
    ),
    "Lykon/dreamshaper-xl-1-0": ModelInfo(
        repo_id="Lykon/dreamshaper-xl-1-0",
        label="DreamShaper XL",
        family="sdxl",
        tag="versatile artistic · lightly filtered",
        blurb="Versatile SDXL finetune — strong on illustration, concept art and stylised portraits.",
        vram_base_gb=8.5,
    ),
    "stabilityai/sdxl-turbo": ModelInfo(
        repo_id="stabilityai/sdxl-turbo",
        label="SDXL Turbo",
        family="sdxl",
        tag="fast (1-4 steps) · filtered",
        blurb="Distilled SDXL — 1-4 steps, near-instant. Lower peak quality; great for fast iteration.",
        vram_base_gb=7.0,
        recommended=Recommended(
            steps=4,
            cfg_scale=1.0,
            sampler="Euler a",
            note="SDXL Turbo is distilled for 1-4 steps at CFG 1.0 — more steps/CFG just degrade it.",
        ),
    ),
    # ---- SDXL family, single-file checkpoint (from_single_file) -----------
    "LyliaEngine/Pony_Diffusion_V6_XL": ModelInfo(
        repo_id="LyliaEngine/Pony_Diffusion_V6_XL",
        label="Pony Diffusion V6 XL",
        family="sdxl",
        tag="UNCENSORED · huge LoRA ecosystem",
        blurb="The dominant uncensored SDXL model. The prompt MUST start with score tags (score_9, score_8_up, score_7_up, ...) — Pony was trained on them.",
        vram_base_gb=8.5,
        single_file=True,
        recommended=Recommended(
            steps=28,
            cfg_scale=7.0,
            sampler="Euler a",
            note="Pony V6: CFG ~7, 25-30 steps. Prompt must lead with score_9, score_8_up, score_7_up...",
        ),
    ),
    # ---- SD3.5 family (StableDiffusion3Pipeline, text2img only) -----------
    "stabilityai/stable-diffusion-3.5-medium": ModelInfo(
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        label="Stable Diffusion 3.5 Medium",
        family="sd3",
        tag="high quality · filtered · gated",
        blurb="SD3.5 Medium — strong prompt-following and text rendering. ~10 GiB at fp16. Requires accepting the license on Hugging Face.",
        vram_base_gb=11.0,
        gated=True,
        recommended=Recommended(
            steps=35,
            cfg_scale=4.5,
            sampler="Euler a",
            note="SD3.5: CFG 4-5, 28-40 steps. Flow-matching model — the sampler picker doesn't apply.",
        ),
    ),
    # ---- Flux family (FluxPipeline, NF4-quantized, text2img only) ---------
    "black-forest-labs/FLUX.1-dev": ModelInfo(
        repo_id="black-forest-labs/FLUX.1-dev",
        label="Flux.1 dev (NF4)",
        family="flux",
        tag="top quality · gated · 4-bit",
        blurb="Highest fidelity and prompt-following. Loaded 4-bit (NF4) with CPU offload to fit 16 GB — expect a slower first run. Requires accepting the Flux license on Hugging Face.",
        vram_base_gb=13.0,
        gated=True,
        recommended=Recommended(
            steps=25,
            cfg_scale=3.5,
            sampler="Euler a",
            note="Flux dev: guidance ~3.5, 20-28 steps. 'CFG' here is Flux's distilled guidance — keep it low.",
        ),
    ),
    "black-forest-labs/FLUX.1-schnell": ModelInfo(
        repo_id="black-forest-labs/FLUX.1-schnell",
        label="Flux.1 schnell (NF4)",
        family="flux",
        tag="fast (4 steps) · gated · 4-bit",
        blurb="Distilled Flux — 4 steps, Apache 2.0. Loaded 4-bit (NF4) with CPU offload to fit 16 GB.",
        vram_base_gb=13.0,
        gated=True,
        recommended=Recommended(
            steps=4,
            cfg_scale=1.0,
            sampler="Euler a",
            note="Flux schnell is distilled for ~4 steps — more steps just waste time. Guidance is effectively off.",
        ),
    ),
}


def catalog_get(repo_id: str) -> ModelInfo:
    """Return the catalog entry for a repo id, or a generic SDXL fallback for
    a model the user typed in that isn't curated."""
    rid = (repo_id or "").strip()
    info = MODEL_CATALOG.get(rid)
    if info is not None:
        return info
    return ModelInfo(
        repo_id=rid,
        label=rid or "Custom model",
        family="sdxl",
        tag="custom",
        blurb="Custom model — not in the curated catalog. Assumed to be SDXL; settings aren't pre-tuned.",
        vram_base_gb=7.5,
    )
