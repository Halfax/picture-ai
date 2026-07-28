from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .model_catalog import VIDEO_FAMILIES, VideoSpec, catalog_get

CallbackType = Optional[Callable[[int, int, object], None]]

# Reference mode constants
REF_MODE_IMG2IMG = "img2img"
REF_MODE_FACE = "face"
REF_MODE_STYLE = "style"

# IP-Adapter configuration
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHTS = {
    REF_MODE_FACE: "ip-adapter-plus-face_sdxl_vit-h.safetensors",
    REF_MODE_STYLE: "ip-adapter-plus_sdxl_vit-h.safetensors",
}

# Samplers (UI label → diffusers scheduler class name + from_config kwargs).
# Names match the A1111 conventions written into PNG metadata.
# `euler_at_final` defends against an IndexError on the last step when a
# checkpoint ships a scheduler_config.json with a non-default
# `final_sigmas_type` (hit by John6666/lustify-* on 2026-05-15). Unlike
# `lower_order_final`, this fires regardless of step count — diffusers gates
# `lower_order_final` behind `len(timesteps) < 15`, so it doesn't help at 20+
# steps. `euler_at_final=True` matches A1111's "Karras" sampler behavior.
_DPM_SAFE = {"euler_at_final": True}
SAMPLERS: dict[str, tuple[str, dict]] = {
    "DPM++ 2M Karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True, **_DPM_SAFE},
    ),
    "DPM++ SDE Karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, **_DPM_SAFE},
    ),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
}
DEFAULT_SAMPLER = "DPM++ 2M Karras"


@dataclass(slots=True)
class LoRAConfig:
    source: str
    weight_name: str | None = None
    scale: float = 1.0

    def key(self) -> str:
        return f"{self.source}|{self.weight_name or ''}|{self.scale:.3f}"


@dataclass(slots=True)
class DeviceInfo:
    kind: str
    description: str
    generator_device: str


@dataclass(slots=True)
class VideoResult:
    """A generated clip, still as frames. Encoding is a separate step
    (`video_export.encode_video`) so the frames can also be inspected,
    thumbnailed or re-encoded at a different quality without regenerating."""

    frames: list
    fps: int
    width: int
    height: int
    seed: Optional[int] = None
    model_id: str = ""

    @property
    def duration(self) -> float:
        return len(self.frames) / float(self.fps or 1)


def _round_video_dims(spec: VideoSpec, width: int, height: int) -> tuple[int, int]:
    """Snap to the dimension multiple the model's VAE requires."""
    m = max(1, spec.dim_multiple)
    return (max(m, round(width / m) * m), max(m, round(height / m) * m))


def _round_frame_count(spec: VideoSpec, num_frames: int) -> int:
    """Snap to a legal frame count: `n % modulus == offset`, clamped to the
    model's maximum. For LTX that is 8k+1 — 121 is legal, 120 is rejected.

    Rounds to the NEAREST legal count, not down: flooring turns a request for
    8 frames into 1, i.e. silently hands back a still image instead of a clip.
    """
    mod, off = max(1, spec.frame_modulus), spec.frame_offset
    floor_n = int(num_frames) - ((int(num_frames) - off) % mod)
    candidates = [c for c in (floor_n, floor_n + mod) if c >= max(off, 1)]
    if not candidates:
        candidates = [max(off, 1)]
    best = min(candidates, key=lambda c: (abs(c - int(num_frames)), c))
    return max(max(off, 1), min(best, spec.max_frames))


class PipelineManager:
    """Load and manage Stable Diffusion XL pipelines with caching."""

    def __init__(
        self,
        models_root: Path,
        *,
        upscale_model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.models_root = models_root
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.lora_root = self.models_root / "loras"
        self.lora_root.mkdir(parents=True, exist_ok=True)
        self.upscale_model_id = upscale_model_id
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._hf_token = self._load_hf_token(models_root)

        self._pipeline = None
        self._img2img_pipeline = None
        self._inpaint_pipeline = None
        self._upscale_pipeline = None
        self._compel = None
        self._current_model_id: Optional[str] = None
        self._current_lora_key: Optional[str] = None
        self._current_sampler: Optional[str] = None
        self._current_family: str = "sdxl"
        self._current_ip_adapter_mode: Optional[str] = None
        self._device_info = DeviceInfo(kind="cpu", description="CPU", generator_device="cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_pipeline(self, model_id: str, lora_config: LoRAConfig | None) -> DeviceInfo:
        lora_key = lora_config.key() if lora_config else None
        with self._lock:
            if (
                self._pipeline is not None
                and self._current_model_id == model_id
                and self._current_lora_key == lora_key
            ):
                return self._device_info

            # Same model, different LoRA: swap the LoRA on the live pipeline
            # instead of reloading the whole model from disk.
            if (
                self._pipeline is not None
                and self._current_model_id == model_id
                and self._current_family == "sdxl"
            ):
                try:
                    self._swap_lora(self._pipeline, lora_config)
                    self._current_lora_key = lora_key
                    return self._device_info
                except Exception as exc:
                    self.logger.warning(
                        "In-place LoRA swap failed (%s); falling back to full reload", exc
                    )

            self._dispose_pipeline()
            self._pipeline = self._create_pipeline(model_id, lora_config)
            self._img2img_pipeline = None
            self._inpaint_pipeline = None
            self._current_model_id = model_id
            self._current_lora_key = lora_key
            return self._device_info

    def ensure_lora_cached(self, source: str) -> None:
        """Download a LoRA repo/weights into the local lora cache so they are
        available offline when the user selects them later."""
        source = source.strip()
        if not source:
            return
        # If source is already a local path with safetensors files, nothing to download
        local_path = Path(source)
        if local_path.is_dir() and any(local_path.glob("*.safetensors")):
            return
        if local_path.is_file():
            return

        safe = source.replace("/", "__").replace(":", "_")
        dest = self.lora_root / safe
        dest.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=source,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=self._hf_token,
                resume_download=True,
            )
            self.logger.info("LoRA cached: %s -> %s", source, dest)
        except Exception as exc:
            self.logger.warning("Failed to cache LoRA %s: %s", source, exc)
            raise

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
        strength: float = 0.7,
        ref_mode: str = REF_MODE_IMG2IMG,
        progress_callback: CallbackType = None,
        reference_images: list[str] | None = None,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_strength: float = 0.35,
    ) -> Image.Image:
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("Pipeline not loaded")
            pipe = self._pipeline
            device_kind = self._device_info.kind
            generator_device = self._device_info.generator_device

        torch = _lazy_import_torch()

        generator = None
        if seed is not None:
            gen_device = generator_device if device_kind in {"cuda", "cpu"} else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        self.logger.info(
            "Generating image | prompt=%s | steps=%s | size=%sx%s | model=%s",
            prompt[:60],
            steps,
            width,
            height,
            self._current_model_id,
        )

        # Build the progress callback wrapper for diffusers >= 0.25
        cb_kwargs = {}
        if progress_callback is not None:
            cb_kwargs["callback_on_step_end"] = lambda _pipe, step, _ts, cb_data: (
                progress_callback(step, 0, None),
                cb_data,
            )[1]

        # A video model loaded into the still-image path would otherwise fail
        # deep inside the pipeline with an opaque shape error.
        if self._current_family in VIDEO_FAMILIES:
            raise RuntimeError(
                f"{self._current_model_id} is a video model — call generate_video()."
            )

        # SD3 / Flux: a single text2img path. compel, IP-Adapter,
        # reference-image blending and hires fix are SDXL-only in this build.
        if self._current_family in ("sd3", "flux"):
            if reference_images:
                self.logger.warning(
                    "%s does not support reference images here — ignoring them",
                    self._current_family,
                )
            return self._generate_dit_text2img(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cb_kwargs=cb_kwargs,
            )

        # Handle reference images
        if reference_images and ref_mode == REF_MODE_FACE:
            # Face mode: composite reference onto a generated scene, then
            # harmonise with a light img2img pass so the face is physically
            # present in the output.
            raw_ref = self._load_references_raw(reference_images)
            if raw_ref is not None:
                return self._generate_composite_refine(
                    ref_image=raw_ref,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                    cb_kwargs=cb_kwargs,
                )

        if reference_images and ref_mode == REF_MODE_STYLE:
            # Style mode: IP-Adapter captures overall aesthetics well.
            raw_ref = self._load_references_raw(reference_images)
            if raw_ref is not None:
                return self._generate_ip_adapter(
                    ref_mode=ref_mode,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_image=raw_ref,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    ip_scale=strength,
                    generator=generator,
                    cb_kwargs=cb_kwargs,
                )

        # Img2Img: resize/blend to output dimensions as init canvas
        init_image = self._load_and_blend_references(reference_images, width, height)
        if init_image is not None:
            return self._generate_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                cb_kwargs=cb_kwargs,
            )

        # Text-to-image path
        call_kwargs = {
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            **cb_kwargs,
        }

        embeds = self._encode_prompts(prompt, negative_prompt)
        try:
            if embeds is not None:
                images = pipe(**embeds, **call_kwargs).images
            else:
                call_kwargs["negative_prompt"] = negative_prompt or None
                images = pipe(prompt, **call_kwargs).images
            base_image = images[0]
        except Exception as exc:
            self.logger.exception("Generation failed: %s", exc)
            raise RuntimeError("Image generation failed") from exc

        if hires_fix and hires_scale > 1.01:
            return self._hires_pass(
                base_image=base_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=hires_scale,
                strength=hires_strength,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        return base_image

    def generate_video(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        width: int = 704,
        height: int = 480,
        num_frames: Optional[int] = None,
        steps: int = 40,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        progress_callback: CallbackType = None,
    ) -> VideoResult:
        """Generate a clip with a video-family pipeline (LTX).

        Frame count and dimensions are rounded to what the model's 3D VAE
        actually accepts (see VideoSpec) rather than passed through — an
        illegal value is a hard pipeline error, and silently rejecting the
        user's number is worse than snapping it to the nearest legal one.
        """
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("Pipeline not loaded")
            if self._current_family not in VIDEO_FAMILIES:
                raise RuntimeError(
                    f"{self._current_model_id} is a {self._current_family} model, "
                    "which makes still images — call generate_image() instead, or "
                    "load a video model (e.g. Lightricks/LTX-Video)."
                )
            pipe = self._pipeline
            model_id = self._current_model_id
            generator_device = self._device_info.generator_device
            device_kind = self._device_info.kind

        torch = _lazy_import_torch()
        spec = catalog_get(model_id).video or VideoSpec()

        req_w, req_h, req_f = width, height, num_frames
        width, height = _round_video_dims(spec, width, height)
        frames_n = _round_frame_count(spec, num_frames if num_frames else spec.num_frames)
        if (width, height) != (req_w, req_h):
            self.logger.info(
                "Rounded size %sx%s -> %sx%s (model needs multiples of %s)",
                req_w, req_h, width, height, spec.dim_multiple,
            )
        if req_f and frames_n != req_f:
            self.logger.info(
                "Rounded frame count %s -> %s (model needs %sk+%s)",
                req_f, frames_n, spec.frame_modulus, spec.frame_offset,
            )

        generator = None
        if seed is not None:
            gen_device = generator_device if device_kind in {"cuda", "cpu"} else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        cb_kwargs = {}
        if progress_callback is not None:
            cb_kwargs["callback_on_step_end"] = lambda _pipe, step, _ts, cb_data: (
                progress_callback(step, steps, None),
                cb_data,
            )[1]

        self.logger.info(
            "Generating video | model=%s | %sx%s | %s frames @ %s fps | steps=%s | guidance=%.2f",
            model_id, width, height, frames_n, spec.fps, steps, guidance_scale,
        )
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_frames=frames_n,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
                **cb_kwargs,
            )
        except Exception as exc:
            self.logger.exception("Video generation failed: %s", exc)
            raise RuntimeError(f"Video generation failed: {exc}") from exc

        frames = list(result.frames[0])
        self.logger.info("Generated %d frames", len(frames))
        return VideoResult(
            frames=frames,
            fps=spec.fps,
            width=width,
            height=height,
            seed=seed,
            model_id=model_id or "",
        )

    def upscale_image(self, base_image: Image.Image, prompt: str) -> Image.Image:
        """AI-upscale an image using the SD x4 upscaler pipeline."""
        torch = _lazy_import_torch()

        with self._lock:
            if self._upscale_pipeline is None:
                self._upscale_pipeline = self._create_upscale_pipeline()
            upscale_pipe = self._upscale_pipeline

        # The x4 upscaler expects a low-res input; resize if too large
        max_input = 512
        w, h = base_image.size
        if w > max_input or h > max_input:
            scale = min(max_input / w, max_input / h)
            base_image = base_image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

        base_image = base_image.convert("RGB")

        try:
            result = upscale_pipe(prompt=prompt, image=base_image, num_inference_steps=20).images[0]
            return result
        except Exception as exc:
            self.logger.exception("AI upscale failed: %s", exc)
            raise RuntimeError("AI upscale failed") from exc

    # ------------------------------------------------------------------
    # Prompt encoding (compel — weighted, >77-token SDXL prompts)
    # ------------------------------------------------------------------
    def _get_compel(self):
        if self._compel is not None:
            return self._compel
        if self._pipeline is None:
            return None
        Compel, ReturnedEmbeddingsType = _lazy_import_compel()
        if Compel is None:
            return None
        try:
            self._compel = Compel(
                tokenizer=[self._pipeline.tokenizer, self._pipeline.tokenizer_2],
                text_encoder=[self._pipeline.text_encoder, self._pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
            self.logger.info("Initialized compel for weighted/long SDXL prompts")
            return self._compel
        except Exception as exc:
            self.logger.warning("Failed to init compel: %s", exc)
            return None

    @staticmethod
    def _compel_padding(compel):
        """Build the padding tensor compel's SDXL path can't build for itself.

        `Compel.pad_conditioning_tensors_to_same_length` reads
        `self.conditioning_provider.empty_z`. For SDXL the provider is an
        `EmbeddingsProviderMulti` (two tokenizers / two text encoders), and
        that class has no `empty_z` — only the single-encoder
        `EmbeddingsProvider` does. So the call raises AttributeError, the
        caller below falls back to the raw prompt, and CLIP silently truncates
        at 77 tokens. That is why long prompts appeared to be ignored past the
        first ~77 tokens even though compel is configured with
        `truncate_long_prompts=False`.

        It only fires when the positive and negative prompts land in a
        *different* number of 77-token chunks — the function returns early when
        all conditionings already share a shape — which is why short prompts
        always worked and a long positive with a short negative did not.

        The sub-providers each expose `empty_z`, and the multi-provider
        concatenates along the embedding dim, so the tensor it wanted is just
        their concatenation. Returns None if the layout isn't what we expect,
        which restores the previous (truncating) behaviour rather than failing.
        """
        provider = getattr(compel, "conditioning_provider", None)
        if provider is None or hasattr(provider, "empty_z"):
            return None  # single-encoder path: compel handles it itself
        subs = getattr(provider, "embedding_providers", None)
        if not subs:
            return None
        try:
            torch = _lazy_import_torch()
            parts = [p.empty_z for p in subs]
            if getattr(provider, "concat_along_embedding_dim", True):
                return torch.cat(parts, dim=-1)
            return parts[0]
        except Exception:
            return None

    def _encode_prompts(self, prompt: str, negative_prompt: str) -> dict | None:
        """Return prompt_embeds kwargs dict if compel is available, else None.
        Caller falls back to raw prompt= / negative_prompt= when None is
        returned (which silently truncates >77 tokens)."""
        compel = self._get_compel()
        if compel is None:
            return None
        try:
            cond, pooled = compel(prompt)
            neg_cond, neg_pooled = compel(negative_prompt or "")
            cond, neg_cond = compel.pad_conditioning_tensors_to_same_length(
                [cond, neg_cond], precomputed_padding=self._compel_padding(compel)
            )
            return {
                "prompt_embeds": cond,
                "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": neg_cond,
                "negative_pooled_prompt_embeds": neg_pooled,
            }
        except Exception as exc:
            self.logger.warning("compel encode failed, falling back to truncated prompt: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Reference image helpers
    # ------------------------------------------------------------------
    def _load_references_raw(
        self,
        reference_images: list[str] | None,
    ) -> Image.Image | None:
        """Load reference images at their original resolution for IP-Adapter.
        If multiple images are provided, blend them at a common size without
        forcing the output resolution.  Returns None if nothing loaded."""
        if not reference_images:
            return None

        loaded: list[Image.Image] = []
        for rp in reference_images:
            try:
                img = Image.open(rp).convert("RGB")
                loaded.append(img)
            except Exception as exc:
                self.logger.warning("Failed to load reference image %s: %s", rp, exc)

        if not loaded:
            return None

        if len(loaded) == 1:
            return loaded[0]  # keep original size

        # Multiple images: blend at the size of the largest one
        max_w = max(img.width for img in loaded)
        max_h = max(img.height for img in loaded)
        resized = [img.resize((max_w, max_h), Image.LANCZOS) for img in loaded]
        arrays = [np.asarray(img, dtype=np.float32) for img in resized]
        blended = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(blended, "RGB")

    def _load_and_blend_references(
        self,
        reference_images: list[str] | None,
        target_w: int,
        target_h: int,
    ) -> Image.Image | None:
        """Load up to 3 reference images, resize them to the target
        dimensions, and alpha-blend them equally into a single init image."""
        if not reference_images:
            return None

        loaded: list[Image.Image] = []
        for rp in reference_images:
            try:
                img = Image.open(rp).convert("RGB")
                loaded.append(img)
            except Exception as exc:
                self.logger.warning("Failed to load reference image %s: %s", rp, exc)

        if not loaded:
            return None

        if len(loaded) == 1:
            return loaded[0].resize((target_w, target_h), Image.LANCZOS)

        # Blend multiple images equally: resize all to target, then average
        resized = [img.resize((target_w, target_h), Image.LANCZOS) for img in loaded]
        arrays = [np.asarray(img, dtype=np.float32) for img in resized]
        blended = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(blended, "RGB")

    # ------------------------------------------------------------------
    # Interpret → IP-Adapter  (Face Reference)
    # ------------------------------------------------------------------
    def _generate_composite_refine(
        self,
        *,
        ref_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        strength: float,
        generator,
        cb_kwargs: dict,
    ) -> Image.Image:
        """Two-phase face reference:
        Phase 1 – *Interpret*: run img2img on the reference at high
                  strength so the model redraws pixel-art / low-res input
                  as a detailed character portrait.
        Phase 2 – *Generate*: use the interpreted portrait as an
                  IP-Adapter face reference for a fresh text2img pass.
                  CLIP can now extract real face features from the
                  high-quality interpreted image.
        """
        with self._lock:
            pipe = self._pipeline

        # === Phase 1: Interpret the reference ============================
        self.logger.info(
            "Face ref 1/2: interpreting reference (%s) via img2img…",
            ref_image.size,
        )
        # Upscale to ~512 px on the long side, preserve aspect ratio,
        # round to multiples of 8 (required by VAE).
        orig_w, orig_h = ref_image.size
        scale_factor = 512.0 / max(orig_w, orig_h)
        interp_w = max(8, int(orig_w * scale_factor) // 8 * 8)
        interp_h = max(8, int(orig_h * scale_factor) // 8 * 8)
        ref_for_interp = ref_image.resize((interp_w, interp_h), Image.LANCZOS)

        img2img_pipe = self._get_or_create_img2img_pipeline()

        # Use a portrait-focused prompt with the user's style cues
        interp_prompt = (
            f"detailed character portrait, face close-up, "
            f"high quality, sharp features, {prompt}"
        )
        interp_neg = "blurry, low quality, worst quality, pixelated, blocky"
        if negative_prompt:
            interp_neg = f"{interp_neg}, {negative_prompt}"

        # High strength (0.70) to aggressively transform pixel-art into
        # a real detailed character while keeping general colours / pose.
        interpreted = img2img_pipe(
            interp_prompt,
            negative_prompt=interp_neg,
            image=ref_for_interp,
            strength=0.70,
            num_inference_steps=max(15, steps),
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        self.logger.info("Interpreted character: %s", interpreted.size)

        # === Phase 2: IP-Adapter text2img with interpreted face ==========
        self.logger.info(
            "Face ref 2/2: generating scene with IP-Adapter face ref…",
        )
        self._ensure_ip_adapter_loaded(REF_MODE_FACE)
        pipe.set_ip_adapter_scale(strength)

        call_kwargs = {
            "ip_adapter_image": interpreted,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            **cb_kwargs,
        }
        embeds = self._encode_prompts(prompt, negative_prompt)
        try:
            self.logger.info(
                "IP-Adapter (face) | scale=%.2f | interp_size=%s | "
                "output=%sx%s",
                strength, interpreted.size, width, height,
            )
            if embeds is not None:
                images = pipe(**embeds, **call_kwargs).images
            else:
                call_kwargs["negative_prompt"] = negative_prompt or None
                images = pipe(prompt, **call_kwargs).images
            return images[0]
        except Exception as exc:
            self.logger.exception("IP-Adapter generation failed: %s", exc)
            raise RuntimeError("IP-Adapter generation failed") from exc

    # ------------------------------------------------------------------
    # IP-Adapter helpers  (Style Reference)
    # ------------------------------------------------------------------
    @staticmethod
    def _upscale_for_clip(image: Image.Image, min_size: int = 512) -> Image.Image:
        """Upscale a tiny reference so CLIP can extract meaningful features.
        Uses LANCZOS for smooth upscaling; preserves aspect ratio."""
        w, h = image.size
        if w >= min_size and h >= min_size:
            return image
        scale = max(min_size / w, min_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)

    def _generate_ip_adapter(
        self,
        *,
        ref_mode: str,
        prompt: str,
        negative_prompt: str,
        ref_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        ip_scale: float,
        generator,
        cb_kwargs: dict,
    ) -> Image.Image:
        """Generate using text2img + IP-Adapter conditioning.
        The reference image is used as an 'image prompt' via IP-Adapter
        while the text prompt fully controls the scene composition.
        Tiny references are upscaled so CLIP can extract useful features."""
        with self._lock:
            pipe = self._pipeline

        self._ensure_ip_adapter_loaded(ref_mode)
        pipe.set_ip_adapter_scale(ip_scale)

        # Upscale tiny references so CLIP ViT-H can see them properly
        orig_size = ref_image.size
        ref_image = self._upscale_for_clip(ref_image)

        call_kwargs = {
            "ip_adapter_image": ref_image,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            **cb_kwargs,
        }

        embeds = self._encode_prompts(prompt, negative_prompt)
        mode_label = "face" if ref_mode == REF_MODE_FACE else "style"
        try:
            self.logger.info(
                "Running text2img + IP-Adapter (%s) | ip_scale=%.2f | "
                "ref_orig=%s ref_clip=%s | output=%sx%s",
                mode_label, ip_scale, orig_size,
                ref_image.size, width, height,
            )
            if embeds is not None:
                images = pipe(**embeds, **call_kwargs).images
            else:
                call_kwargs["negative_prompt"] = negative_prompt or None
                images = pipe(prompt, **call_kwargs).images
            return images[0]
        except Exception as exc:
            self.logger.exception("IP-Adapter generation failed: %s", exc)
            raise RuntimeError("IP-Adapter generation failed") from exc

    def _ensure_ip_adapter_loaded(self, ref_mode: str) -> None:
        """Load (or swap) the correct IP-Adapter weights on the main pipeline.
        Skips loading if the requested mode is already active."""
        if self._current_ip_adapter_mode == ref_mode:
            return

        with self._lock:
            pipe = self._pipeline
            if pipe is None:
                raise RuntimeError("Base pipeline not loaded")

            # Unload previous IP-Adapter if any
            if self._current_ip_adapter_mode is not None:
                try:
                    pipe.unload_ip_adapter()
                    self.logger.info("Unloaded previous IP-Adapter")
                except Exception as exc:
                    self.logger.warning("Failed to unload IP-Adapter: %s", exc)

            weight_name = IP_ADAPTER_WEIGHTS.get(ref_mode)
            if weight_name is None:
                self._current_ip_adapter_mode = None
                return

            load_kwargs = {}
            if self._hf_token:
                load_kwargs["token"] = self._hf_token

            try:
                self.logger.info(
                    "Loading IP-Adapter: %s/%s/%s",
                    IP_ADAPTER_REPO, IP_ADAPTER_SUBFOLDER, weight_name,
                )
                pipe.load_ip_adapter(
                    IP_ADAPTER_REPO,
                    subfolder=IP_ADAPTER_SUBFOLDER,
                    weight_name=weight_name,
                    image_encoder_folder="models/image_encoder",
                    **load_kwargs,
                )
                self._current_ip_adapter_mode = ref_mode
                self.logger.info("IP-Adapter loaded: %s", weight_name)
            except Exception as exc:
                self._current_ip_adapter_mode = None
                self.logger.exception("Failed to load IP-Adapter: %s", exc)
                raise RuntimeError(
                    f"Failed to load IP-Adapter ({ref_mode}). "
                    "Make sure you have internet access for the first download. "
                    f"Error: {exc}"
                ) from exc

    def _hires_pass(
        self,
        *,
        base_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        scale: float,
        strength: float,
        steps: int,
        guidance_scale: float,
        generator,
    ) -> Image.Image:
        """A1111-style HiRes Fix: PIL-upscale the base image, then run a
        low-denoise img2img pass on it. Catches the soft-detail problem at
        20-30 steps without re-running full diffusion at high resolution."""
        base_w, base_h = base_image.size
        # SDXL VAE requires multiples of 8.
        new_w = max(8, int(base_w * scale) // 8 * 8)
        new_h = max(8, int(base_h * scale) // 8 * 8)
        upscaled = base_image.resize((new_w, new_h), Image.LANCZOS)
        self.logger.info(
            "HiRes Fix | %sx%s -> %sx%s | strength=%.2f",
            base_w, base_h, new_w, new_h, strength,
        )

        img2img_pipe = self._get_or_create_img2img_pipeline()
        embeds = self._encode_prompts(prompt, negative_prompt)
        call_kwargs = {
            "image": upscaled,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        try:
            if embeds is not None:
                return img2img_pipe(**embeds, **call_kwargs).images[0]
            call_kwargs["negative_prompt"] = negative_prompt or None
            return img2img_pipe(prompt, **call_kwargs).images[0]
        except Exception as exc:
            self.logger.exception("HiRes pass failed: %s", exc)
            # Fall back to the un-refined upscale rather than failing the whole gen.
            return upscaled

    def _generate_dit_text2img(
        self,
        *,
        pipe,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator,
        cb_kwargs: dict,
    ) -> Image.Image:
        """Text-to-image for the DiT families (SD3 / Flux). No compel and no
        prompt-embed path — these pipelines take the raw prompt directly."""
        family = self._current_family
        # Flux's VAE needs dimensions that are multiples of 16; SD3 is fine
        # with the multiples of 8 the UI already enforces.
        if family == "flux":
            width = max(16, width - width % 16)
            height = max(16, height - height % 16)
        call_kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            **cb_kwargs,
        }
        # Flux dev/schnell are guidance-distilled and take no negative prompt.
        if family == "sd3":
            call_kwargs["negative_prompt"] = negative_prompt or None
        try:
            self.logger.info(
                "Running %s text2img | steps=%s | size=%sx%s | guidance=%.2f",
                family, steps, width, height, guidance_scale,
            )
            return pipe(**call_kwargs).images[0]
        except Exception as exc:
            self.logger.exception("%s generation failed: %s", family, exc)
            raise RuntimeError(f"{family} image generation failed: {exc}") from exc

    def _generate_img2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        strength: float,
        generator,
        cb_kwargs: dict,
    ) -> Image.Image:
        """Run img2img generation using the SDXL Img2Img pipeline,
        reusing components from the already-loaded text2img pipeline."""
        img2img_pipe = self._get_or_create_img2img_pipeline()

        # img2img does not accept width/height directly; the output size
        # matches the input image, so the init_image is already resized.
        call_kwargs = {
            "image": init_image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **cb_kwargs,
        }

        embeds = self._encode_prompts(prompt, negative_prompt)
        try:
            self.logger.info(
                "Running SDXL img2img | strength=%.2f | init_size=%s",
                strength,
                init_image.size,
            )
            if embeds is not None:
                images = img2img_pipe(**embeds, **call_kwargs).images
            else:
                call_kwargs["negative_prompt"] = negative_prompt or None
                images = img2img_pipe(prompt, **call_kwargs).images
            return images[0]
        except Exception as exc:
            self.logger.exception("Img2Img generation failed: %s", exc)
            raise RuntimeError("Img2Img generation failed") from exc

    def _get_or_create_inpaint_pipeline(self):
        """Create an SDXL Inpaint pipeline reusing the loaded text2img
        pipeline’s components (shared GPU memory).  Works in ‘legacy’
        latent-blending mode with the standard (non-inpaint) UNet."""
        with self._lock:
            if self._inpaint_pipeline is not None:
                return self._inpaint_pipeline

            if self._pipeline is None:
                raise RuntimeError("Base pipeline not loaded")

            try:
                from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "StableDiffusionXLInpaintPipeline not available. "
                    "Please upgrade diffusers: pip install -U diffusers"
                )

            self._inpaint_pipeline = StableDiffusionXLInpaintPipeline(
                vae=self._pipeline.vae,
                text_encoder=self._pipeline.text_encoder,
                text_encoder_2=self._pipeline.text_encoder_2,
                tokenizer=self._pipeline.tokenizer,
                tokenizer_2=self._pipeline.tokenizer_2,
                unet=self._pipeline.unet,
                scheduler=self._pipeline.scheduler,
            )
            self.logger.info("Created SDXL Inpaint pipeline (shared weights)")
            return self._inpaint_pipeline

    def _get_or_create_img2img_pipeline(self):
        """Create an SDXL Img2Img pipeline by reusing components from the
        loaded text2img pipeline to avoid doubling VRAM usage."""
        with self._lock:
            if self._img2img_pipeline is not None:
                return self._img2img_pipeline

            if self._pipeline is None:
                raise RuntimeError("Base pipeline not loaded")

            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "StableDiffusionXLImg2ImgPipeline not available. "
                    "Please upgrade diffusers: pip install -U diffusers"
                )

            # Build from the components of the already-loaded pipeline
            # so we share the model weights in GPU memory
            self._img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
                vae=self._pipeline.vae,
                text_encoder=self._pipeline.text_encoder,
                text_encoder_2=self._pipeline.text_encoder_2,
                tokenizer=self._pipeline.tokenizer,
                tokenizer_2=self._pipeline.tokenizer_2,
                unet=self._pipeline.unet,
                scheduler=self._pipeline.scheduler,
            )
            self.logger.info("Created SDXL Img2Img pipeline (shared weights)")
            return self._img2img_pipeline

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def _load_hf_token(self, models_root: Path) -> Optional[str]:
        """Load Hugging Face token from project token.txt or environment variables."""
        try:
            token_file = models_root.parent / "token.txt"
            if token_file.exists():
                token = token_file.read_text(encoding="utf-8").strip()
                if token:
                    return token

            for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
                val = os.environ.get(name)
                if val:
                    return val
        except Exception as exc:
            try:
                self.logger.warning("Failed to read Hugging Face token: %s", exc)
            except Exception:
                pass
        return None

    def _model_dir_for_id(self, model_id: str) -> Path:
        """Return a local cache directory path for a given model id."""
        safe_name = model_id.replace("/", "__").replace(":", "_")
        return self.models_root / safe_name

    def _dispose_pipeline(self) -> None:
        """Dispose of the currently loaded pipeline, freeing device memory."""
        torch = _lazy_import_torch()
        # Unload IP-Adapter before disposing
        if self._current_ip_adapter_mode is not None:
            try:
                if self._pipeline is not None:
                    self._pipeline.unload_ip_adapter()
            except Exception:
                pass
            self._current_ip_adapter_mode = None

        # Compel holds references to the old text encoders; drop it.
        self._compel = None
        self._current_family = "sdxl"

        for attr in ("_inpaint_pipeline", "_img2img_pipeline", "_pipeline"):
            try:
                pipe = getattr(self, attr, None)
                if pipe is not None:
                    setattr(self, attr, None)
                    del pipe
            except Exception as exc:
                self.logger.warning("Error disposing %s: %s", attr, exc)

        # Break reference cycles (compel / the shared img2img pipes held
        # encoder refs) so the old model's VRAM is released before the next load.
        gc.collect()

        # Free CUDA cache
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _detect_device(self) -> None:
        """Probe CUDA and record the result on self._device_info."""
        torch = _lazy_import_torch()
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    devname = torch.cuda.get_device_name(0)
                except Exception:
                    devname = "CUDA"
                self._device_info = DeviceInfo(kind="cuda", description=devname, generator_device="cuda")
            else:
                self._device_info = DeviceInfo(kind="cpu", description="CPU", generator_device="cpu")
        except Exception:
            self._device_info = DeviceInfo(kind="cpu", description="CPU", generator_device="cpu")

    def _create_pipeline(self, model_id: str, lora_config: LoRAConfig | None):
        """Create a pipeline for `model_id`, dispatching on its model family
        (sdxl / sd3 / flux — see model_catalog)."""
        info = catalog_get(model_id)
        family = info.family
        self._detect_device()
        self.logger.info(
            "Loading model %s | family=%s | device=%s",
            model_id, family, self._device_info.description,
        )

        if family == "ltx":
            if lora_config is not None:
                self.logger.warning("LoRA is not applied to LTX video models in this build")
            pipe = self._create_ltx_pipeline(model_id)
        elif family == "flux":
            if lora_config is not None:
                self.logger.warning("LoRA is not applied to Flux models in this build")
            pipe = self._create_flux_pipeline(model_id)
        elif family == "sd3":
            if lora_config is not None:
                self.logger.warning("LoRA is not applied to SD3 models in this build")
            pipe = self._create_sd3_pipeline(model_id)
        else:
            pipe = self._create_sdxl_pipeline(model_id, lora_config, info.single_file)

        self._current_family = family
        return pipe

    def _create_sdxl_pipeline(self, model_id: str, lora_config: LoRAConfig | None,
                              single_file: bool):
        """Load an optimized SDXL pipeline — either a diffusers-format repo
        or a single-file .safetensors checkpoint — then set the default
        sampler and apply any LoRA."""
        torch = _lazy_import_torch()
        _SDXLPipeline = _lazy_import_sdxl_pipeline()

        local_dir = self._model_dir_for_id(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)

        if single_file:
            pipe = self._load_sdxl_single_file(_SDXLPipeline, model_id, local_dir)
        else:
            load_kwargs = {
                "cache_dir": str(local_dir),
                "torch_dtype": torch.float16,
                "variant": "fp16",
            }
            if self._hf_token:
                load_kwargs["token"] = self._hf_token
            # Try fp16 diffusers-format; fall back to default precision, then
            # to single-file (covers Civitai-style one-file checkpoints).
            try:
                self.logger.info("Loading SDXL pipeline %s (diffusers, fp16)", model_id)
                pipe = _SDXLPipeline.from_pretrained(model_id, **load_kwargs)
            except Exception:
                self.logger.info("fp16 variant not found, trying default precision")
                load_kwargs.pop("variant", None)
                try:
                    pipe = _SDXLPipeline.from_pretrained(model_id, **load_kwargs)
                except Exception as exc:
                    self.logger.info(
                        "from_pretrained failed for %s (%s); trying single-file",
                        model_id, exc,
                    )
                    try:
                        pipe = self._load_sdxl_single_file(_SDXLPipeline, model_id, local_dir)
                    except Exception as exc2:
                        self.logger.exception("Failed to load SDXL model %s: %s", model_id, exc2)
                        raise

        # Move to device
        try:
            pipe = pipe.to(self._device_info.generator_device)
        except Exception:
            try:
                pipe = pipe.to("cpu")
            except Exception:
                self.logger.warning("Failed to move pipeline to device; leaving as-loaded")

        self._apply_optimizations(pipe)
        # Default sampler (UI may override via set_sampler before each run)
        self._apply_sampler(pipe, DEFAULT_SAMPLER)
        if lora_config is not None:
            self._apply_lora(pipe, lora_config)
        return pipe

    def _load_sdxl_single_file(self, pipeline_cls, model_id: str, local_dir: Path):
        """Load an SDXL pipeline from a single .safetensors checkpoint — a
        Hugging Face repo id holding one file, a direct URL, or a local path.
        This is what unlocks Civitai-style checkpoints (Pony, Illustrious,
        DreamShaper, ...) that aren't published in diffusers folder format."""
        torch = _lazy_import_torch()
        load_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": str(local_dir),
        }
        if self._hf_token:
            load_kwargs["token"] = self._hf_token
        self.logger.info("Loading SDXL pipeline %s (single-file checkpoint)", model_id)
        return pipeline_cls.from_single_file(model_id, **load_kwargs)

    def _create_sd3_pipeline(self, model_id: str):
        """Load Stable Diffusion 3.5 (StableDiffusion3Pipeline). SD3.5 Medium
        is ~10 GiB at fp16 and fits the 5080 directly."""
        torch = _lazy_import_torch()
        StableDiffusion3Pipeline = _lazy_import_sd3_pipeline()

        local_dir = self._model_dir_for_id(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs = {
            "cache_dir": str(local_dir),
            "torch_dtype": torch.float16,
        }
        if self._hf_token:
            load_kwargs["token"] = self._hf_token

        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            self.logger.exception("Failed to load SD3 model %s: %s", model_id, exc)
            raise RuntimeError(
                f"Failed to load {model_id}. SD3.5 is gated — accept its license "
                "on huggingface.co and put a valid token in token.txt. "
                f"Error: {exc}"
            ) from exc

        try:
            pipe = pipe.to(self._device_info.generator_device)
        except Exception:
            self.logger.warning("Failed to move SD3 pipeline to device; leaving as-loaded")
        self._apply_dit_optimizations(pipe)
        return pipe

    def _create_flux_pipeline(self, model_id: str):
        """Load Flux NF4-quantized so it fits 16 GB. The transformer and the
        T5 text encoder load 4-bit; CLIP and the VAE stay full precision
        (they're small). enable_model_cpu_offload keeps peak VRAM in budget."""
        torch = _lazy_import_torch()
        FluxPipeline = _lazy_import_flux_pipeline()
        quant_config = _build_flux_quant_config(torch)
        if quant_config is None:
            self.logger.warning(
                "Flux: bitsandbytes / diffusers quantization unavailable — "
                "loading unquantized, which will likely exceed 16 GB VRAM"
            )

        local_dir = self._model_dir_for_id(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs = {
            "cache_dir": str(local_dir),
            "torch_dtype": torch.bfloat16,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        if self._hf_token:
            load_kwargs["token"] = self._hf_token

        try:
            pipe = FluxPipeline.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            self.logger.exception("Failed to load Flux model %s: %s", model_id, exc)
            raise RuntimeError(
                f"Failed to load {model_id}. Flux is gated (accept the license "
                "on huggingface.co) and needs `bitsandbytes` installed for "
                f"4-bit quantization. Error: {exc}"
            ) from exc

        # With a quantized pipeline, do NOT call .to('cuda'); CPU offload
        # streams components on demand and keeps peak residency under 16 GB.
        try:
            pipe.enable_model_cpu_offload()
            self.logger.info("Flux: enabled model CPU offload")
        except Exception as exc:
            self.logger.warning("Flux: enable_model_cpu_offload failed: %s", exc)
            try:
                pipe = pipe.to(self._device_info.generator_device)
            except Exception:
                self.logger.warning("Flux: could not move pipeline to device")
        self._apply_dit_optimizations(pipe)
        return pipe

    def _create_ltx_pipeline(self, model_id: str):
        """Load LTX-Video. The 2B transformer is small (~4 GiB at bf16); the
        T5-XXL text encoder is the problem (~9.5 GiB at bf16), so this uses
        `enable_model_cpu_offload()` like the Flux path — the encoder is
        resident only while encoding the prompt, then streamed back out before
        the denoise loop. Peak stays well inside 16 GB."""
        torch = _lazy_import_torch()
        LTXPipeline, _ = _lazy_import_ltx_pipelines()

        local_dir = self._model_dir_for_id(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs = {
            "cache_dir": str(local_dir),
            "torch_dtype": torch.bfloat16,
        }
        if self._hf_token:
            load_kwargs["token"] = self._hf_token

        try:
            pipe = LTXPipeline.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            self.logger.exception("Failed to load LTX model %s: %s", model_id, exc)
            raise RuntimeError(
                f"Failed to load {model_id}. Fetch its diffusers weights first with "
                f"`venv\\Scripts\\python.exe scripts\\fetch_ltx.py` (the repo root "
                f"holds ~254 GB of checkpoint variants that must NOT be pulled). "
                f"Error: {exc}"
            ) from exc

        try:
            pipe.enable_model_cpu_offload()
            self.logger.info("LTX: enabled model CPU offload")
        except Exception as exc:
            self.logger.warning("LTX: enable_model_cpu_offload failed: %s", exc)
            try:
                pipe = pipe.to(self._device_info.generator_device)
            except Exception:
                self.logger.warning("LTX: could not move pipeline to device")
        # 🔴 VAE tiling is not optional here. Video latents are frames x H x W
        # and the whole stack is decoded to pixels AFTER the last sampling
        # step, so an untiled decode spikes past anything the denoise loop
        # used and dies at the very end of an otherwise-successful run. Same
        # failure shape as DECISIONS §22 rule A on the Betelgeuse iGPU.
        self._apply_dit_optimizations(pipe)
        return pipe

    def _apply_dit_optimizations(self, pipe) -> None:
        """VAE slicing/tiling for the DiT pipelines (SD3 / Flux). These keep
        the VAE decode of a 1024px image inside budget; the SDXL-only tweaks
        (DPM++ sampler swap, xformers UNet attention) do not apply here."""
        for name, fn in (("VAE slicing", "enable_vae_slicing"),
                         ("VAE tiling", "enable_vae_tiling")):
            try:
                getattr(pipe, fn)()
                self.logger.info("Enabled %s", name)
            except Exception:
                pass

    def _apply_optimizations(self, pipe) -> None:
        """Apply speed optimizations to the SDXL pipeline. On a 16 GB card
        SDXL fp16 fits with headroom, so the low-VRAM crutches (attention
        slicing, VAE slicing/tiling) are deliberately NOT enabled — under
        PyTorch 2.x SDPA they only slow inference, and attention slicing
        breaks IP-Adapter processor loading."""
        torch = _lazy_import_torch()
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            self.logger.info("UNet set to channels_last memory format")
        except Exception:
            pass

    def _apply_sampler(self, pipe, sampler: str) -> None:
        """Swap pipe.scheduler to the named sampler (in SAMPLERS table)."""
        spec = SAMPLERS.get(sampler) or SAMPLERS[DEFAULT_SAMPLER]
        cls_name, kwargs = spec
        try:
            from diffusers import (  # type: ignore
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                UniPCMultistepScheduler,
            )
            cls_map = {
                "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
                "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
                "UniPCMultistepScheduler": UniPCMultistepScheduler,
            }
            cls = cls_map[cls_name]
            pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
            if cls is DPMSolverMultistepScheduler:
                self._patch_dpm_init_step_index(pipe.scheduler)
            self._current_sampler = sampler
            self.logger.info("Sampler set: %s", sampler)
        except Exception as exc:
            self.logger.warning("Failed to set sampler %s: %s", sampler, exc)

    @staticmethod
    def _patch_dpm_init_step_index(scheduler) -> None:
        """Override `_init_step_index` so the first step picks the FIRST
        occurrence of a duplicated timestep, not the second.

        Diffusers' default heuristic picks index_candidates[1] when a
        timestep appears more than once — intended for img2img mid-schedule
        starts. With karras sigmas on certain checkpoints (Lustify), two
        adjacent karras sigmas round to the same integer timestep, so the
        very first t2i step starts at index 1 instead of 0. step_index then
        overshoots by 1 across the run and the final step accesses
        sigmas[n+1], throwing IndexError. Pipelines that need the original
        behaviour set begin_index explicitly — we honour that path."""
        import types
        import torch

        def _init_step_index(self, timestep):
            if self._begin_index is not None:
                self._step_index = self._begin_index
                return
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            candidates = (self.timesteps == timestep).nonzero()
            if len(candidates) == 0:
                self._step_index = len(self.timesteps) - 1
            else:
                self._step_index = candidates[0].item()

        scheduler._init_step_index = types.MethodType(_init_step_index, scheduler)

    def set_sampler(self, sampler: str) -> None:
        """Swap the scheduler on the active pipeline; no model reload needed.
        Cheap; safe to call before every generation."""
        with self._lock:
            if self._pipeline is None:
                return
            if self._current_family != "sdxl":
                return  # SD3 / Flux keep their native flow-matching scheduler
            if self._current_sampler == sampler:
                return
            self._apply_sampler(self._pipeline, sampler)
            # Img2img / inpaint pipes were built from text2img components and
            # captured the original scheduler reference at creation time; rebind.
            for shared in (self._img2img_pipeline, self._inpaint_pipeline):
                if shared is not None:
                    shared.scheduler = self._pipeline.scheduler

    def _apply_lora(self, pipe, lora_config: LoRAConfig) -> None:
        """Load and fuse LoRA weights into the pipeline."""
        source = lora_config.source.strip()
        if not source:
            return

        self.logger.info("Loading LoRA from %s (weight=%s, scale=%.2f)",
                         source, lora_config.weight_name, lora_config.scale)

        try:
            # Determine the actual path/repo to load from
            local_path = Path(source)
            safe = source.replace("/", "__").replace(":", "_")
            cached_dir = self.lora_root / safe

            load_kwargs = {}
            if lora_config.weight_name:
                load_kwargs["weight_name"] = lora_config.weight_name
            if self._hf_token:
                load_kwargs["token"] = self._hf_token

            if local_path.is_dir():
                pipe.load_lora_weights(str(local_path), **load_kwargs)
            elif local_path.is_file():
                pipe.load_lora_weights(
                    str(local_path.parent),
                    weight_name=local_path.name,
                )
            elif cached_dir.is_dir() and any(cached_dir.glob("*.safetensors")):
                pipe.load_lora_weights(str(cached_dir), **load_kwargs)
            else:
                pipe.load_lora_weights(source, **load_kwargs)

            pipe.fuse_lora(lora_scale=lora_config.scale)
            self.logger.info("LoRA fused successfully (scale=%.2f)", lora_config.scale)
        except Exception as exc:
            self.logger.warning("Failed to load/fuse LoRA: %s", exc)
            raise

    def _swap_lora(self, pipe, lora_config: LoRAConfig | None) -> None:
        """Replace (or remove) the fused LoRA on a live SDXL pipeline
        without reloading the base weights."""
        if self._current_lora_key is not None:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
            self.logger.info("Unfused previous LoRA")
        if lora_config is not None:
            self._apply_lora(pipe, lora_config)

    def _create_upscale_pipeline(self):
        """Create the SD x4 upscaler pipeline."""
        torch = _lazy_import_torch()
        try:
            from diffusers import StableDiffusionUpscalePipeline  # type: ignore
        except ImportError:
            raise RuntimeError("StableDiffusionUpscalePipeline not available")

        load_kwargs = {"torch_dtype": torch.float16}
        if self._hf_token:
            load_kwargs["token"] = self._hf_token

        local_dir = self._model_dir_for_id(self.upscale_model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["cache_dir"] = str(local_dir)

        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.upscale_model_id, **load_kwargs
        )
        pipe = pipe.to(self._device_info.generator_device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self.logger.info("Loaded upscale pipeline: %s", self.upscale_model_id)
        return pipe


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------
def _lazy_import_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed: %s" % exc)


def _lazy_import_sdxl_pipeline():
    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
        return StableDiffusionXLPipeline
    except ImportError:
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore
            return StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Could not import a compatible diffusers pipeline: %s" % exc)


def _lazy_import_compel():
    try:
        from compel import Compel, ReturnedEmbeddingsType  # type: ignore
        return Compel, ReturnedEmbeddingsType
    except ImportError:
        return None, None


def _lazy_import_sd3_pipeline():
    try:
        from diffusers import StableDiffusion3Pipeline  # type: ignore
        return StableDiffusion3Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "StableDiffusion3Pipeline not available — upgrade diffusers: %s" % exc
        )


def _lazy_import_ltx_pipelines():
    """(LTXPipeline, LTXImageToVideoPipeline) — text2video and image2video."""
    try:
        from diffusers import LTXImageToVideoPipeline, LTXPipeline  # type: ignore
        return LTXPipeline, LTXImageToVideoPipeline
    except ImportError as exc:
        raise RuntimeError(
            "LTXPipeline not available — upgrade diffusers (needs >=0.32): %s" % exc
        )


def _lazy_import_flux_pipeline():
    try:
        from diffusers import FluxPipeline  # type: ignore
        return FluxPipeline
    except ImportError as exc:
        raise RuntimeError(
            "FluxPipeline not available — upgrade diffusers: %s" % exc
        )


def _build_flux_quant_config(torch):
    """Build a PipelineQuantizationConfig that loads the Flux transformer and
    the T5 text encoder in 4-bit NF4. Returns None if bitsandbytes or the
    diffusers quantization API isn't available — the caller then loads
    unquantized (which will likely OOM on 16 GB, surfaced as a clear error)."""
    try:
        from diffusers import BitsAndBytesConfig as DiffusersBnB  # type: ignore
        from transformers import BitsAndBytesConfig as TransformersBnB  # type: ignore
        try:
            from diffusers.quantizers import PipelineQuantizationConfig  # type: ignore
        except Exception:
            from diffusers import PipelineQuantizationConfig  # type: ignore
        import bitsandbytes  # noqa: F401  — ensure the 4-bit backend is installed
    except Exception:
        return None
    return PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersBnB(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "text_encoder_2": TransformersBnB(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    )
