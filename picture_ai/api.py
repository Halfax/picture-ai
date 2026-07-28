"""Programmatic API for picture-ai — the callable surface the Tkinter app is
one client of, rather than the only way in.

Before this existed the supported headless path was "drive PipelineManager
directly", which meant every caller re-derived model loading, family rules,
frame-count legality and encoding. This module owns those so a caller can say
what it wants and nothing else:

    from picture_ai import api

    img = api.generate_image("a lantern in fog")
    img.save("still.png")

    clip = api.generate_video("a lantern swinging in fog, slow dolly in",
                              seconds=4)
    clip.save("clip.mp4")            # NVENC-encoded

The engine is a process-wide singleton so consecutive calls reuse the loaded
model instead of paying a multi-GB reload per call; `Engine` is available if a
caller wants its own. Generation is serialised by a lock — one GPU, and two
concurrent denoise loops on 16 GB is an OOM, not a speedup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

from .model_catalog import MODEL_CATALOG, VIDEO_FAMILIES, catalog_get, is_video_model
from .pipeline_manager import LoRAConfig, PipelineManager, VideoResult
from .video_export import DEFAULT_QUALITY, encode_video

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = ROOT / "models_cache"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

DEFAULT_IMAGE_MODEL = "RunDiffusion/Juggernaut-XL-v9"
DEFAULT_VIDEO_MODEL = "Lightricks/LTX-Video"

ProgressFn = Optional[Callable[[int, int], None]]


@dataclass(slots=True)
class ImageResult:
    image: Image.Image
    model_id: str
    seed: Optional[int]
    prompt: str

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)
        return path


@dataclass(slots=True)
class Clip:
    """A generated clip. Frames are kept as PIL images and encoding is a
    separate call, so the same generation can be saved at several qualities,
    thumbnailed, or handed to a caller that wants the frames themselves."""

    frames: list
    fps: int
    width: int
    height: int
    model_id: str
    seed: Optional[int]
    prompt: str

    @property
    def duration(self) -> float:
        return len(self.frames) / float(self.fps or 1)

    def save(
        self,
        path: str | Path,
        *,
        codec: str = "auto",
        quality: int = DEFAULT_QUALITY,
        fps: Optional[int] = None,
    ) -> Path:
        return encode_video(
            self.frames, path, fps=fps or self.fps, codec=codec, quality=quality
        )

    def save_frames(self, directory: str | Path, prefix: str = "frame") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.frames):
            frame.save(directory / f"{prefix}_{i:05d}.png")
        return directory

    def thumbnail(self) -> Image.Image:
        """First frame — the cheap stand-in wherever a still is expected."""
        return self.frames[0]


@dataclass
class Engine:
    """Owns one PipelineManager and serialises access to it."""

    models_root: Path = DEFAULT_MODELS_ROOT
    _pm: Optional[PipelineManager] = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def manager(self) -> PipelineManager:
        if self._pm is None:
            self._pm = PipelineManager(models_root=self.models_root, logger=logger)
        return self._pm

    def load(self, model_id: str, *, lora: LoRAConfig | None = None):
        """Load a model, returning its DeviceInfo. Cheap if already loaded."""
        with self._lock:
            return self.manager.ensure_pipeline(model_id, lora)

    @property
    def loaded_model(self) -> Optional[str]:
        return self.manager._current_model_id  # noqa: SLF001

    def generate_image(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
        lora: LoRAConfig | None = None,
        progress: ProgressFn = None,
        **kwargs,
    ) -> ImageResult:
        model = model or DEFAULT_IMAGE_MODEL
        if is_video_model(model):
            raise ValueError(
                f"{model} is a video model — use generate_video() (or pass a "
                "still-image model)."
            )
        info = catalog_get(model)
        steps = info.recommended.steps if steps is None else steps
        guidance_scale = (
            info.recommended.cfg_scale if guidance_scale is None else guidance_scale
        )
        with self._lock:
            self.load(model, lora=lora)
            if sampler:
                self.manager.set_sampler(sampler)
            cb = (lambda step, total, _x: progress(step, steps)) if progress else None
            image = self.manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=cb,
                **kwargs,
            )
        return ImageResult(image=image, model_id=model, seed=seed, prompt=prompt)

    def generate_video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: str = "",
        width: int = 704,
        height: int = 480,
        seconds: Optional[float] = None,
        num_frames: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        progress: ProgressFn = None,
    ) -> Clip:
        """Generate a clip. Give either `seconds` (converted using the model's
        native fps) or an explicit `num_frames`; both get snapped to a frame
        count the model's VAE actually accepts."""
        model = model or DEFAULT_VIDEO_MODEL
        info = catalog_get(model)
        if info.family not in VIDEO_FAMILIES:
            raise ValueError(
                f"{model} is a {info.family} model and makes still images — "
                "use generate_image(), or pass a video model such as "
                f"{DEFAULT_VIDEO_MODEL}."
            )
        spec = info.video
        if num_frames is None:
            num_frames = round((seconds or 0) * spec.fps) if seconds else spec.num_frames
        steps = info.recommended.steps if steps is None else steps
        guidance_scale = (
            info.recommended.cfg_scale if guidance_scale is None else guidance_scale
        )

        with self._lock:
            self.load(model)
            cb = (lambda step, total, _x: progress(step, steps)) if progress else None
            result: VideoResult = self.manager.generate_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=cb,
            )
        return Clip(
            frames=result.frames,
            fps=result.fps,
            width=result.width,
            height=result.height,
            model_id=model,
            seed=seed,
            prompt=prompt,
        )

    def unload(self) -> None:
        """Release the GPU. Useful before handing the card to another process."""
        with self._lock:
            if self._pm is not None:
                self._pm._dispose_pipeline()  # noqa: SLF001
                self._pm._current_model_id = None  # noqa: SLF001


# ----------------------------------------------------------------------
# Module-level convenience — a shared engine so repeat calls reuse the model
# ----------------------------------------------------------------------
_default_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _default_engine
    with _engine_lock:
        if _default_engine is None:
            _default_engine = Engine()
        return _default_engine


def generate_image(prompt: str, **kwargs) -> ImageResult:
    return get_engine().generate_image(prompt, **kwargs)


def generate_video(prompt: str, **kwargs) -> Clip:
    return get_engine().generate_video(prompt, **kwargs)


def encode(frames: Iterable, path: str | Path, **kwargs) -> Path:
    """Encode an existing frame sequence (PIL images or arrays) to a video."""
    return encode_video(frames, path, **kwargs)


def list_models(kind: str = "all") -> list[dict]:
    """Catalog summary. `kind` is 'all', 'image' or 'video'."""
    out = []
    for repo_id, info in MODEL_CATALOG.items():
        is_video = info.family in VIDEO_FAMILIES
        if kind == "video" and not is_video:
            continue
        if kind == "image" and is_video:
            continue
        entry = {
            "id": repo_id,
            "label": info.label,
            "family": info.family,
            "tag": info.tag,
            "blurb": info.blurb,
            "kind": "video" if is_video else "image",
            "gated": info.gated,
            "vram_base_gb": info.vram_base_gb,
            "recommended": {
                "steps": info.recommended.steps,
                "guidance_scale": info.recommended.cfg_scale,
                "note": info.recommended.note,
            },
        }
        if is_video and info.video:
            entry["video"] = {
                "fps": info.video.fps,
                "default_frames": info.video.num_frames,
                "max_frames": info.video.max_frames,
                "frame_rule": f"n % {info.video.frame_modulus} == {info.video.frame_offset}",
                "dim_multiple": info.video.dim_multiple,
            }
        out.append(entry)
    return out


__all__ = [
    "Engine", "Clip", "ImageResult", "LoRAConfig",
    "generate_image", "generate_video", "encode", "list_models", "get_engine",
    "DEFAULT_IMAGE_MODEL", "DEFAULT_VIDEO_MODEL", "DEFAULT_OUTPUT_DIR",
]
