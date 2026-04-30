from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

CallbackType = Optional[Callable[[int, int, object], None]]


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


class PipelineManager:
    """Load and manage Stable Diffusion pipelines with caching."""

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
        self._current_model_id: Optional[str] = None
        self._current_lora_key: Optional[str] = None
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

            self._dispose_pipeline()
            self._pipeline = self._create_pipeline(model_id, lora_config)
            self._current_model_id = model_id
            self._current_lora_key = lora_key
            return self._device_info

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
        progress_callback: CallbackType = None,
    ) -> Image.Image:
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("Pipeline not loaded")
            pipe = self._pipeline
            device_kind = self._device_info.kind
            generator_device = self._device_info.generator_device

        torch, _StableDiffusionXLPipeline = _lazy_import_diffusers()

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

        images = pipe(
            prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback=progress_callback,
            callback_steps=1,
        ).images
        return images[0]

    def upscale_image(self, base_image: Image.Image, prompt: str) -> Image.Image:
        torch, _, StableDiffusionUpscalePipeline = _lazy_import_diffusers(include_upscaler=True)
        device_info = self._select_device(torch)

        local_dir = self._model_dir_for_id(self.upscale_model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Loading AI upscaler %s on %s", self.upscale_model_id, device_info.description)

        up_pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.upscale_model_id,
            cache_dir=str(local_dir),
        )

        try:
            target = device_info.kind
            if device_info.kind == "dml":
                import torch_directml  # type: ignore

                dml_device = torch_directml.device()
                up_pipe = up_pipe.to(dml_device)
            else:
                up_pipe = up_pipe.to(target)

            with torch.inference_mode():
                result = up_pipe(prompt=prompt, image=base_image)
            return result.images[0]
        finally:
            del up_pipe

    def current_model_id(self) -> Optional[str]:
        with self._lock:
            return self._current_model_id

    def ensure_lora_cached(self, source: str) -> None:
        if not source:
            return
        try:
            self._resolve_lora_source(source)
        except Exception as exc:
            self.logger.warning("Failed to cache LoRA %s: %s", source, exc)
            raise

    # ------------------------------------------------------------------
    def _create_pipeline(self, model_id: str, lora_config: LoRAConfig | None):
        torch, StableDiffusionXLPipeline = _lazy_import_diffusers()
        device_info = self._select_device(torch)
        self.logger.info("Loading model %s on %s", model_id, device_info.description)

        local_dir = self._model_dir_for_id(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "cache_dir": str(local_dir),
            "local_files_only": False,
        }
        if model_id == "RunDiffusion/Juggernaut-XL-v9":
            kwargs["variant"] = "fp16"
            kwargs["torch_dtype"] = torch.float16
        elif device_info.kind == "cuda":
            kwargs["torch_dtype"] = torch.float16

        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)
        if device_info.kind == "dml":
            import torch_directml  # type: ignore

            dml_device = torch_directml.device()
            pipe = pipe.to(dml_device)
        else:
            pipe = pipe.to(device_info.kind)

        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._apply_scheduler(pipe)
        self._apply_attention_optimizations(pipe)
        self._apply_lora(pipe, lora_config)

        with self._lock:
            self._device_info = device_info

        return pipe

    def _apply_scheduler(self, pipe) -> None:
        try:
            from diffusers import DPMSolverMultistepScheduler  # type: ignore

            if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
            else:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
        except Exception as exc:  # pragma: no cover - scheduler fallback
            self.logger.warning("Failed to configure DPMSolver scheduler: %s", exc)

    def _apply_attention_optimizations(self, pipe) -> None:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            self.logger.debug("attention_slicing unavailable")

        try:
            pipe.enable_vae_slicing()
        except Exception:
            self.logger.debug("vae_slicing unavailable")

        try:
            pipe.enable_vae_tiling()
        except Exception:
            self.logger.debug("vae_tiling unavailable")

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            self.logger.debug("xformers attention unavailable")

    def _apply_lora(self, pipe, lora_config: LoRAConfig | None) -> None:
        if lora_config is None:
            return

        try:
            import peft  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - defensive path
            raise RuntimeError(
                "LoRA support requires the 'peft' package. Install it with `pip install peft`."
            ) from exc

        source_path = self._resolve_lora_source(lora_config.source)

        try:
            adapter_name = "_active_lora"
            load_kwargs: dict[str, object] = {"adapter_name": adapter_name}
            if lora_config.weight_name:
                load_kwargs["weight_name"] = lora_config.weight_name
            pipe.load_lora_weights(source_path, **load_kwargs)
            pipe.fuse_lora(adapter_name=adapter_name, lora_scale=lora_config.scale)
            self.logger.info(
                "Applied LoRA %s (weight=%s, scale=%.2f)",
                lora_config.source,
                lora_config.weight_name or "auto",
                lora_config.scale,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load LoRA '{lora_config.source}': {exc}") from exc

    def _dispose_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                del self._pipeline
            except Exception:
                pass
            self._pipeline = None
            self._current_model_id = None
            self._current_lora_key = None

    def _select_device(self, torch_module):
        if torch_module.cuda.is_available():
            try:
                gpu_name = torch_module.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA GPU"
            return DeviceInfo(kind="cuda", description=f"CUDA GPU: {gpu_name}", generator_device="cuda")

        try:
            import torch_directml  # type: ignore

            dml_device = torch_directml.device()
            return DeviceInfo(kind="dml", description="DirectML GPU", generator_device="cpu")
        except Exception:
            pass

        return DeviceInfo(kind="cpu", description="CPU", generator_device="cpu")

    def _model_dir_for_id(self, model_id: str) -> Path:
        safe_name = self._safe_name(model_id)
        return self.models_root / safe_name

    def _resolve_lora_source(self, source: str) -> str:
        source_path = Path(source)
        if source_path.exists():
            return str(source_path)

        target_dir = self.lora_root / self._safe_name(source)
        if not target_dir.exists() or not any(target_dir.iterdir()):
            try:
                from huggingface_hub import snapshot_download  # type: ignore
            except ImportError as exc:  # pragma: no cover - defensive path
                raise RuntimeError(
                    "huggingface_hub package is required for downloading LoRA presets. Install it via `pip install huggingface_hub`."
                ) from exc

            self.logger.info("Downloading LoRA %s into %s", source, target_dir)
            try:
                snapshot_download(
                    repo_id=source,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=self._hf_token,
                    resume_download=True,
                )
            except Exception as exc:
                auth_hint = ""
                text = str(exc)
                if "401" in text or "Invalid username" in text:
                    auth_hint = (
                        " Authentication failed. Run `huggingface-cli login` in this environment "
                        "or place a token in config/hf_token.txt."
                    )
                raise RuntimeError(f"Failed to download LoRA '{source}'.{auth_hint} {exc}") from exc
        return str(target_dir)

    @staticmethod
    def _safe_name(value: str) -> str:
        return value.replace("/", "__").replace(":", "_")

    def _load_hf_token(self, models_root: Path) -> str | None:
        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if token and token.strip():
            return token.strip()

        root_dir = models_root.parent
        candidate_paths = [
            root_dir / "config" / "hf_token.txt",
            root_dir / "hf_token.txt",
            root_dir / "token.txt",
        ]
        for token_file in candidate_paths:
            if not token_file.is_file():
                continue
            try:
                contents = token_file.read_text(encoding="utf-8").strip()
                if contents:
                    return contents
            except Exception as exc:  # pragma: no cover - defensive path
                self.logger.warning("Failed to read Hugging Face token file %s: %s", token_file, exc)
        return None


def _lazy_import_diffusers(include_upscaler: bool = False):
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionXLPipeline  # type: ignore
        if include_upscaler:
            from diffusers import StableDiffusionUpscalePipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Required packages are missing. Install diffusers[torch] pillow transformers accelerate safetensors"
        ) from exc

    if include_upscaler:
        return torch, StableDiffusionXLPipeline, StableDiffusionUpscalePipeline  # type: ignore[name-defined]
    return torch, StableDiffusionXLPipeline
