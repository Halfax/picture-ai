from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List

DEFAULT_MODEL_IDS: list[str] = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "SG161222/RealVisXL_V5.0",
    "RunDiffusion/Juggernaut-XL-v9",
]
DEFAULT_STYLE = "Photoreal portrait"


@dataclass(slots=True)
class UserSettings:
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 35
    guidance_scale: float = 4.5
    style: str = DEFAULT_STYLE
    seed: str = ""
    model_id: str = DEFAULT_MODEL_IDS[0]
    dark_mode: bool = True
    lora_enabled: bool = False
    lora_source: str = ""
    lora_weight_name: str = ""
    lora_scale: float = 0.8
    size_preset: str = "Custom"
    lora_preset: str = "Custom"
    ref_strength: float = 0.7
    ref_mode: str = "img2img"
    # Up to 3 reference image paths (local file paths). Empty list by default.
    reference_images: list[str] = None


class SettingsStore:
    """Persist user preferences and custom model list to disk."""

    def __init__(self, settings_path: Path, models_path: Path, *, logger: logging.Logger) -> None:
        self.settings_path = settings_path
        self.models_path = models_path
        self.logger = logger
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # User settings
    # ------------------------------------------------------------------
    def load_user_settings(self) -> UserSettings:
        if not self.settings_path.is_file():
            # default should have reference_images as an empty list
            defaults = asdict(UserSettings())
            defaults["reference_images"] = []
            return UserSettings(**defaults)  # type: ignore[arg-type]

        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            merged = {**asdict(UserSettings()), **data}  # type: ignore[arg-type]
            # Ensure reference_images is a list of up to 3 strings
            refs = merged.get("reference_images") or []
            if not isinstance(refs, list):
                refs = []
            refs = [str(r) for r in refs if r]
            refs = refs[:3]
            merged["reference_images"] = refs
            return UserSettings(**merged)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Failed to read settings file %s: %s", self.settings_path, exc)
            return UserSettings()

    def save_user_settings(self, settings: UserSettings) -> None:
        try:
            payload = asdict(settings)
            # Ensure reference_images is a list (not None)
            if payload.get("reference_images") is None:
                payload["reference_images"] = []
            self.settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.error("Failed to write settings file %s: %s", self.settings_path, exc)

    # ------------------------------------------------------------------
    # Model list persistence
    # ------------------------------------------------------------------
    def load_model_ids(self) -> list[str]:
        if not self.models_path.is_file():
            return DEFAULT_MODEL_IDS.copy()

        try:
            raw = json.loads(self.models_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                merged = self._merge_unique(DEFAULT_MODEL_IDS, raw)
                return merged
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to read model list %s: %s", self.models_path, exc)
        return DEFAULT_MODEL_IDS.copy()

    def save_model_ids(self, model_ids: Iterable[str]) -> None:
        try:
            payload = list(dict.fromkeys(model_ids))
            self.models_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            self.logger.error("Failed to write model list %s: %s", self.models_path, exc)

    def ensure_model_id(self, model_ids: list[str], new_model: str) -> list[str]:
        new_model = new_model.strip()
        if not new_model:
            return model_ids
        if new_model not in model_ids:
            model_ids.append(new_model)
            self.save_model_ids(model_ids)
        return model_ids

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_unique(defaults: Iterable[str], custom: Iterable[Any]) -> list[str]:
        seen: dict[str, None] = {}
        for mid in list(defaults) + [c for c in custom if isinstance(c, str)]:
            if mid not in seen:
                seen[mid] = None
        return list(seen.keys())
