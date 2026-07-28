from __future__ import annotations

import json
import random
import threading
import uuid
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from .logging_utils import configure_logging
from .pipeline_manager import (
    LoRAConfig,
    PipelineManager,
    REF_MODE_IMG2IMG,
    REF_MODE_FACE,
    REF_MODE_STYLE,
    SAMPLERS,
    DEFAULT_SAMPLER,
    _round_frame_count,
)
from .model_catalog import catalog_get, is_video_model
from .settings_store import DEFAULT_MODEL_IDS, SettingsStore, UserSettings
from .video_export import DEFAULT_QUALITY, available_encoders, encode_video

REF_MODE_LABELS = {
    "Img2Img (Regenerate)": REF_MODE_IMG2IMG,
    "Face Reference (IP-Adapter)": REF_MODE_FACE,
    "Style Reference (IP-Adapter)": REF_MODE_STYLE,
}
REF_MODE_LABELS_INV = {v: k for k, v in REF_MODE_LABELS.items()}

STYLE_PRESETS: dict[str, tuple[str, str]] = {
    "Photoreal portrait": (
        "(photorealistic:1.2) portrait, (detailed skin texture:1.3), (sharp focus:1.1), realistic lighting, 85mm, detailed hands, 8k",
        "cartoon, anime, illustration, plastic skin, airbrushed, low quality, blurry, deformed hands, extra fingers, missing fingers, disfigured, bad anatomy",
    ),
    "Raw photo (analog)": (
        "(raw photo:1.3), analog film grain, natural skin pores, ambient lighting, candid composition, kodak portra 400, slight chromatic aberration",
        "(over-processed:1.2), HDR, oversaturated, plastic skin, airbrushed, cartoon, illustration, 3d render, deformed hands, bad anatomy",
    ),
    "Editorial film": (
        "(editorial film style:1.2), magazine photography, soft rim lighting, shallow depth of field, 35mm grain, fashion photography composition",
        "amateur, snapshot, harsh flash, oversaturated, plastic skin, cartoon, anime, deformed hands, bad anatomy",
    ),
    "Cinematic": (
        "(cinematic wide shot:1.2), dramatic lighting, film still, 35mm, depth of field, highly detailed, realistic hands, anamorphic lens flare",
        "cartoon, anime, flat shading, low quality, noisy, deformed hands, extra fingers, missing fingers, bad anatomy",
    ),
    "NSFW photoreal": (
        "(raw photo:1.3), (detailed skin texture:1.3), natural body proportions, soft natural lighting, sharp focus, 8k",
        "(plastic skin:1.2), airbrushed, cartoon, anime, illustration, 3d render, deformed hands, extra fingers, missing fingers, bad anatomy, mutated",
    ),
    "Anime": (
        "(anime illustration:1.2), clean lines, vibrant colors, highly detailed, expressive pose, sharp linework",
        "(photorealistic:1.2), realistic skin, grainy, noisy, low quality, deformed hands, extra fingers, missing fingers, bad anatomy",
    ),
    "Fantasy art": (
        "(fantasy painting:1.2), epic composition, dramatic lighting, intricate detail, painterly brush strokes, vibrant palette",
        "photograph, photorealistic, low quality, deformed hands, extra fingers, missing fingers, bad anatomy, watermark",
    ),
}

QUALITY_BOOSTER_POS = "(masterpiece:1.2), (best quality:1.2), (highly detailed:1.1), sharp focus"
QUALITY_BOOSTER_NEG = "(worst quality:1.4), (low quality:1.4), lowres, jpeg artifacts, watermark, signature, text, blurry"

SIZE_PRESETS: list[tuple[str, tuple[int, int]]] = [
    ("Square 1024×1024", (1024, 1024)),
    ("Square 768×768", (768, 768)),
    ("Portrait 832×1216", (832, 1216)),
    ("Portrait 768×1152", (768, 1152)),
    ("Landscape 1216×832", (1216, 832)),
    ("Landscape 1152×768", (1152, 768)),
]
SIZE_PRESET_CUSTOM = "Custom"

DEFAULT_LORA_PRESETS: list[dict[str, object]] = [
    {
        "label": "Fabricated Reality",
        "source": "ostris/fabricated-reality-sdxl",
        "weight_name": "fabricated_reality_sdxl_v14.safetensors",
        "scale": 0.8,
    },
    {
        "label": "Objective Reality",
        "source": "ostris/objective-reality",
        "weight_name": "objectiveReality_v20.safetensors",
        "scale": 0.75,
    },
    {
        "label": "Face Helper XL",
        "source": "ostris/face-helper-sdxl-lora",
        "weight_name": "face_xl_v0_1.safetensors",
        "scale": 0.7,
    },
]
LORA_PRESET_CUSTOM = "Custom"


class PictureAIApp(tk.Tk):
    def __init__(
        self,
        *,
        logger,
        settings_store: SettingsStore,
        pipeline_manager: PipelineManager,
    ) -> None:
        super().__init__()
        self.title("Halfax Image Generator")
        self.geometry("960x760")

        self.logger = logger
        self.settings_store = settings_store
        self.pipeline_manager = pipeline_manager

        self.cache_root = Path.cwd() / "image_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._prune_image_cache()

        self.lora_presets = self._load_lora_presets()

        self.current_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._is_generating = False
        self._is_ai_upscaling = False
        self._last_generation_metadata: Optional[str] = None
        self._progress_total = 30

        # Video state. `_clip_photos` holds pre-scaled PhotoImages for every
        # frame: building them once up front is what makes playback smooth —
        # converting a PIL frame per tick cannot keep 24 fps in Tk. A
        # reference to each must be retained or Tk garbage-collects the image
        # out from under the label and the preview goes blank.
        self.current_clip = None
        self._clip_photos: list[ImageTk.PhotoImage] = []
        self._clip_index = 0
        self._is_playing = False
        self._play_job: Optional[str] = None
        self._is_video_mode = False

        self.model_ids = self.settings_store.load_model_ids()
        self.user_settings = self.settings_store.load_user_settings()

        # Image and video want very different resolutions (1024² vs 704x480),
        # so the two sizes are tracked separately and each mode restores its
        # own. Persisted as width/height and video_width/video_height; whichever
        # mode is inactive keeps its remembered value rather than being
        # overwritten by the active one.
        self._image_size = (self.user_settings.width, self.user_settings.height)
        self._video_size = (self.user_settings.video_width, self.user_settings.video_height)

        self._is_applying_size_preset = False
        self._is_applying_lora_preset = False
        self._lora_preset_errors: dict[str, str] = {}

        self._build_ui()
        self._apply_user_settings()
        self._start_lora_prefetch()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")

        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.guidance_var = tk.DoubleVar(value=self.user_settings.guidance_scale)
        self.size_preset_var = tk.StringVar(value=self.user_settings.size_preset or SIZE_PRESET_CUSTOM)
        self.lora_preset_var = tk.StringVar(value=self.user_settings.lora_preset or LORA_PRESET_CUSTOM)
        self.lora_enabled_var = tk.BooleanVar(value=self.user_settings.lora_enabled)
        self.lora_source_var = tk.StringVar(value=self.user_settings.lora_source)
        self.lora_weight_var = tk.StringVar(value=self.user_settings.lora_weight_name)
        self.lora_scale_var = tk.DoubleVar(value=self.user_settings.lora_scale)

        self.prompt_entry = ttk.Entry(top_frame)
        self._add_labeled_row(top_frame, row=0, label="Positive prompt:", widget=self.prompt_entry, span=3)

        self.negative_prompt_entry = ttk.Entry(top_frame)
        self._add_labeled_row(top_frame, row=1, label="Negative prompt:", widget=self.negative_prompt_entry, span=3)

        self.style_var = tk.StringVar(value=self.user_settings.style)
        self.style_combobox = ttk.Combobox(top_frame, textvariable=self.style_var, state="readonly")
        self.style_combobox["values"] = tuple(STYLE_PRESETS.keys())
        self.style_combobox.bind("<<ComboboxSelected>>", self._on_style_preset_changed)
        self._add_labeled_row(top_frame, row=2, label="Style preset:", widget=self.style_combobox)

        # Output mode. This is the discoverable entry point to video: without
        # it, video existed only as one more entry in a 12-long model dropdown
        # with nothing marking it as different, so there was no way to find it
        # without already knowing. Picking a mode filters the model list.
        mode_frame = ttk.Frame(top_frame)
        mode_frame.grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=(15, 0), pady=(5, 0))
        ttk.Label(mode_frame, text="Output:").pack(side=tk.LEFT)
        self.output_mode_var = tk.StringVar(value="Image")
        self._syncing_output_mode = False
        for label in ("Image", "Video"):
            ttk.Radiobutton(
                mode_frame, text=label, value=label,
                variable=self.output_mode_var,
                command=self._on_output_mode_selected,
            ).pack(side=tk.LEFT, padx=(8, 0))

        self.model_var = tk.StringVar(value=self.user_settings.model_id)
        self.model_combobox = ttk.Combobox(top_frame, textvariable=self.model_var, state="normal")
        self.model_combobox["values"] = tuple(self.model_ids)
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_changed)
        self._add_labeled_row(top_frame, row=3, label="Model:", widget=self.model_combobox)

        # One-line tag + usage hint for the selected model (from model_catalog).
        self.model_info_label = ttk.Label(
            top_frame, text="", foreground="#7c7c8a", wraplength=720, justify=tk.LEFT
        )
        self.model_info_label.grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=(5, 0), pady=(2, 0))

        numeric_frame = ttk.Frame(top_frame)
        numeric_frame.grid(row=5, column=0, columnspan=4, pady=(10, 0), sticky=tk.EW)
        numeric_frame.columnconfigure(11, weight=1)

        ttk.Label(numeric_frame, text="Steps:").grid(row=0, column=0, sticky=tk.W)
        self.steps_var = tk.IntVar(value=self.user_settings.steps)
        ttk.Spinbox(numeric_frame, from_=10, to=100, textvariable=self.steps_var, width=5).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(numeric_frame, text="Width:").grid(row=0, column=2, padx=(15, 0), sticky=tk.W)
        self.width_var = tk.IntVar(value=self.user_settings.width)
        self.width_entry = ttk.Entry(numeric_frame, textvariable=self.width_var, width=6)
        self.width_entry.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(numeric_frame, text="Height:").grid(row=0, column=4, padx=(15, 0), sticky=tk.W)
        self.height_var = tk.IntVar(value=self.user_settings.height)
        self.height_entry = ttk.Entry(numeric_frame, textvariable=self.height_var, width=6)
        self.height_entry.grid(row=0, column=5, sticky=tk.W)

        ttk.Label(numeric_frame, text="Size preset:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.size_preset_combobox = ttk.Combobox(
            numeric_frame,
            state="readonly",
            textvariable=self.size_preset_var,
            values=[label for label, _ in SIZE_PRESETS] + [SIZE_PRESET_CUSTOM],
        )
        self.size_preset_combobox.grid(row=1, column=1, columnspan=4, sticky=tk.W, pady=(6, 0))
        self.size_preset_combobox.bind("<<ComboboxSelected>>", self._on_size_preset_changed)

        ttk.Label(numeric_frame, text="Seed:").grid(row=0, column=6, padx=(15, 0), sticky=tk.W)
        self.seed_var = tk.StringVar(value=self.user_settings.seed)
        ttk.Entry(numeric_frame, textvariable=self.seed_var, width=10).grid(row=0, column=7, sticky=tk.W)
        ttk.Button(numeric_frame, text="Random", command=self._on_random_seed).grid(row=0, column=8, padx=(5, 0))

        ttk.Label(numeric_frame, text="CFG:").grid(row=0, column=9, padx=(15, 0), sticky=tk.W)
        ttk.Spinbox(
            numeric_frame,
            from_=1.0,
            to=12.0,
            increment=0.1,
            textvariable=self.guidance_var,
            width=5,
        ).grid(row=0, column=10, sticky=tk.W)

        ttk.Label(numeric_frame, text="Sampler:").grid(row=1, column=6, padx=(15, 0), sticky=tk.W, pady=(6, 0))
        sampler_default = self.user_settings.sampler if self.user_settings.sampler in SAMPLERS else DEFAULT_SAMPLER
        self.sampler_var = tk.StringVar(value=sampler_default)
        self.sampler_combobox = ttk.Combobox(
            numeric_frame,
            textvariable=self.sampler_var,
            values=tuple(SAMPLERS.keys()),
            state="readonly",
            width=18,
        )
        self.sampler_combobox.grid(row=1, column=7, columnspan=4, sticky=tk.W, pady=(6, 0))
        self.sampler_combobox.bind("<<ComboboxSelected>>", lambda _e: self._save_settings())

        ttk.Button(top_frame, text="Performance preset", command=self._on_performance_preset).grid(row=6, column=0, pady=(10, 0), sticky=tk.W)
        self.dark_mode_var = tk.BooleanVar(value=self.user_settings.dark_mode)
        ttk.Checkbutton(top_frame, text="Dark UI", variable=self.dark_mode_var, command=self._on_theme_toggle).grid(row=6, column=1, pady=(10, 0), sticky=tk.W)
        self.quality_booster_var = tk.BooleanVar(value=self.user_settings.quality_booster)
        ttk.Checkbutton(
            top_frame,
            text="Quality booster",
            variable=self.quality_booster_var,
            command=self._save_settings,
        ).grid(row=6, column=2, pady=(10, 0), sticky=tk.W)
        ttk.Button(
            top_frame,
            text="Recommended settings",
            command=self._on_apply_recommended,
        ).grid(row=6, column=3, pady=(10, 0), sticky=tk.W)

        hires_frame = ttk.Frame(top_frame)
        hires_frame.grid(row=7, column=0, columnspan=4, pady=(8, 0), sticky=tk.W)
        self.hires_frame = hires_frame
        self.hires_fix_var = tk.BooleanVar(value=self.user_settings.hires_fix)
        ttk.Checkbutton(
            hires_frame,
            text="HiRes Fix (two-pass)",
            variable=self.hires_fix_var,
            command=self._save_settings,
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(hires_frame, text="Scale:").grid(row=0, column=1, padx=(15, 4), sticky=tk.W)
        self.hires_scale_var = tk.DoubleVar(value=self.user_settings.hires_scale)
        ttk.Spinbox(
            hires_frame,
            from_=1.10, to=2.00, increment=0.05,
            textvariable=self.hires_scale_var, width=6,
            command=self._save_settings,
        ).grid(row=0, column=2, sticky=tk.W)
        ttk.Label(hires_frame, text="Denoise:").grid(row=0, column=3, padx=(15, 4), sticky=tk.W)
        self.hires_strength_var = tk.DoubleVar(value=self.user_settings.hires_strength)
        ttk.Spinbox(
            hires_frame,
            from_=0.10, to=0.60, increment=0.05,
            textvariable=self.hires_strength_var, width=6,
            command=self._save_settings,
        ).grid(row=0, column=4, sticky=tk.W)

        # Video controls share row 7 with the HiRes controls: the two modes are
        # mutually exclusive, so _update_mode_controls() grids one and
        # grid_remove()s the other rather than renumbering the whole layout.
        video_frame = ttk.LabelFrame(top_frame, text="Video")
        video_frame.grid(row=7, column=0, columnspan=4, pady=(8, 0), sticky=tk.EW)
        self.video_frame = video_frame

        ttk.Label(video_frame, text="Length (s):").grid(row=0, column=0, sticky=tk.W, padx=(5, 4))
        self.video_seconds_var = tk.DoubleVar(value=self.user_settings.video_seconds)
        ttk.Spinbox(
            video_frame, from_=0.5, to=10.0, increment=0.5,
            textvariable=self.video_seconds_var, width=6,
            command=self._on_video_length_changed,
        ).grid(row=0, column=1, sticky=tk.W)
        self.video_seconds_var.trace_add("write", lambda *_: self._on_video_length_changed())

        self.video_frames_label = ttk.Label(video_frame, text="", foreground="#7c7c8a")
        self.video_frames_label.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        ttk.Label(video_frame, text="Codec:").grid(row=1, column=0, sticky=tk.W, padx=(5, 4), pady=(6, 5))
        self.video_codec_var = tk.StringVar(value=self.user_settings.video_codec)
        codec_choices = ["auto"] + [
            c for c in ("h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264")
            if c in available_encoders()
        ]
        self.video_codec_combo = ttk.Combobox(
            video_frame, textvariable=self.video_codec_var, values=codec_choices,
            state="readonly", width=12,
        )
        self.video_codec_combo.grid(row=1, column=1, sticky=tk.W, pady=(6, 5))
        self.video_codec_combo.bind("<<ComboboxSelected>>", lambda _e: self._save_settings())

        ttk.Label(video_frame, text="Quality (lower=better):").grid(
            row=1, column=2, sticky=tk.W, padx=(12, 4), pady=(6, 5)
        )
        self.video_quality_var = tk.IntVar(value=self.user_settings.video_quality)
        ttk.Spinbox(
            video_frame, from_=1, to=51, increment=1,
            textvariable=self.video_quality_var, width=5,
            command=self._save_settings,
        ).grid(row=1, column=3, sticky=tk.W, pady=(6, 5))

        action_frame = ttk.Frame(top_frame)
        action_frame.grid(row=8, column=0, columnspan=4, pady=(10, 0), sticky=tk.W)

        ttk.Button(action_frame, text="Estimate VRAM", command=self._on_estimate_vram).grid(row=0, column=0, padx=(0, 10))
        self.generate_button = ttk.Button(action_frame, text="Generate", command=self.on_generate_clicked)
        self.generate_button.grid(row=0, column=1, padx=(0, 10))
        self.upscale_button = ttk.Button(action_frame, text="Upscale 2x", command=self.on_upscale_clicked, state=tk.DISABLED)
        self.upscale_button.grid(row=0, column=2, padx=(0, 10))
        self.ai_upscale_button = ttk.Button(action_frame, text="AI Upscale", command=self.on_ai_upscale_clicked, state=tk.DISABLED)
        self.ai_upscale_button.grid(row=0, column=3, padx=(0, 10))
        self.save_button = ttk.Button(action_frame, text="Save Image", command=self.on_save_clicked, state=tk.DISABLED)
        self.save_button.grid(row=0, column=4)
        self.save_video_button = ttk.Button(
            action_frame, text="Save Video", command=self.on_save_video_clicked, state=tk.DISABLED
        )
        self.save_video_button.grid(row=0, column=5, padx=(10, 0))

        lora_frame = ttk.LabelFrame(top_frame, text="LoRA (optional)")
        lora_frame.grid(row=9, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))
        lora_frame.columnconfigure(1, weight=1)
        self.lora_frame = lora_frame

        ttk.Checkbutton(
            lora_frame,
            text="Enable LoRA",
            variable=self.lora_enabled_var,
            command=self._on_lora_toggle,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(lora_frame, text="Preset:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.lora_preset_combobox = ttk.Combobox(
            lora_frame,
            state="readonly",
            textvariable=self.lora_preset_var,
            values=[preset["label"] for preset in self.lora_presets] + [LORA_PRESET_CUSTOM],
        )
        self.lora_preset_combobox.grid(row=1, column=1, sticky=tk.EW, pady=(5, 0))
        self.lora_preset_combobox.bind("<<ComboboxSelected>>", self._on_lora_preset_selected)

        ttk.Label(lora_frame, text="Source (repo or path):").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.lora_source_entry = ttk.Entry(lora_frame, textvariable=self.lora_source_var)
        self.lora_source_entry.grid(row=2, column=1, sticky=tk.EW, pady=(5, 0))

        ttk.Label(lora_frame, text="Weight name (optional):").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.lora_weight_entry = ttk.Entry(lora_frame, textvariable=self.lora_weight_var)
        self.lora_weight_entry.grid(row=3, column=1, sticky=tk.EW, pady=(5, 0))

        ttk.Label(lora_frame, text="Scale:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.lora_scale_spin = ttk.Spinbox(
            lora_frame,
            from_=0.1,
            to=2.0,
            increment=0.1,
            textvariable=self.lora_scale_var,
            width=6,
        )
        self.lora_scale_spin.grid(row=4, column=1, sticky=tk.W, pady=(5, 0))

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(top_frame, orient=tk.HORIZONTAL, mode="determinate", variable=self.progress_var)
        self.progress_bar.grid(row=10, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))

        self.status_label = ttk.Label(top_frame, text="Model not loaded", foreground="gray")
        self.status_label.grid(row=11, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        # Reference images UI (up to 3)
        ref_frame = ttk.LabelFrame(top_frame, text="Reference images (up to 3)")
        ref_frame.grid(row=12, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))
        ref_frame.columnconfigure(0, weight=1)
        self.ref_frame = ref_frame

        self.reference_paths: list[str] = ["", "", ""]
        self._ref_thumb_imgs: list[Optional[ImageTk.PhotoImage]] = [None, None, None]

        for i in range(3):
            btn = ttk.Button(ref_frame, text=f"Add / Change #{i+1}", command=lambda i=i: self._choose_reference_image(i))
            btn.grid(row=0, column=i, padx=5)
            rem = ttk.Button(ref_frame, text="Remove", command=lambda i=i: self._remove_reference_image(i))
            rem.grid(row=1, column=i, padx=5)
            lbl = ttk.Label(ref_frame, text="(empty)", anchor=tk.CENTER)
            lbl.grid(row=2, column=i, padx=5)
            # store label widgets for thumbnail updates
            if not hasattr(self, "_ref_labels"):
                self._ref_labels = []
            self._ref_labels.append(lbl)

        mode_row = ttk.Frame(ref_frame)
        mode_row.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(5, 0))
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        saved_label = REF_MODE_LABELS_INV.get(self.user_settings.ref_mode, "Img2Img (Regenerate)")
        self.ref_mode_var = tk.StringVar(value=saved_label)
        ref_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.ref_mode_var,
            values=list(REF_MODE_LABELS.keys()),
            state="readonly",
            width=30,
        )
        ref_mode_combo.pack(side=tk.LEFT, padx=(5, 0))

        strength_row = ttk.Frame(ref_frame)
        strength_row.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(5, 5))
        ttk.Label(strength_row, text="Strength / IP scale:").pack(side=tk.LEFT)
        self.ref_strength_var = tk.DoubleVar(value=self.user_settings.ref_strength)
        ttk.Spinbox(
            strength_row,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.ref_strength_var,
            width=6,
        ).pack(side=tk.LEFT, padx=(5, 0))
        self._ref_strength_hint = ttk.Label(strength_row, text="")
        self._ref_strength_hint.pack(side=tk.LEFT, padx=(10, 0))
        self.ref_mode_var.trace_add("write", lambda *_: self._update_ref_mode_hint())
        self._update_ref_mode_hint()

        image_frame = ttk.Frame(self, padding=10)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(image_frame, text="No image yet", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Playback strip. Tk has no video widget, so the preview is the frame
        # sequence cycled through the same label at the model's fps — which is
        # all a "video preview" needs to be here, and it costs no dependency.
        # Packed only while a clip is loaded.
        self.playback_frame = ttk.Frame(image_frame)
        self.play_button = ttk.Button(self.playback_frame, text="Play", command=self.on_play_pause, width=7)
        self.play_button.pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(
            self.playback_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_scrub
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.frame_counter_label = ttk.Label(self.playback_frame, text="", width=22)
        self.frame_counter_label.pack(side=tk.LEFT)

    def _update_ref_mode_hint(self) -> None:
        """Update the hint text next to the strength spinner based on selected mode."""
        label = self.ref_mode_var.get()
        mode = REF_MODE_LABELS.get(label, REF_MODE_IMG2IMG)
        hints = {
            REF_MODE_IMG2IMG: "(low = subtle change, high = redraw closely)",
            REF_MODE_FACE: "(how strongly the interpreted face guides the new scene)",
            REF_MODE_STYLE: "(how strongly the style influences the new scene)",
        }
        self._ref_strength_hint.config(text=hints.get(mode, ""))

    def _add_labeled_row(self, frame: ttk.Frame, *, row: int, label: str, widget: ttk.Widget, span: int = 1) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=(5 if row else 0, 0))
        widget.grid(row=row, column=1, columnspan=span, sticky=tk.EW, padx=(5, 0), pady=(5 if row else 0, 0))
        frame.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def _apply_user_settings(self) -> None:
        self.prompt_entry.delete(0, tk.END)
        self.prompt_entry.insert(0, self.user_settings.prompt)
        self.negative_prompt_entry.delete(0, tk.END)
        self.negative_prompt_entry.insert(0, self.user_settings.negative_prompt)
        if self.user_settings.style in STYLE_PRESETS:
            self.style_var.set(self.user_settings.style)
        else:
            self.style_var.set(next(iter(STYLE_PRESETS)))

        if self.user_settings.model_id not in self.model_ids:
            self.model_ids.append(self.user_settings.model_id)
        self.model_combobox["values"] = tuple(self.model_ids)
        self.model_var.set(self.user_settings.model_id)

        self.guidance_var.set(self.user_settings.guidance_scale)
        self.size_preset_var.set(self.user_settings.size_preset or SIZE_PRESET_CUSTOM)
        self.lora_preset_var.set(self.user_settings.lora_preset or LORA_PRESET_CUSTOM)
        self.lora_enabled_var.set(self.user_settings.lora_enabled)
        self.lora_source_var.set(self.user_settings.lora_source)
        self.lora_weight_var.set(self.user_settings.lora_weight_name)
        self.lora_scale_var.set(self.user_settings.lora_scale)

        # Load reference images paths, mode and strength from settings
        self.ref_strength_var.set(self.user_settings.ref_strength)
        saved_label = REF_MODE_LABELS_INV.get(self.user_settings.ref_mode, "Img2Img (Regenerate)")
        self.ref_mode_var.set(saved_label)
        self._update_ref_mode_hint()
        refs = self.user_settings.reference_images or []
        # Normalize to length 3
        refs = (refs + [""] * 3)[:3]
        self.reference_paths = refs
        self._update_reference_previews()

        self._apply_theme(self.user_settings.dark_mode)
        self._sync_size_preset_to_dimensions(self.width_var.get(), self.height_var.get())
        self._sync_lora_preset_to_fields()
        self._update_lora_controls()

        self._update_model_info()
        # Start in the mode the saved model implies. initial=True: this is a
        # boot, not a mode switch, so nothing is remembered or overwritten.
        self._update_mode_controls(initial=True)
        if not self._is_video_mode:
            self.video_frame.grid_remove()
        self.width_var.trace_add("write", lambda *_: self._on_dimensions_manual_change())
        self.height_var.trace_add("write", lambda *_: self._on_dimensions_manual_change())
        self.model_var.trace_add("write", lambda *_: self._update_model_info())
        # Typing a model id (not just picking one) must also switch modes.
        self.model_var.trace_add("write", lambda *_: self._update_mode_controls())
        self.lora_source_var.trace_add("write", self._on_lora_fields_changed)
        self.lora_weight_var.trace_add("write", self._on_lora_fields_changed)
        self.lora_scale_var.trace_add("write", self._on_lora_fields_changed)

    def _apply_theme(self, dark_mode: bool) -> None:
        if dark_mode:
            bg = "#1b1c24"
            fg = "#f4f4f7"
            accent = "#7bc3ff"
        else:
            bg = "#f3f3f3"
            fg = "#101019"
            accent = "#3062c8"

        self.configure(bg=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TEntry", fieldbackground="#ffffff" if not dark_mode else "#272937", foreground=fg)
        self.style.configure("TCombobox", fieldbackground="#ffffff" if not dark_mode else "#272937", foreground=fg)
        self.style.configure("TButton", padding=5)
        self.style.configure("Horizontal.TProgressbar", background=accent)
        self.image_label.configure(background=bg, foreground=fg)

    # ------------------------------------------------------------------
    def _on_style_preset_changed(self, _event=None) -> None:
        preset = self.style_var.get()
        prompts = STYLE_PRESETS.get(preset)
        if prompts:
            pos, neg = prompts
            self.prompt_entry.delete(0, tk.END)
            self.prompt_entry.insert(0, pos)
            self.negative_prompt_entry.delete(0, tk.END)
            self.negative_prompt_entry.insert(0, neg)
        self._save_settings()

    def _on_model_changed(self, _event=None) -> None:
        self._set_status("Model changed – will load on next run…")
        self._update_mode_controls()
        self._save_settings()

    def _on_random_seed(self) -> None:
        self.seed_var.set(str(random.randint(0, 2**31 - 1)))
        self._save_settings()

    def _on_performance_preset(self) -> None:
        self.width_var.set(768)
        self.height_var.set(768)
        self._set_size_preset_by_dimensions(768, 768)
        self.steps_var.set(20)
        self.guidance_var.set(4.0)
        self._save_settings()

    def _update_model_info(self) -> None:
        """Refresh the tag + usage-hint label under the model picker."""
        label = getattr(self, "model_info_label", None)
        if label is None:
            return
        info = catalog_get(self.model_var.get())
        text = f"{info.tag} — {info.blurb}" if info.tag else info.blurb
        label.config(text=text)

    def _on_apply_recommended(self) -> None:
        """Apply the selected model's recommended steps / CFG / sampler."""
        info = catalog_get(self.model_var.get().strip() or DEFAULT_MODEL_IDS[0])
        rec = info.recommended
        self.steps_var.set(rec.steps)
        self.guidance_var.set(rec.cfg_scale)
        if rec.sampler in SAMPLERS:
            self.sampler_var.set(rec.sampler)
        self._save_settings()
        self._set_status(f"Recommended settings for {info.label} — {rec.note}")

    def _on_theme_toggle(self) -> None:
        self._apply_theme(self.dark_mode_var.get())
        self._save_settings()

    # ------------------------------------------------------------------
    def _on_estimate_vram(self) -> None:
        width, height = self._clamped_dimensions()
        steps = max(10, min(100, int(self.steps_var.get() or 30)))
        model_id = self.model_var.get().strip() or DEFAULT_MODEL_IDS[0]

        base_pixels = 1024 * 1024
        scale_pixels = (width * height) / base_pixels
        base_gb = catalog_get(model_id).vram_base_gb
        est_gb = base_gb * scale_pixels * (steps / 30) ** 0.5
        est_gb = max(3.0, min(20.0, est_gb))

        message = (
            f"Approximate VRAM needed:\n"
            f"Model: {model_id}\n"
            f"Resolution: {width}x{height}\n"
            f"Steps: {steps}\n\n"
            f"Estimated VRAM: ~{est_gb:.1f} GB (heuristic)"
        )
        self._set_status(f"Estimated VRAM ~{est_gb:.1f} GB")
        messagebox.showinfo("Estimated VRAM", message)

    # ------------------------------------------------------------------
    def on_generate_clicked(self) -> None:
        if self._is_generating:
            return
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Missing Prompt", "Please enter a prompt.")
            return

        width, height = self._clamped_dimensions()
        steps = max(10, min(100, int(self.steps_var.get() or 30)))
        seed = self._parse_seed()
        if seed is None:
            # Materialize a random seed so it is recoverable from saved metadata.
            seed = random.randint(0, 2**31 - 1)
            self.seed_var.set(str(seed))
        model_id = self.model_var.get().strip() or DEFAULT_MODEL_IDS[0]
        guidance_scale = max(1.0, min(12.0, float(self.guidance_var.get() or 4.5)))

        self.model_ids = self.settings_store.ensure_model_id(self.model_ids, model_id)
        # Keep the dropdown filtered to the current output mode — assigning the
        # full list here would quietly undo the Image/Video narrowing.
        self._sync_output_mode_to_model()

        self._is_generating = True
        self._toggle_generation_controls(disabled=True)

        if is_video_model(model_id):
            self._save_settings()
            self._start_video_generation(
                prompt=prompt,
                model_id=model_id,
                steps=steps,
                seed=seed,
                guidance_scale=guidance_scale,
            )
            return

        self._set_status("Preparing model…")
        self.progress_bar.config(maximum=steps)
        self.progress_var.set(0)
        self._progress_total = steps
        self._save_settings()

        negative_prompt = self.negative_prompt_entry.get().strip()
        if self.quality_booster_var.get():
            prompt = f"{prompt}, {QUALITY_BOOSTER_POS}"
            negative_prompt = (
                f"{negative_prompt}, {QUALITY_BOOSTER_NEG}" if negative_prompt else QUALITY_BOOSTER_NEG
            )
        lora_config = self._current_lora_config()
        reference_images = [p for p in self.reference_paths if p]
        ref_strength = max(0.1, min(1.0, float(self.ref_strength_var.get() or 0.7)))
        ref_mode = REF_MODE_LABELS.get(self.ref_mode_var.get(), REF_MODE_IMG2IMG)

        if self.lora_enabled_var.get() and lora_config is None:
            messagebox.showwarning("LoRA Missing", "Enable LoRA requires a source path or Hugging Face repo ID.")
            self._toggle_generation_controls(disabled=False)
            self._is_generating = False
            return

        active_lora_preset = self.lora_preset_var.get()
        if (
            self.lora_enabled_var.get()
            and active_lora_preset != LORA_PRESET_CUSTOM
            and active_lora_preset in self._lora_preset_errors
        ):
            self._toggle_generation_controls(disabled=False)
            self._is_generating = False
            messagebox.showerror(
                "LoRA Unavailable",
                (
                    f"The preset '{active_lora_preset}' couldn't be downloaded automatically.\n"
                    "Pick another preset, set a custom local LoRA path, or obtain access to the original repo."
                ),
            )
            return

        sampler = self._current_sampler_label()

        def worker() -> None:
            try:
                device_info = self.pipeline_manager.ensure_pipeline(model_id, lora_config)
                self.pipeline_manager.set_sampler(sampler)
                self.after(0, lambda: self._set_status(f"Model ready on {device_info.description} – generating…"))
                hires_fix = bool(self.hires_fix_var.get())
                hires_scale = float(self.hires_scale_var.get() or 1.5)
                hires_strength = float(self.hires_strength_var.get() or 0.35)
                image = self.pipeline_manager.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    strength=ref_strength,
                    ref_mode=ref_mode,
                    progress_callback=self._progress_callback,
                    reference_images=reference_images,
                    hires_fix=hires_fix,
                    hires_scale=hires_scale,
                    hires_strength=hires_strength,
                )
                metadata = self._build_generation_metadata(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    model_id=model_id,
                    sampler=self._current_sampler_label(),
                    hires_fix=hires_fix,
                    hires_scale=hires_scale,
                    hires_strength=hires_strength,
                )
                self._last_generation_metadata = metadata
                self.after(0, lambda img=image: self._finish_generation(img))
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Generation failed: %s", exc)
                self.after(0, lambda e=exc: self._fail_generation(e))
            finally:
                self._is_generating = False

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    def on_save_clicked(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("No Image", "Generate an image first.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All Files", "*.*")],
        )
        if not file_path:
            return
        try:
            self._save_image_with_metadata(self.current_image, file_path)
            self._set_status(f"Saved to {file_path}")
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Save failed: %s", exc)
            messagebox.showerror("Save Error", str(exc))

    def on_upscale_clicked(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("No Image", "Generate an image first.")
            return
        img = self.current_image
        w, h = img.size
        new_size = (w * 2, h * 2)
        max_dim = 4096
        if new_size[0] > max_dim or new_size[1] > max_dim:
            scale = min(max_dim / new_size[0], max_dim / new_size[1])
            new_size = (int(new_size[0] * scale), int(new_size[1] * scale))
        upscaled = img.resize(new_size, Image.LANCZOS)
        self.current_image = upscaled
        self._auto_cache_image(upscaled)
        self._update_image_preview()
        self._set_status(f"Upscaled to {new_size[0]}x{new_size[1]}")

    def on_ai_upscale_clicked(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("No Image", "Generate an image first.")
            return
        if self._is_ai_upscaling:
            return

        base_image = self.current_image
        prompt = self.prompt_entry.get().strip() or "highly detailed image"

        self._is_ai_upscaling = True
        self._toggle_generation_controls(disabled=True)
        self._set_status("AI upscaling image…")

        def worker() -> None:
            try:
                image = self.pipeline_manager.upscale_image(base_image, prompt)
                self.after(0, lambda img=image: self._finish_upscale(img))
            except Exception as exc:  # pragma: no cover
                self.logger.exception("AI upscaling failed: %s", exc)
                self.after(0, lambda e=exc: self._fail_upscale(e))
            finally:
                self._is_ai_upscaling = False

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    def _progress_callback(self, step: int, _timestep: int, _latents: object) -> None:
        # Runs on the diffusers worker thread — don't touch Tk vars here.
        total = max(1, self._progress_total)
        current = max(0, min(total, step + 1))
        self.after(0, lambda c=current, t=total: self._update_progress(c, t))

    def _update_progress(self, current: int, total: int) -> None:
        self.progress_bar.config(maximum=total)
        self.progress_var.set(current)
        if current >= total:
            self._set_status("Finishing image…")

    def _finish_generation(self, image: Image.Image) -> None:
        self.current_image = image
        self._auto_cache_image(image)
        self._update_image_preview()
        self.progress_var.set(self.progress_bar["maximum"])
        self._set_status("Generation complete")
        self._toggle_generation_controls(disabled=False)

    def _fail_generation(self, exc: Exception) -> None:
        messagebox.showerror("Generation Error", str(exc))
        self._set_status("Generation failed")
        self._toggle_generation_controls(disabled=False)

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    def _models_of_kind(self, video: bool) -> list[str]:
        return [m for m in self.model_ids if is_video_model(m) == video]

    def _on_output_mode_selected(self) -> None:
        """User picked Image or Video. Narrow the model list to that kind and,
        if the current model is the wrong kind, move to one that isn't —
        otherwise the radio would claim a mode the loaded model can't do."""
        if self._syncing_output_mode:
            return
        want_video = self.output_mode_var.get() == "Video"
        candidates = self._models_of_kind(want_video)
        if not candidates:
            messagebox.showinfo(
                "No models",
                "No video models are configured."
                if want_video else "No image models are configured.",
            )
            self._sync_output_mode_to_model()
            return
        self.model_combobox["values"] = tuple(candidates)
        if is_video_model(self.model_var.get().strip()) != want_video:
            self.model_var.set(candidates[0])
        self._update_mode_controls()
        self._save_settings()

    def _sync_output_mode_to_model(self) -> None:
        """Keep the radio honest when the model is changed directly (typed or
        picked), without re-entering _on_output_mode_selected."""
        self._syncing_output_mode = True
        try:
            video = is_video_model(self.model_var.get().strip() or DEFAULT_MODEL_IDS[0])
            self.output_mode_var.set("Video" if video else "Image")
            self.model_combobox["values"] = tuple(self._models_of_kind(video))
        finally:
            self._syncing_output_mode = False

    def _update_mode_controls(self, *, initial: bool = False) -> None:
        """Show the controls that belong to the selected model's family.

        Video and still models share the prompt, size, steps, CFG and seed
        widgets; everything else differs. Rather than leaving image-only
        controls live and silently ignoring them at generate time (the failure
        mode where the UI implies something it won't do), the ones that don't
        apply are removed from the layout.

        `initial=True` means "the app booted in this mode", which is NOT a
        transition: there is no outgoing size to remember, and treating it as
        one captured the *video* dimensions as the remembered image size and
        made the image mode restore 704x480 forever after.
        """
        video = is_video_model(self.model_var.get().strip() or DEFAULT_MODEL_IDS[0])
        self._sync_output_mode_to_model()
        if video == self._is_video_mode and not initial:
            return
        was_video = self._is_video_mode
        self._is_video_mode = video
        self.generate_button.config(text="Generate Video" if video else "Generate")

        if not initial:
            # Remember the size the mode we are leaving was using.
            outgoing = (self.width_var.get(), self.height_var.get())
            if was_video:
                self._video_size = outgoing
            else:
                self._image_size = outgoing

        if video:
            self.hires_frame.grid_remove()
            self.video_frame.grid()
            if not initial:
                self.width_var.set(self._video_size[0])
                self.height_var.set(self._video_size[1])
            # LoRA, reference images and the sampler picker are SDXL-only.
            for widget in (self.lora_frame, self.ref_frame):
                widget.grid_remove()
            self.sampler_combobox.config(state=tk.DISABLED)
            self._on_video_length_changed()
        else:
            self.video_frame.grid_remove()
            self.hires_frame.grid()
            if not initial:
                self.width_var.set(self._image_size[0])
                self.height_var.set(self._image_size[1])
            for widget in (self.lora_frame, self.ref_frame):
                widget.grid()
            self.sampler_combobox.config(state="readonly")
        self._update_model_info()

    def _on_video_length_changed(self) -> None:
        """Show the frame count the model will actually use — it snaps to a
        legal value (8k+1 for LTX), so the seconds spinner alone would imply a
        precision the model doesn't have."""
        label = getattr(self, "video_frames_label", None)
        if label is None:
            return
        info = catalog_get(self.model_var.get().strip() or DEFAULT_MODEL_IDS[0])
        spec = info.video
        if spec is None:
            return
        try:
            seconds = float(self.video_seconds_var.get() or 4.0)
        except Exception:
            seconds = 4.0
        frames = _round_frame_count(spec, max(1, round(seconds * spec.fps)))
        label.config(
            text=f"= {frames} frames @ {spec.fps} fps ({frames / spec.fps:.2f}s actual)"
        )

    def _start_video_generation(
        self, *, prompt: str, model_id: str, steps: int, seed: int, guidance_scale: float
    ) -> None:
        width, height = self._clamped_dimensions()
        negative_prompt = self.negative_prompt_entry.get().strip()
        # Deliberately NOT applying the quality booster: its tags are compel
        # weighted-prompt syntax ("(masterpiece:1.2)") aimed at SDXL's CLIP
        # encoders. LTX conditions on T5, which reads that as literal text.
        try:
            seconds = float(self.video_seconds_var.get() or 4.0)
        except Exception:
            seconds = 4.0
        codec = self.video_codec_var.get() or "auto"
        quality = int(self.video_quality_var.get() or DEFAULT_QUALITY)
        spec = catalog_get(model_id).video
        num_frames = _round_frame_count(spec, max(1, round(seconds * spec.fps)))

        self._stop_playback()
        self._set_status(f"Preparing {catalog_get(model_id).label}…")
        self.progress_bar.config(maximum=steps)
        self.progress_var.set(0)
        self._progress_total = steps

        def worker() -> None:
            try:
                device_info = self.pipeline_manager.ensure_pipeline(model_id, None)
                self.after(0, lambda: self._set_status(
                    f"Model ready on {device_info.description} — generating "
                    f"{num_frames} frames…"
                ))
                clip = self.pipeline_manager.generate_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    progress_callback=self._progress_callback,
                )
                self._last_generation_metadata = self._build_generation_metadata(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=clip.width,
                    height=clip.height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    model_id=model_id,
                    sampler="(flow matching)",
                    hires_fix=False,
                    hires_scale=1.0,
                    hires_strength=0.0,
                )
                self.after(0, lambda c=clip, cd=codec, q=quality: self._finish_video_generation(c, cd, q))
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Video generation failed: %s", exc)
                self.after(0, lambda e=exc: self._fail_generation(e))
            finally:
                self._is_generating = False

        threading.Thread(target=worker, daemon=True).start()

    def _finish_video_generation(self, clip, codec: str, quality: int) -> None:
        self.current_clip = clip
        self._video_codec_used = codec
        self._video_quality_used = quality
        # Frame 0 doubles as `current_image` so Save Image / upscale keep
        # working on a still pulled out of the clip.
        self.current_image = clip.frames[0]
        self._prepare_clip_photos(clip)
        self.progress_var.set(self.progress_bar["maximum"])
        self._set_status(
            f"Generated {len(clip.frames)} frames — {clip.duration:.2f}s @ {clip.fps} fps, "
            f"{clip.width}x{clip.height}. Save Video to encode."
        )
        self._toggle_generation_controls(disabled=False)
        self._show_frame(0)
        self.on_play_pause()  # autoplay: the point of a clip is the motion

    def _prepare_clip_photos(self, clip) -> None:
        """Pre-scale every frame to a PhotoImage once. Doing this per tick
        cannot keep up with 24 fps in Tk, and the references must be held or
        Tk collects them and the label goes blank."""
        max_w, max_h = 768, 512
        w, h = clip.frames[0].size
        scale = min(max_w / w, max_h / h, 1.0)
        size = (int(w * scale), int(h * scale))
        self._clip_photos = [
            ImageTk.PhotoImage(f.resize(size, Image.LANCZOS)) for f in clip.frames
        ]
        self.frame_scale.config(to=max(0, len(self._clip_photos) - 1))
        self.playback_frame.pack(fill=tk.X, pady=(8, 0))

    def _show_frame(self, index: int) -> None:
        if not self._clip_photos:
            return
        index = max(0, min(len(self._clip_photos) - 1, int(index)))
        self._clip_index = index
        self.image_label.config(image=self._clip_photos[index], text="")
        clip = self.current_clip
        if clip is not None:
            self.frame_counter_label.config(
                text=f"frame {index + 1}/{len(self._clip_photos)}  "
                     f"({index / clip.fps:.2f}s)"
            )

    def _on_scrub(self, value: str) -> None:
        if not self._clip_photos:
            return
        index = int(float(value))
        if index != self._clip_index:
            self._stop_playback()
            self._show_frame(index)

    def on_play_pause(self) -> None:
        if not self._clip_photos:
            return
        if self._is_playing:
            self._stop_playback()
        else:
            self._is_playing = True
            self.play_button.config(text="Pause")
            self._tick()

    def _stop_playback(self) -> None:
        self._is_playing = False
        if self._play_job is not None:
            try:
                self.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        if hasattr(self, "play_button"):
            self.play_button.config(text="Play")

    def _tick(self) -> None:
        if not self._is_playing or not self._clip_photos:
            return
        index = (self._clip_index + 1) % len(self._clip_photos)
        self._show_frame(index)
        self.frame_scale.set(index)
        fps = self.current_clip.fps if self.current_clip else 24
        self._play_job = self.after(max(1, int(1000 / max(1, fps))), self._tick)

    def on_save_video_clicked(self) -> None:
        if self.current_clip is None:
            messagebox.showinfo("No Video", "Generate a video first.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("Matroska", "*.mkv"), ("All Files", "*.*")],
        )
        if not file_path:
            return
        codec = getattr(self, "_video_codec_used", "auto")
        quality = getattr(self, "_video_quality_used", DEFAULT_QUALITY)
        self._set_status("Encoding…")
        self.save_video_button.config(state=tk.DISABLED)

        def worker() -> None:
            try:
                encode_video(
                    self.current_clip.frames, file_path,
                    fps=self.current_clip.fps, codec=codec, quality=quality,
                )
                self.after(0, lambda: self._set_status(f"Saved to {file_path}"))
            except Exception as exc:
                self.logger.exception("Encode failed: %s", exc)
                self.after(0, lambda e=exc: messagebox.showerror("Encode Error", str(e)))
                self.after(0, lambda: self._set_status("Encode failed"))
            finally:
                self.after(0, lambda: self.save_video_button.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_upscale(self, image: Image.Image) -> None:
        self.current_image = image
        self._auto_cache_image(image)
        self._update_image_preview()
        self._set_status("AI upscaling complete")
        self._toggle_generation_controls(disabled=False)

    def _fail_upscale(self, exc: Exception) -> None:
        messagebox.showerror("AI Upscale Error", str(exc))
        self._set_status("AI upscaling failed")
        self._toggle_generation_controls(disabled=False)

    def _toggle_generation_controls(self, *, disabled: bool) -> None:
        state = tk.DISABLED if disabled else tk.NORMAL
        self.generate_button.config(state=state)
        self.save_button.config(state=tk.NORMAL if (self.current_image and not disabled) else tk.DISABLED)
        self.upscale_button.config(state=tk.NORMAL if (self.current_image and not disabled) else tk.DISABLED)
        self.ai_upscale_button.config(state=tk.NORMAL if (self.current_image and not disabled) else tk.DISABLED)
        self.save_video_button.config(
            state=tk.NORMAL if (self.current_clip is not None and not disabled) else tk.DISABLED
        )
        if not disabled and not self._is_video_mode:
            self._update_lora_controls()

    # ------------------------------------------------------------------
    def _prune_image_cache(self, max_files: int = 500) -> None:
        """Cap the auto-cache: keep the newest `max_files` PNGs, delete the rest."""
        try:
            files = sorted(
                self.cache_root.glob("image_*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            stale = files[max_files:]
            for path in stale:
                try:
                    path.unlink()
                except OSError:
                    pass
            if stale:
                self.logger.info(
                    "Pruned %d old cached images (cap %d)", len(stale), max_files
                )
        except Exception as exc:
            self.logger.debug("Image cache prune failed: %s", exc)

    def _auto_cache_image(self, image: Image.Image) -> None:
        try:
            filename = f"image_{uuid.uuid4().hex}.png"
            self._save_image_with_metadata(image, str(self.cache_root / filename))
        except Exception as exc:
            self.logger.debug("Failed to auto-cache image: %s", exc)

    def _update_image_preview(self) -> None:
        # Showing a still supersedes any loaded clip — stop playback and take
        # the transport controls away, or Play would animate frames that no
        # longer match what is on screen.
        self._stop_playback()
        if self.current_clip is not None:
            self.current_clip = None
            self._clip_photos = []
            self.playback_frame.pack_forget()
        if self.current_image is None:
            self.image_label.config(text="No image", image="")
            return
        img = self.current_image
        max_w, max_h = 768, 512
        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        resized = img.resize(new_size, Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.image_label.config(image=self._tk_image, text="")
        self.save_button.config(state=tk.NORMAL)
        self.upscale_button.config(state=tk.NORMAL)
        self.ai_upscale_button.config(state=tk.NORMAL)

    def _clamped_dimensions(self) -> tuple[int, int]:
        try:
            width = int(self.width_var.get() or 1024)
            height = int(self.height_var.get() or 1024)
        except Exception:
            width = height = 1024
        width = max(256, min(1536, width))
        height = max(256, min(1536, height))
        width -= width % 8
        height -= height % 8
        return width, height

    def _parse_seed(self) -> Optional[int]:
        seed_text = self.seed_var.get().strip()
        if not seed_text:
            return None
        try:
            return int(seed_text)
        except Exception:
            return None

    def _set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def _current_sampler_label(self) -> str:
        """Human-readable label of the active sampler (set by the sampler picker)."""
        var = getattr(self, "sampler_var", None)
        if var is not None:
            try:
                v = var.get().strip()
                if v:
                    return v
            except Exception:
                pass
        return "DPM++ 2M Karras"

    def _build_generation_metadata(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        model_id: str,
        sampler: str,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_strength: float = 0.35,
    ) -> str:
        """A1111-compatible `parameters` string for PNG embed.
        Loadable by Automatic1111, ComfyUI metadata viewers, and re-readable
        by this app via `image.info['parameters']`."""
        parts: list[str] = [prompt or ""]
        if negative_prompt:
            parts.append(f"Negative prompt: {negative_prompt}")
        tail = (
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {guidance_scale}, "
            f"Seed: {seed}, Size: {width}x{height}, Model: {model_id}"
        )
        if hires_fix:
            tail += (
                f", Hires upscale: {hires_scale}, Hires steps: {steps}, "
                f"Denoising strength: {hires_strength}"
            )
        parts.append(tail)
        return "\n".join(parts)

    def _save_image_with_metadata(self, image: Image.Image, path: str) -> None:
        """Save image; embed A1111-format `parameters` PNG tEXt chunk if PNG."""
        suffix = Path(path).suffix.lower()
        metadata = self._last_generation_metadata
        if suffix in ("", ".png") and metadata:
            from PIL import PngImagePlugin
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", metadata)
            image.save(path, pnginfo=pnginfo)
        else:
            image.save(path)

    def _save_settings(self) -> None:
        settings = UserSettings(
            prompt=self.prompt_entry.get().strip(),
            negative_prompt=self.negative_prompt_entry.get().strip(),
            # In video mode the live width/height belong to the VIDEO; writing
            # them into width/height would destroy the remembered image size
            # (and vice versa), which is what made the app boot at 704x480.
            width=(self._image_size[0] if self._is_video_mode else int(self.width_var.get() or 1024)),
            height=(self._image_size[1] if self._is_video_mode else int(self.height_var.get() or 1024)),
            steps=int(self.steps_var.get() or 30),
            guidance_scale=float(self.guidance_var.get() or 4.5),
            style=self.style_var.get() or next(iter(STYLE_PRESETS)),
            seed=self.seed_var.get().strip(),
            model_id=self.model_var.get().strip() or DEFAULT_MODEL_IDS[0],
            dark_mode=self.dark_mode_var.get(),
            lora_enabled=self.lora_enabled_var.get(),
            lora_source=self.lora_source_var.get().strip(),
            lora_weight_name=self.lora_weight_var.get().strip(),
            lora_scale=float(self.lora_scale_var.get() or 1.0),
            size_preset=self.size_preset_var.get() or SIZE_PRESET_CUSTOM,
            lora_preset=self.lora_preset_var.get() or LORA_PRESET_CUSTOM,
            ref_strength=float(self.ref_strength_var.get() or 0.7),
            ref_mode=REF_MODE_LABELS.get(self.ref_mode_var.get(), REF_MODE_IMG2IMG),
            sampler=self._current_sampler_label(),
            quality_booster=bool(self.quality_booster_var.get()),
            hires_fix=bool(self.hires_fix_var.get()),
            hires_scale=float(self.hires_scale_var.get() or 1.5),
            hires_strength=float(self.hires_strength_var.get() or 0.35),
            reference_images=[p for p in self.reference_paths if p],
            video_seconds=float(self.video_seconds_var.get() or 4.0),
            video_codec=self.video_codec_var.get() or "auto",
            video_quality=int(self.video_quality_var.get() or DEFAULT_QUALITY),
            # Mirror image of the above: the video size only comes from the
            # live widgets while video mode is actually active.
            video_width=(
                int(self.width_var.get() or 704) if self._is_video_mode else self._video_size[0]
            ),
            video_height=(
                int(self.height_var.get() or 480) if self._is_video_mode else self._video_size[1]
            ),
        )
        self.settings_store.save_user_settings(settings)

    def _on_close(self) -> None:
        self._stop_playback()
        self._save_settings()
        self.destroy()

    def _on_lora_toggle(self) -> None:
        self._update_lora_controls()
        self._save_settings()

    def _update_lora_controls(self) -> None:
        enabled = self.lora_enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (self.lora_source_entry, self.lora_weight_entry, self.lora_scale_spin):
            widget.config(state=state)
        self._start_lora_prefetch()

    def _load_lora_presets(self) -> list[dict[str, object]]:
        presets = [dict(preset) for preset in DEFAULT_LORA_PRESETS]
        config_path = Path.cwd() / "config" / "lora_presets.json"
        if config_path.is_file():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        label = str(item.get("label", "")).strip()
                        source = str(item.get("source", "")).strip()
                        if not label or not source:
                            continue
                        weight = str(item.get("weight_name", "")).strip() or None
                        try:
                            scale = float(item.get("scale", 1.0))
                        except Exception:
                            scale = 1.0
                        presets.append(
                            {
                                "label": label,
                                "source": source,
                                "weight_name": weight,
                                "scale": scale,
                            }
                        )
            except Exception as exc:  # pragma: no cover - defensive path
                self.logger.warning("Failed to read LoRA presets %s: %s", config_path, exc)
        return presets

    def _start_lora_prefetch(self) -> None:
        def worker() -> None:
            for preset in self.lora_presets:
                source = str(preset.get("source", "")).strip()
                if not source:
                    continue
                label = preset.get("label") or source
                try:
                    self.pipeline_manager.ensure_lora_cached(source)
                    self.logger.info("LoRA preset %s ready", label)
                    self._set_lora_preset_error_async(label, None)
                except Exception as exc:
                    self.logger.warning("LoRA preset %s failed to cache: %s", label, exc)
                    self._set_lora_preset_error_async(label, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _current_lora_config(self) -> LoRAConfig | None:
        if not self.lora_enabled_var.get():
            return None
        source = self.lora_source_var.get().strip()
        if not source:
            return None
        try:
            scale = float(self.lora_scale_var.get())
        except Exception:
            scale = 1.0
        scale = max(0.1, min(2.0, scale))
        weight = self.lora_weight_var.get().strip() or None
        return LoRAConfig(source=source, weight_name=weight, scale=scale)

    def _on_size_preset_changed(self, _event=None) -> None:
        preset_name = self.size_preset_var.get()
        if preset_name == SIZE_PRESET_CUSTOM:
            return
        for label, (w, h) in SIZE_PRESETS:
            if label == preset_name:
                self._apply_size_preset(w, h)
                break
        self._save_settings()

    def _apply_size_preset(self, width: int, height: int) -> None:
        self._is_applying_size_preset = True
        try:
            self.width_var.set(width)
            self.height_var.set(height)
        finally:
            self._is_applying_size_preset = False

    def _on_dimensions_manual_change(self) -> None:
        if self._is_applying_size_preset:
            return
        width = self.width_var.get()
        height = self.height_var.get()
        if not self._set_size_preset_by_dimensions(width, height):
            self.size_preset_var.set(SIZE_PRESET_CUSTOM)

    def _set_size_preset_by_dimensions(self, width: int, height: int) -> bool:
        for label, (w, h) in SIZE_PRESETS:
            if width == w and height == h:
                if self.size_preset_var.get() != label:
                    self.size_preset_var.set(label)
                return True
        return False

    def _sync_size_preset_to_dimensions(self, width: int, height: int) -> None:
        if not self._set_size_preset_by_dimensions(width, height):
            self.size_preset_var.set(SIZE_PRESET_CUSTOM)

    def _on_lora_preset_selected(self, _event=None) -> None:
        preset_name = self.lora_preset_var.get()
        if preset_name == LORA_PRESET_CUSTOM:
            return
        preset = next((p for p in self.lora_presets if p["label"] == preset_name), None)
        if not preset:
            return
        self._is_applying_lora_preset = True
        try:
            self.lora_source_var.set(str(preset.get("source", "")))
            weight = preset.get("weight_name") or ""
            self.lora_weight_var.set(weight)
            scale = preset.get("scale", 1.0)
            try:
                self.lora_scale_var.set(float(scale))
            except Exception:
                self.lora_scale_var.set(1.0)
        finally:
            self._is_applying_lora_preset = False
        if not self.lora_enabled_var.get():
            self.lora_enabled_var.set(True)
            self._update_lora_controls()
        self._ensure_lora_ready_async(preset)
        self._save_settings()

        error_msg = self._lora_preset_errors.get(preset_name)
        if error_msg:
            short_msg = error_msg.splitlines()[0]
            messagebox.showwarning(
                "LoRA Preset Unavailable",
                (
                    f"Preset '{preset_name}' couldn't be auto-downloaded yet.\n"
                    f"Reason: {short_msg}\n\n"
                    "If you already have the weights, point the source to the local folder or choose Custom."
                ),
            )

    def _on_lora_fields_changed(self, *_args) -> None:
        if self._is_applying_lora_preset:
            return
        if not self._set_lora_preset_by_fields():
            self.lora_preset_var.set(LORA_PRESET_CUSTOM)

    def _set_lora_preset_by_fields(self) -> bool:
        source = self.lora_source_var.get().strip()
        weight = self.lora_weight_var.get().strip() or None
        try:
            scale = float(self.lora_scale_var.get())
        except Exception:
            scale = 0.0
        rounded = round(scale, 3)
        for preset in self.lora_presets:
            preset_weight = (preset.get("weight_name") or None)
            preset_scale = round(float(preset.get("scale", 1.0)), 3)
            if source == preset.get("source") and weight == preset_weight and rounded == preset_scale:
                if self.lora_preset_var.get() != preset["label"]:
                    self.lora_preset_var.set(preset["label"])
                return True
        return False

    def _sync_lora_preset_to_fields(self) -> None:
        if not self._set_lora_preset_by_fields():
            self.lora_preset_var.set(LORA_PRESET_CUSTOM)

    def _ensure_lora_ready_async(self, preset: dict[str, object]) -> None:
        source = str(preset.get("source", "")).strip()
        if not source:
            return
        label = preset.get("label") or source

        def worker() -> None:
            self.after(0, lambda: self._set_status(f"Preparing LoRA preset {label}…"))
            try:
                self.pipeline_manager.ensure_lora_cached(source)
            except Exception as exc:
                self.logger.warning("LoRA preset %s download failed: %s", label, exc)
                self._set_lora_preset_error_async(label, str(exc))
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "LoRA Download Failed",
                        f"Failed to prepare LoRA preset '{label}'.\n{exc}",
                    ),
                )
            else:
                self._set_lora_preset_error_async(label, None)
                self.after(0, lambda: self._set_status(f"LoRA preset {label} ready"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_lora_preset_error_async(self, label: str, error_message: str | None) -> None:
        def updater() -> None:
            if error_message:
                self._lora_preset_errors[label] = error_message
                short_msg = error_message.splitlines()[0]
                self._set_status(f"LoRA preset {label} unavailable: {short_msg}")
            else:
                if label in self._lora_preset_errors:
                    del self._lora_preset_errors[label]

        self.after(0, updater)

    # Reference images helpers
    def _choose_reference_image(self, index: int) -> None:
        path = filedialog.askopenfilename(title="Select reference image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp" )])
        if not path:
            return
        self.reference_paths[index] = path
        self._update_reference_previews()
        self._save_settings()

    def _remove_reference_image(self, index: int) -> None:
        self.reference_paths[index] = ""
        self._update_reference_previews()
        self._save_settings()

    def _update_reference_previews(self) -> None:
        from PIL import Image as PILImage

        for i, lbl in enumerate(getattr(self, "_ref_labels", [])):
            path = self.reference_paths[i]
            if path:
                try:
                    img = PILImage.open(path)
                    img.thumbnail((96, 96))
                    tkimg = ImageTk.PhotoImage(img)
                    self._ref_thumb_imgs[i] = tkimg
                    lbl.config(image=tkimg, text="")
                except Exception:
                    lbl.config(text="(invalid)", image="")
            else:
                lbl.config(text="(empty)", image="")


def launch_app() -> None:
    working_dir = Path.cwd()
    logs_dir = working_dir / "logs"
    logger = configure_logging(logs_dir / "picture_ai.log")
    settings_store = SettingsStore(
        settings_path=working_dir / "config" / "settings.json",
        models_path=working_dir / "models_cache" / "models.json",
        logger=logger,
    )
    pipeline_manager = PipelineManager(models_root=working_dir / "models_cache", logger=logger)
    logger.info("Starting PictureAI GUI")
    app = PictureAIApp(logger=logger, settings_store=settings_store, pipeline_manager=pipeline_manager)
    app.mainloop()
