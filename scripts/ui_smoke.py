"""Headless-ish smoke test for the Tkinter UI.

Builds the real window, drives the image<->video mode switch, and exercises the
clip playback path with a synthetic clip so the transport controls are tested
without spending minutes on a real generation. Destroys the window at the end.

    venv\\Scripts\\python.exe scripts\\ui_smoke.py

It needs a desktop session (it really does create a window) but never blocks:
there is no mainloop, just explicit update() pumps.
"""

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from picture_ai.app import PictureAIApp  # noqa: E402
from picture_ai.logging_utils import configure_logging  # noqa: E402
from picture_ai.pipeline_manager import PipelineManager, VideoResult  # noqa: E402
from picture_ai.settings_store import SettingsStore  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    logger = configure_logging(ROOT / "logs" / "ui_smoke.log")
    # 🔴 A THROWAWAY settings file, NOT config/settings.json. The app saves on
    # nearly every interaction, so pointing this at the real one silently
    # rewrites the user's model, size and seed — which is exactly what happened
    # the first time this test ran, leaving the GUI booting into video mode at
    # 704x480. A test must not mutate the state it is testing against.
    tmp = Path(tempfile.mkdtemp(prefix="picture_ai_uismoke_"))
    store = SettingsStore(
        settings_path=tmp / "settings.json",
        models_path=tmp / "models.json",
        logger=logger,
    )
    pm = PipelineManager(models_root=ROOT / "models_cache", logger=logger)

    app = PictureAIApp(logger=logger, settings_store=store, pipeline_manager=pm)
    app.update()
    print("window built")

    # -- image mode is the default ------------------------------------
    app.model_var.set("RunDiffusion/Juggernaut-XL-v9")
    app._update_mode_controls()
    app.update()
    check("image mode: video panel hidden", not app.video_frame.winfo_ismapped())
    check("image mode: hires panel shown", app.hires_frame.winfo_ismapped())
    check("image mode: LoRA panel shown", app.lora_frame.winfo_ismapped())

    check("image mode: Output radio says Image", app.output_mode_var.get() == "Image",
          app.output_mode_var.get())
    check("image mode: dropdown lists only image models",
          all(not m.startswith("Lightricks/") for m in app.model_combobox["values"]),
          str(app.model_combobox["values"][-1]))

    # -- the Output radio is the discoverable route into video ---------
    app.output_mode_var.set("Video")
    app._on_output_mode_selected()
    app.update()
    check("Video radio selects a video model", app.model_var.get() == "Lightricks/LTX-Video",
          app.model_var.get())
    check("Video radio filters the dropdown", list(app.model_combobox["values"]) == ["Lightricks/LTX-Video"],
          str(list(app.model_combobox["values"])))
    check("Generate button relabelled", app.generate_button.cget("text") == "Generate Video",
          str(app.generate_button.cget("text")))

    # -- switch to a video model --------------------------------------
    app.model_var.set("Lightricks/LTX-Video")
    app._update_mode_controls()
    app.update()
    check("video mode: video panel shown", app.video_frame.winfo_ismapped())
    check("video mode: hires panel hidden", not app.hires_frame.winfo_ismapped())
    check("video mode: LoRA panel hidden", not app.lora_frame.winfo_ismapped())
    check("video mode: refs panel hidden", not app.ref_frame.winfo_ismapped())
    check("video mode: size defaulted", (app.width_var.get(), app.height_var.get()) == (704, 480),
          f"{app.width_var.get()}x{app.height_var.get()}")

    app.video_seconds_var.set(4.0)
    app._on_video_length_changed()
    app.update()
    text = app.video_frames_label.cget("text")
    check("frame-count hint computed", "97 frames" in text, text)

    # -- synthetic clip drives the playback transport ------------------
    frames = [Image.new("RGB", (128, 96), (i * 20 % 255, 60, 120)) for i in range(6)]
    clip = VideoResult(frames=frames, fps=24, width=128, height=96, seed=1,
                       model_id="Lightricks/LTX-Video")
    app._finish_video_generation(clip, "auto", 18)
    app.update()
    check("clip stored", app.current_clip is clip)
    check("photos prebuilt for every frame", len(app._clip_photos) == 6,
          f"{len(app._clip_photos)}")
    check("transport visible", app.playback_frame.winfo_ismapped())
    check("Save Video enabled", str(app.save_video_button.cget("state")) == "normal")
    check("autoplay started", app._is_playing)

    app.on_play_pause()
    app.update()
    check("pause stops playback", not app._is_playing)
    check("pause cancels the tick job", app._play_job is None)

    app._on_scrub("3")
    app.update()
    check("scrub moves to frame", app._clip_index == 3, f"index={app._clip_index}")
    check("counter text updated", "4/6" in app.frame_counter_label.cget("text"),
          app.frame_counter_label.cget("text"))

    # -- a still supersedes the clip ----------------------------------
    app.current_image = Image.new("RGB", (64, 64), (10, 10, 10))
    app._update_image_preview()
    app.update()
    check("still clears the clip", app.current_clip is None)
    check("transport hidden again", not app.playback_frame.winfo_ismapped())

    # -- back to image mode restores the panels ------------------------
    app.output_mode_var.set("Image")
    app._on_output_mode_selected()
    app.update()
    check("Image radio moves off the video model", not app.model_var.get().startswith("Lightricks/"),
          app.model_var.get())
    check("Generate button relabelled back", app.generate_button.cget("text") == "Generate",
          str(app.generate_button.cget("text")))
    app.model_var.set("RunDiffusion/Juggernaut-XL-v9")
    app._update_mode_controls()
    app.update()
    check("back to image: hires shown", app.hires_frame.winfo_ismapped())
    check("back to image: video hidden", not app.video_frame.winfo_ismapped())
    check("back to image: size restored", (app.width_var.get(), app.height_var.get()) != (704, 480),
          f"{app.width_var.get()}x{app.height_var.get()}")

    app._stop_playback()
    app.destroy()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("all UI checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
