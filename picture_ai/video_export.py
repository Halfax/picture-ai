"""Frame-sequence -> video file encoding for the video families.

Deliberately NOT `diffusers.utils.export_to_video`: that helper picks whatever
backend it can find, and with only `cv2` installed it silently falls back to the
OpenCV writer, which gives essentially no codec control and a mediocre mp4. We
pipe raw RGB frames into ffmpeg ourselves so CRF/CQ, pixel format, container and
hardware encoding are all explicit.

No system ffmpeg is required: `imageio-ffmpeg` bundles its own binary, and the
bundled 7.1 win-x86_64 build ships `h264_nvenc` / `hevc_nvenc` / `av1_nvenc`, so
the RTX 5080's encoder is available out of the box (Blackwell has AV1 encode).

`codec="auto"` prefers NVENC and falls back to libx264 if the binary or the
driver can't provide it, so this works on a machine with no NVIDIA GPU too.

🔴 **AV1 CAVEAT — measured 2026-07-27, don't re-derive.** `av1_nvenc` works and
produces genuinely valid files (verified by decoding a generated clip back to
its frames and eyeballing the content). But **the bundled ffmpeg cannot decode
what it just encoded**: this build ships no `libdav1d`, its native `av1`
decoder demands hardware acceleration, and the `libaom-av1` decoder rejects
NVENC's bitstream with "Bitstream not supported by this decoder". So:

  - AV1 output is real, but **unverifiable with the tools in this venv** —
    check it in the actual target player before depending on it.
  - This is why `_AUTO_PREFERENCE` stops at h264: "auto" must produce something
    every consumer can read.
  - It is also concrete evidence for the HalfaxForge cutscene-codec decision in
    `projects/VIDEO-IDEAS.md` §3: if the engine side goes AV1, **dav1d is the
    requirement, not an implementation detail** — libaom is not a substitute.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Quality defaults. For x264 this is -crf; for NVENC it is -cq. Both are
# "lower is better", both are roughly comparable in the 18-28 band, and 18 is
# about where generated video stops visibly losing detail to the encoder.
DEFAULT_QUALITY = 18
DEFAULT_FPS = 24

# Preference order for codec="auto". NVENC first (near-free on the 5080), then
# the software encoder that is always present.
_AUTO_PREFERENCE = ("h264_nvenc", "libx264")

# Per-codec rate-control flags. NVENC needs `-b:v 0` or -cq is ignored and it
# silently falls back to a default bitrate.
_CODEC_ARGS: dict[str, list[str]] = {
    "libx264": ["-preset", "slow", "-crf", "{q}"],
    "libx265": ["-preset", "slow", "-crf", "{q}"],
    "h264_nvenc": ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "hevc_nvenc": ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "av1_nvenc": ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "libvpx-vp9": ["-b:v", "0", "-crf", "{q}"],
    "libaom-av1": ["-cpu-used", "4", "-crf", "{q}", "-b:v", "0"],
}

_encoder_cache: Optional[set[str]] = None


def resolve_ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary: the one imageio-ffmpeg bundles, else whatever
    is on PATH. Returns None if neither is available."""
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def available_encoders(ffmpeg: Optional[str] = None) -> set[str]:
    """Video encoder names the ffmpeg binary was built with (cached)."""
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache
    exe = ffmpeg or resolve_ffmpeg()
    if not exe:
        _encoder_cache = set()
        return _encoder_cache
    names: set[str] = set()
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            # Encoder lines look like: " V....D libx264   H.264 ..."
            if len(parts) >= 2 and parts[0].startswith("V"):
                names.add(parts[1])
    except Exception as exc:  # pragma: no cover - depends on the local binary
        logger.warning("Could not enumerate ffmpeg encoders: %s", exc)
    _encoder_cache = names
    return names


def pick_codec(codec: str = "auto", ffmpeg: Optional[str] = None) -> str:
    """Resolve `codec`, honouring an explicit request but falling back when the
    binary doesn't have it. Being built with NVENC is not proof the *driver*
    will accept it, so a caller using "auto" also gets a runtime retry in
    `encode_video`; an explicit codec is not silently substituted here."""
    have = available_encoders(ffmpeg)
    if codec and codec != "auto":
        if have and codec not in have:
            logger.warning(
                "Requested codec %s is not in this ffmpeg build; falling back to libx264",
                codec,
            )
            return "libx264"
        return codec
    for name in _AUTO_PREFERENCE:
        if not have or name in have:
            return name
    return "libx264"


def _to_rgb_array(frame) -> np.ndarray:
    """Normalise one frame to a uint8 HxWx3 RGB array."""
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        # diffusers can hand back float frames in [0, 1].
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def encode_video(
    frames: Sequence | Iterable,
    out_path: str | Path,
    *,
    fps: int = DEFAULT_FPS,
    codec: str = "auto",
    quality: int = DEFAULT_QUALITY,
    ffmpeg: Optional[str] = None,
    faststart: bool = True,
) -> Path:
    """Encode `frames` (PIL Images or HxWx3 arrays) to a video file.

    Returns the written path. Raises RuntimeError if no ffmpeg is available or
    the encode fails.
    """
    frames = list(frames)
    if not frames:
        raise RuntimeError("No frames to encode")

    exe = ffmpeg or resolve_ffmpeg()
    if not exe:
        raise RuntimeError(
            "No ffmpeg binary found. Install the bundled one with: "
            "venv\\Scripts\\python.exe -m pip install imageio-ffmpeg"
        )

    first = _to_rgb_array(frames[0])
    height, width = first.shape[:2]
    # yuv420p subsamples 2x2, so both dimensions must be even. The video models
    # already produce multiples of 32, but a caller could pass anything.
    crop_w, crop_h = width - (width % 2), height - (height % 2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chosen = pick_codec(codec, exe)
    attempts = [chosen]
    if codec == "auto" and chosen != "libx264":
        # Built-with-NVENC != driver-will-run-NVENC (in use, unsupported on this
        # card, or a headless/remote session). Keep a software fallback ready.
        attempts.append("libx264")

    last_error = ""
    for attempt, name in enumerate(attempts):
        args = [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an",
        ]
        if (crop_w, crop_h) != (width, height):
            args += ["-vf", f"crop={crop_w}:{crop_h}:0:0"]
        args += ["-c:v", name]
        args += [a.format(q=quality) for a in _CODEC_ARGS.get(name, ["-crf", "{q}"])]
        args += ["-pix_fmt", "yuv420p"]
        if faststart and out_path.suffix.lower() in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        args.append(str(out_path))

        logger.info(
            "Encoding %d frames -> %s | %dx%d @ %d fps | codec=%s q=%d",
            len(frames), out_path.name, crop_w, crop_h, fps, name, quality,
        )
        if name.startswith("av1"):
            logger.warning(
                "AV1: the file will be valid, but this ffmpeg build cannot "
                "decode it back (no libdav1d) — verify it in your target "
                "player before relying on it. See the module docstring."
            )
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(_to_rgb_array(frame).tobytes())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            # ffmpeg died early (usually a rejected encoder); its stderr says why.
            pass
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        rc = proc.wait()
        if rc == 0:
            logger.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
            return out_path

        last_error = stderr.strip()[-2000:]
        if attempt + 1 < len(attempts):
            logger.warning(
                "%s encode failed (rc=%s), retrying with %s. ffmpeg said: %s",
                name, rc, attempts[attempt + 1], last_error.splitlines()[-1] if last_error else "?",
            )

    raise RuntimeError(f"ffmpeg encode failed:\n{last_error}")
