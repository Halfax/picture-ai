"""Command-line entry points for picture-ai.

    python -m picture_ai.cli models --kind video
    python -m picture_ai.cli image "a lantern in fog" --out still.png
    python -m picture_ai.cli video "a lantern swinging in fog" --seconds 4
    python -m picture_ai.cli encode frames_dir out.mp4 --fps 24
    python -m picture_ai.cli serve --port 8770

Run it with the venv interpreter directly — the console-script wrappers in
this project's venv are stale (see CLAUDE.md):

    venv\\Scripts\\python.exe -m picture_ai.cli video "..." --seconds 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import api
from .video_export import DEFAULT_QUALITY, available_encoders, resolve_ffmpeg


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().replace(" ", "").split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"--size wants WxH, e.g. 704x480 (got {text!r})")


def _progress(label: str):
    state = {"t0": time.time(), "last": -1}

    def cb(step: int, total: int) -> None:
        if step == state["last"]:
            return
        state["last"] = step
        elapsed = time.time() - state["t0"]
        pct = (step / total * 100) if total else 0
        sys.stderr.write(f"\r  {label}: step {step}/{total} ({pct:3.0f}%) {elapsed:5.1f}s")
        sys.stderr.flush()

    return cb


def cmd_models(args) -> int:
    entries = api.list_models(args.kind)
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0
    for e in entries:
        flags = []
        if e["gated"]:
            flags.append("gated")
        if e["kind"] == "video":
            v = e["video"]
            flags.append(f"{v['fps']}fps, default {v['default_frames']}f")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{e['kind']:>5}  {e['id']}")
        print(f"         {e['label']} — {e['tag']}{suffix}")
    return 0


def cmd_image(args) -> int:
    w, h = args.size
    res = api.generate_image(
        args.prompt,
        model=args.model,
        negative_prompt=args.negative,
        width=w, height=h,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        progress=_progress("image") if not args.quiet else None,
    )
    sys.stderr.write("\n")
    out = Path(args.out) if args.out else api.DEFAULT_OUTPUT_DIR / f"image_{int(time.time())}.png"
    print(res.save(out))
    return 0


def cmd_video(args) -> int:
    w, h = args.size
    clip = api.generate_video(
        args.prompt,
        model=args.model,
        negative_prompt=args.negative,
        width=w, height=h,
        seconds=args.seconds,
        num_frames=args.frames,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        progress=_progress("video") if not args.quiet else None,
    )
    sys.stderr.write("\n")
    out = Path(args.out) if args.out else api.DEFAULT_OUTPUT_DIR / f"video_{int(time.time())}.mp4"
    path = clip.save(out, codec=args.codec, quality=args.quality)
    if args.frames_dir:
        clip.save_frames(args.frames_dir)
        print(f"frames: {args.frames_dir}")
    if args.thumbnail:
        clip.thumbnail().save(args.thumbnail)
        print(f"thumb:  {args.thumbnail}")
    print(f"{path}  ({len(clip.frames)} frames, {clip.duration:.1f}s @ {clip.fps}fps, "
          f"{clip.width}x{clip.height})")
    return 0


def cmd_encode(args) -> int:
    from PIL import Image

    src = Path(args.frames_dir)
    files = sorted(src.glob(args.glob))
    if not files:
        print(f"No frames matching {args.glob!r} in {src}", file=sys.stderr)
        return 1
    frames = [Image.open(f) for f in files]
    path = api.encode(frames, args.out, fps=args.fps, codec=args.codec, quality=args.quality)
    print(f"{path}  ({len(frames)} frames @ {args.fps}fps)")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(host=args.host, port=args.port)
    return 0


def cmd_doctor(args) -> int:
    """Report what the install can actually do — cheap to run, and it answers
    'why did my encode come out badly' without a round of guessing."""
    ffmpeg = resolve_ffmpeg()
    print(f"ffmpeg:   {ffmpeg or 'NOT FOUND — pip install imageio-ffmpeg'}")
    if ffmpeg:
        have = available_encoders(ffmpeg)
        for name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264"):
            print(f"  {'ok ' if name in have else '-- '} {name}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch:    {torch.__version__} | cuda={torch.version.cuda} | available={cuda}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"gpu:      {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    except Exception as exc:
        print(f"torch:    unavailable ({exc})")
    try:
        import diffusers

        print(f"diffusers:{diffusers.__version__}")
    except Exception as exc:
        print(f"diffusers: unavailable ({exc})")
    n_video = len(api.list_models("video"))
    print(f"models:   {len(api.list_models('image'))} image, {n_video} video")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="picture_ai.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="log at INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="list catalog models")
    m.add_argument("--kind", choices=("all", "image", "video"), default="all")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_models)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("prompt")
    common.add_argument("--model")
    common.add_argument("--negative", default="")
    common.add_argument("--steps", type=int)
    common.add_argument("--cfg", type=float, dest="cfg")
    common.add_argument("--seed", type=int)
    common.add_argument("--out")
    common.add_argument("--quiet", action="store_true")

    i = sub.add_parser("image", parents=[common], help="generate a still")
    i.add_argument("--size", type=_parse_size, default=(1024, 1024))
    i.set_defaults(func=cmd_image)

    v = sub.add_parser("video", parents=[common], help="generate a clip")
    v.add_argument("--size", type=_parse_size, default=(704, 480))
    g = v.add_mutually_exclusive_group()
    g.add_argument("--seconds", type=float, help="length; converted at the model's fps")
    g.add_argument("--frames", type=int, help="explicit frame count")
    v.add_argument("--codec", default="auto", help="auto|h264_nvenc|av1_nvenc|libx264|...")
    v.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="CRF/CQ, lower is better")
    v.add_argument("--frames-dir", help="also write the PNG frames here")
    v.add_argument("--thumbnail", help="also write frame 0 here")
    v.set_defaults(func=cmd_video)

    e = sub.add_parser("encode", help="encode an existing frame directory")
    e.add_argument("frames_dir")
    e.add_argument("out")
    e.add_argument("--fps", type=int, default=24)
    e.add_argument("--glob", default="*.png")
    e.add_argument("--codec", default="auto")
    e.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    e.set_defaults(func=cmd_encode)

    s = sub.add_parser("serve", help="run the local HTTP API")
    s.add_argument("--host", default="127.0.0.1",
                   help="bind address; anything other than loopback exposes the GPU "
                        "to the network (see DECISIONS §19)")
    s.add_argument("--port", type=int, default=8770)
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("doctor", help="report GPU / encoder / model availability")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
