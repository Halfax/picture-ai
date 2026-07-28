"""Local HTTP API for picture-ai.

    venv\\Scripts\\python.exe -m picture_ai.cli serve            # 127.0.0.1:8770

Generation takes minutes, so this is a JOB API, not a request/response one:
POST returns a job id immediately, the caller polls for progress, then fetches
the artifact. One GPU means one worker thread and a FIFO queue — parallel
denoise loops on 16 GB is an OOM, not a speedup.

Built on stdlib `http.server`. That is a deliberate choice, not an oversight:
this replaces a documented "no server" rule with the smallest thing that can
honour the new requirement, and adding FastAPI/uvicorn to a desktop app's
dependency set for a loopback job queue would not earn its keep.

🔴 TRUST BOUNDARY — read before changing `host`.
There is NO authentication. The default bind is loopback, where the trust
boundary is "processes on this machine". Binding to 0.0.0.0 puts unauthenticated
GPU-consuming, disk-writing endpoints on the LAN/mesh — that is DECISIONS §19
territory (network origin IS the boundary) and needs a firewall rule, not just a
flag. The flag exists because the fleet may legitimately want it; the warning
exists because the default must never drift.

Endpoints
    GET    /healthz              worker + queue health (see §17 note below)
    GET    /v1/models[?kind=]    catalog
    POST   /v1/images            {prompt, ...}        -> 202 {job_id}
    POST   /v1/videos            {prompt, seconds..}  -> 202 {job_id}
    GET    /v1/jobs              recent jobs
    GET    /v1/jobs/<id>         job state + progress
    GET    /v1/jobs/<id>/file    the artifact bytes
    DELETE /v1/jobs/<id>         cancel a job that has not started
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from . import api

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1 << 20     # 1 MiB — these are small JSON prompts
JOB_HISTORY = 200            # keep this many finished jobs in memory

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str                       # "image" | "video"
    params: dict
    state: str = STATE_QUEUED
    step: int = 0
    total_steps: int = 0
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    path: Optional[str] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        out = {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "step": self.step,
            "of": self.total_steps,
            "created": self.created,
            "prompt": self.params.get("prompt", ""),
        }
        if self.started:
            out["elapsed"] = round((self.finished or time.time()) - self.started, 2)
        if self.path:
            out["path"] = self.path
            out["file_url"] = f"/v1/jobs/{self.id}/file"
        if self.error:
            out["error"] = self.error
        if self.meta:
            out["meta"] = self.meta
        return out


class JobRunner:
    """FIFO queue + one worker thread. The worker owns the GPU."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or api.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._work, name="picture-ai-worker", daemon=True)
        self._started = time.time()
        self._last_finished: Optional[float] = None
        self._last_error: Optional[str] = None
        self._completed = 0
        self._failed = 0
        self._thread.start()

    # -- public -------------------------------------------------------
    def submit(self, kind: str, params: dict) -> Job:
        job = Job(id="j_" + uuid.uuid4().hex[:10], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._prune()
        self._q.put(job.id)
        logger.info("queued %s job %s", kind, job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i].public() for i in ids if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        """Only a job that hasn't started can be cancelled — a denoise loop in
        flight is not safely interruptible mid-step."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != STATE_QUEUED:
                return False
            job.state = STATE_CANCELLED
            job.finished = time.time()
            return True

    def health(self) -> dict:
        """🔴 DECISIONS §17: this service has a worker thread distinct from the
        HTTP handler, so 'the server answered' proves nothing about whether
        work is being done. Report work OUTPUT — is the worker thread alive,
        is the queue draining, when did a job last finish, what failed last."""
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.state == STATE_QUEUED)
            running = sum(1 for j in self._jobs.values() if j.state == STATE_RUNNING)
            oldest_queued = min(
                (j.created for j in self._jobs.values() if j.state == STATE_QUEUED),
                default=None,
            )
        alive = self._thread.is_alive()
        # A dead worker with a non-empty queue is the failure this exists to
        # catch: HTTP still answers 200 while nothing will ever be produced.
        stuck = (not alive) and (queued or running)
        return {
            "ok": bool(alive and not stuck),
            "service": "picture-ai",
            "worker_alive": alive,
            "queued": queued,
            "running": running,
            "completed": self._completed,
            "failed": self._failed,
            "last_finished": self._last_finished,
            "seconds_since_last_finish": (
                round(time.time() - self._last_finished, 1) if self._last_finished else None
            ),
            "oldest_queued_age": (
                round(time.time() - oldest_queued, 1) if oldest_queued else None
            ),
            "last_error": self._last_error,
            "uptime": round(time.time() - self._started, 1),
            "loaded_model": api.get_engine().loaded_model,
        }

    # -- internals ----------------------------------------------------
    def _prune(self) -> None:
        while len(self._order) > JOB_HISTORY:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def _work(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.get(job_id)
            if job is None or job.state == STATE_CANCELLED:
                continue
            job.state = STATE_RUNNING
            job.started = time.time()
            try:
                self._run(job)
                job.state = STATE_DONE
                self._completed += 1
            except Exception as exc:
                logger.exception("job %s failed", job.id)
                job.state = STATE_ERROR
                job.error = f"{type(exc).__name__}: {exc}"
                self._last_error = job.error
                self._failed += 1
                job.meta["traceback"] = traceback.format_exc()[-2000:]
            finally:
                job.finished = time.time()
                self._last_finished = job.finished

    def _run(self, job: Job) -> None:
        p = dict(job.params)
        prompt = p.pop("prompt")

        def progress(step: int, total: int) -> None:
            job.step, job.total_steps = step, total

        if job.kind == "image":
            codec_args = {k: p.pop(k) for k in ("codec", "quality") if k in p}
            del codec_args
            res = api.generate_image(prompt, progress=progress, **p)
            out = self.output_dir / f"{job.id}.png"
            res.save(out)
            job.path = str(out)
            job.meta.update(model=res.model_id, seed=res.seed)
        else:
            codec = p.pop("codec", "auto")
            quality = p.pop("quality", 18)
            clip = api.generate_video(prompt, progress=progress, **p)
            out = self.output_dir / f"{job.id}.mp4"
            clip.save(out, codec=codec, quality=quality)
            clip.thumbnail().save(self.output_dir / f"{job.id}_thumb.png")
            job.path = str(out)
            job.meta.update(
                model=clip.model_id, seed=clip.seed, fps=clip.fps,
                frames=len(clip.frames), duration=round(clip.duration, 2),
                width=clip.width, height=clip.height,
                thumbnail=str(self.output_dir / f"{job.id}_thumb.png"),
            )


# ----------------------------------------------------------------------
# Request parameter validation
# ----------------------------------------------------------------------
_IMAGE_FIELDS = {
    "model": str, "negative_prompt": str, "width": int, "height": int,
    "steps": int, "guidance_scale": float, "seed": int, "sampler": str,
}
_VIDEO_FIELDS = {
    "model": str, "negative_prompt": str, "width": int, "height": int,
    "seconds": float, "num_frames": int, "steps": int,
    "guidance_scale": float, "seed": int, "codec": str, "quality": int,
}


def _clean(body: dict, allowed: dict) -> dict:
    """Keep only known fields, coerced to the right type. An unknown key is a
    client bug — reject it loudly rather than silently ignoring a misspelled
    parameter and returning something that quietly isn't what was asked for."""
    if "prompt" not in body or not str(body["prompt"]).strip():
        raise ValueError("'prompt' is required")
    unknown = set(body) - set(allowed) - {"prompt"}
    if unknown:
        raise ValueError(
            f"unknown field(s): {', '.join(sorted(unknown))}. "
            f"Allowed: prompt, {', '.join(sorted(allowed))}"
        )
    out: dict[str, Any] = {"prompt": str(body["prompt"])}
    for key, caster in allowed.items():
        if key in body and body[key] is not None:
            try:
                out[key] = caster(body[key])
            except (TypeError, ValueError):
                raise ValueError(f"field {key!r} must be {caster.__name__}")
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "picture-ai/1.0"
    runner: JobRunner  # set on the server instance

    # -- helpers ------------------------------------------------------
    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- routes -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path, qs = parsed.path.rstrip("/") or "/", parse_qs(parsed.query)
        runner = self.server.runner  # type: ignore[attr-defined]

        if path in ("/healthz", "/"):
            health = runner.health()
            return self._json(200 if health["ok"] else 503, health)
        if path == "/v1/models":
            kind = (qs.get("kind") or ["all"])[0]
            if kind not in ("all", "image", "video"):
                return self._error(400, "kind must be all|image|video")
            return self._json(200, {"models": api.list_models(kind)})
        if path == "/v1/jobs":
            limit = int((qs.get("limit") or ["50"])[0])
            return self._json(200, {"jobs": runner.recent(min(limit, JOB_HISTORY))})
        if path.startswith("/v1/jobs/"):
            rest = path[len("/v1/jobs/"):]
            if rest.endswith("/file"):
                return self._send_file(runner, rest[:-len("/file")])
            job = runner.get(rest)
            if job is None:
                return self._error(404, f"no such job: {rest}")
            return self._json(200, job.public())
        return self._error(404, f"no such endpoint: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        runner = self.server.runner  # type: ignore[attr-defined]
        try:
            body = self._body()
            if path == "/v1/images":
                params = _clean(body, _IMAGE_FIELDS)
                job = runner.submit("image", params)
            elif path == "/v1/videos":
                params = _clean(body, _VIDEO_FIELDS)
                job = runner.submit("video", params)
            else:
                return self._error(404, f"no such endpoint: {path}")
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._json(202, job.public())

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        runner = self.server.runner  # type: ignore[attr-defined]
        if not path.startswith("/v1/jobs/"):
            return self._error(404, f"no such endpoint: {path}")
        job_id = path[len("/v1/jobs/"):]
        if runner.get(job_id) is None:
            return self._error(404, f"no such job: {job_id}")
        if runner.cancel(job_id):
            return self._json(200, {"job_id": job_id, "state": STATE_CANCELLED})
        return self._error(
            409, "job already started — a denoise loop is not interruptible mid-step"
        )

    def _send_file(self, runner: JobRunner, job_id: str) -> None:
        job = runner.get(job_id)
        if job is None:
            return self._error(404, f"no such job: {job_id}")
        if not job.path:
            return self._error(409, f"job is {job.state}, no artifact yet")
        path = Path(job.path)
        if not path.exists():
            return self._error(410, "artifact no longer on disk")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8770, output_dir: Path | None = None) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.runner = JobRunner(output_dir)  # type: ignore[attr-defined]
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "BOUND TO %s — this API has NO authentication. Anyone who can reach "
            "this port can consume the GPU and write files. See DECISIONS §19.",
            host,
        )
    print(f"picture-ai API on http://{host}:{port}  (Ctrl-C to stop)")
    print(f"  health: http://{host}:{port}/healthz")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
