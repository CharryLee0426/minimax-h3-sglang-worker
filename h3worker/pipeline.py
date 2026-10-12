"""Run one generation job against the local SGLang server and upload the result."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

import requests

from .request import JobSpec, build_video_request, effective_prompt

Progress = Callable[[dict], None]

POLL_INTERVAL_S = 2.0
UPLOAD_ATTEMPTS = 4


class GenerationError(RuntimeError):
    """SGLang rejected or failed the request."""


class GenerationTimeout(GenerationError):
    """The generation outlived its budget; SGLang may still be busy with it."""


class JobCancelled(Exception):
    """The job was cancelled while the worker was processing it."""


class UploadError(RuntimeError):
    """The finished video could not be stored."""


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:500]}"
    detail = body.get("detail") if isinstance(body, dict) else None
    return f"HTTP {response.status_code}: {detail or body}"


def submit(base_url: str, spec: JobSpec, session: requests.Session) -> dict:
    """Create the SGLang job. SGLang downloads the reference images before answering, so allow time."""
    try:
        response = session.post(
            f"{base_url}/v1/videos", json=build_video_request(spec), timeout=(10, 300)
        )
    except requests.Timeout as exc:
        raise GenerationError(f"SGLang did not accept the request in time: {exc}") from exc
    if response.status_code >= 400:
        raise GenerationError(f"SGLang rejected the request ({_error_message(response)})")
    return response.json()


def wait_for_video(
    base_url: str,
    video_id: str,
    session: requests.Session,
    progress: Progress,
    timeout_s: float,
    poll_interval_s: float = POLL_INTERVAL_S,
    cancel: threading.Event | None = None,
) -> dict:
    started = time.time()
    while True:
        if cancel is not None and cancel.is_set():
            raise JobCancelled()
        response = session.get(f"{base_url}/v1/videos/{video_id}", timeout=30)
        if response.status_code >= 400:
            raise GenerationError(f"status check failed ({_error_message(response)})")
        job = response.json()
        status = job.get("status")
        if status == "completed":
            return job
        if status in ("failed", "deleted"):
            message = (job.get("error") or {}).get("message") or status
            raise GenerationError(f"generation failed: {message}")
        elapsed = time.time() - started
        if elapsed > timeout_s:
            raise GenerationTimeout(f"generation exceeded {timeout_s:.0f}s")
        progress({"stage": "generating", "elapsed_s": round(elapsed)})
        time.sleep(poll_interval_s)


def download_video(base_url: str, video_id: str, dest: str, session: requests.Session) -> int:
    with session.get(f"{base_url}/v1/videos/{video_id}/content", stream=True, timeout=(10, 600)) as r:
        if r.status_code >= 400:
            raise GenerationError(f"could not read the video ({_error_message(r)})")
        size = 0
        with open(dest, "wb") as handle:
            for chunk in r.iter_content(1 << 20):
                handle.write(chunk)
                size += len(chunk)
    if size < 1024:
        raise GenerationError(f"generated video is unexpectedly small ({size} bytes)")
    return size


def upload_video(path: str, put_url: str, headers: dict[str, str], session: requests.Session) -> None:
    size = os.path.getsize(path)
    request_headers = {"Content-Type": "video/mp4", **headers}
    last_error = "unknown error"
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            with open(path, "rb") as handle:
                response = session.put(put_url, data=handle, headers=request_headers, timeout=(10, 600))
            if response.status_code < 300:
                return
            last_error = _error_message(response)
            if response.status_code < 500 and response.status_code != 429:
                break
        except requests.RequestException as exc:
            last_error = str(exc)
        if attempt < UPLOAD_ATTEMPTS:
            time.sleep(2**attempt)
    raise UploadError(f"upload of {size} bytes failed: {last_error}")


def run_job(
    spec: JobSpec,
    *,
    base_url: str,
    output_dir: str,
    progress: Progress,
    job_timeout_s: float,
    session: requests.Session | None = None,
    poll_interval_s: float = POLL_INTERVAL_S,
    cancel: threading.Event | None = None,
) -> dict:
    session = session or requests.Session()
    started = time.time()
    progress({"stage": "submitting"})
    created = submit(base_url, spec, session)
    video_id = created["id"]
    local_path = os.path.join(output_dir, f"upload-{video_id}.mp4")
    meta: dict = {}
    size = 0
    generation_s = 0.0
    try:
        meta = wait_for_video(
            base_url, video_id, session, progress, job_timeout_s, poll_interval_s, cancel
        )
        generation_s = time.time() - started
        progress({"stage": "uploading"})
        size = download_video(base_url, video_id, local_path, session)
        upload_video(local_path, spec.output.put_url, spec.output.headers, session)
    finally:
        _cleanup(local_path, created, meta, base_url, video_id, session)
    return {
        "video_key": spec.output.key,
        "size_bytes": size,
        "duration_seconds": spec.duration_seconds,
        "seed": spec.seed,
        "resolution": meta.get("size"),
        "generation_seconds": round(generation_s, 1),
        "total_seconds": round(time.time() - started, 1),
        "effective_prompt": effective_prompt(spec),
    }


def _cleanup(
    local_path: str, created: dict, meta: dict, base_url: str, video_id: str, session: requests.Session
) -> None:
    """Remove local files and the SGLang job record; never raises."""
    paths = [local_path]
    for source in (created, meta):
        paths.extend(source.get("file_paths") or [])
        if source.get("file_path"):
            paths.append(source["file_path"])
    for path in dict.fromkeys(paths):
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        session.delete(f"{base_url}/v1/videos/{video_id}", timeout=10)
    except requests.RequestException:
        pass
