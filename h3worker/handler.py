"""Runpod Serverless entrypoint: MiniMax-H3 Ref2VA (+ optional LoRA) behind SGLang."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
from typing import Callable

from . import bootstrap
from .config import WorkerConfig, load_config
from .pipeline import GenerationError, GenerationTimeout, JobCancelled, UploadError, run_job
from .request import InputError, parse_job_input
from .sglang_server import SglangServer

PROGRESS_MIN_INTERVAL_S = 15.0


def prepare_weights(cfg: WorkerConfig) -> str:
    """Make sure the model (and LoRA) are on the volume; return the model path."""
    if cfg.auto_download:
        bootstrap.ensure_model(cfg.model_repo, cfg.model_dir, cfg.variant)
        lora = cfg.lora
        if lora is not None:
            bootstrap.ensure_lora(lora.repo, lora.filename, cfg.lora_dir)
    return cfg.model_dir


def allowed_host_suffixes() -> tuple[str, ...]:
    raw = os.environ.get("H3_ALLOWED_URL_SUFFIXES", "")
    return tuple(s.strip().lower() for s in raw.split(",") if s.strip())


class ProgressReporter:
    """Forward progress to Runpod when the stage changes or at most every few seconds.

    Every Runpod progress call opens its own thread and HTTP session, so polling-rate updates are wasteful.
    """

    def __init__(self, send: Callable[[dict], None], min_interval_s: float = PROGRESS_MIN_INTERVAL_S):
        self._send = send
        self._min_interval_s = min_interval_s
        self._last_stage: str | None = None
        self._last_at = 0.0

    def __call__(self, update: dict) -> None:
        now = time.monotonic()
        stage = update.get("stage")
        if stage == self._last_stage and now - self._last_at < self._min_interval_s:
            return
        self._last_stage, self._last_at = stage, now
        self._send(update)


class IdleWatchdog:
    """Exit the worker when it has been ready and idle too long, so the GPUs are released.

    Runpod bills an idle worker until it scales down; if its autoscaler leaves idle workers running,
    this bounds the cost of each one. Exiting while the model is still loading or a job is active never happens.
    """

    def __init__(
        self,
        idle_exit_s: float,
        is_ready: Callable[[], bool],
        exit_fn: Callable[[int], None] = os._exit,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.idle_exit_s = idle_exit_s
        self._is_ready = is_ready
        self._exit = exit_fn
        self._clock = clock
        self._active = 0
        self._last = clock()
        self._lock = threading.Lock()

    def job_started(self) -> None:
        with self._lock:
            self._active += 1

    def job_finished(self) -> None:
        with self._lock:
            self._active -= 1
            self._last = self._clock()

    def check(self) -> bool:
        """Exit (and return True) if the worker is ready, has no job, and has been idle past the limit."""
        with self._lock:
            if self._active or not self._is_ready():
                self._last = self._clock()  # the idle period starts once the worker is ready and free
                return False
            idle = self._clock() - self._last
        if idle < self.idle_exit_s:
            return False
        print(f"[worker] idle for {idle:.0f}s; exiting to release the GPUs", flush=True)
        self._exit(0)
        return True

    def run_forever(self, interval_s: float = 10.0) -> None:
        while True:
            time.sleep(interval_s)
            self.check()


def make_handler(
    cfg: WorkerConfig,
    server: SglangServer,
    send_progress: Callable[[dict, dict], None],
    exit_fn: Callable[[int], None] = os._exit,
    watchdog: IdleWatchdog | None = None,
) -> Callable[[dict], "asyncio.Future[dict]"]:
    suffixes = allowed_host_suffixes()

    async def handler(job: dict) -> dict:
        try:
            spec = parse_job_input(job.get("input"), suffixes)
        except InputError as exc:
            return {"error": f"Invalid input: {exc}"}

        if watchdog is not None:
            watchdog.job_started()
        try:
            return await _process(job, spec, suffixes)
        finally:
            if watchdog is not None:
                watchdog.job_finished()

    async def _process(job: dict, spec, suffixes) -> dict:
        cancel = threading.Event()
        progress = ProgressReporter(lambda update: send_progress(job, update))

        def work() -> dict:
            try:
                server.wait_ready(
                    cfg.ready_timeout_s,
                    on_tick=lambda elapsed: progress({"stage": "loading_model", "elapsed_s": round(elapsed)}),
                    cancel=cancel,
                )
                return run_job(
                    spec,
                    base_url=cfg.base_url,
                    output_dir=cfg.output_dir,
                    progress=progress,
                    job_timeout_s=cfg.job_timeout_s,
                    cancel=cancel,
                )
            except JobCancelled:
                return {"error": "Cancelled"}
            except GenerationTimeout as exc:
                # SGLang keeps working on a timed-out generation, so recycle the worker.
                return {"error": str(exc), "refresh_worker": True}
            except (GenerationError, UploadError) as exc:
                return {"error": str(exc)}
            except Exception as exc:
                traceback.print_exc()
                return {"error": f"{type(exc).__name__}: {exc}", "refresh_worker": True}

        # Runpod runs the handler on its event loop; a worker thread keeps the loop free so a
        # cancel request can reach us. SGLang cannot abort a generation, so cancelling stops the
        # server and ends this process to release the GPUs; Runpod then starts a fresh worker.
        try:
            return await asyncio.to_thread(work)
        except asyncio.CancelledError:
            cancel.set()
            print("[worker] job cancelled; stopping SGLang and recycling the worker", flush=True)
            server.stop()
            exit_fn(0)
            raise

    return handler


def main() -> None:
    import runpod

    cfg = load_config()
    server = SglangServer(cfg, prepare=lambda: prepare_weights(cfg))
    server.start()
    watchdog = IdleWatchdog(cfg.idle_exit_s, lambda: server.ready)
    if cfg.idle_exit_s > 0:
        threading.Thread(target=watchdog.run_forever, name="idle-watchdog", daemon=True).start()
    runpod.serverless.start(
        {"handler": make_handler(cfg, server, runpod.serverless.progress_update, watchdog=watchdog)}
    )


if __name__ == "__main__":
    main()
