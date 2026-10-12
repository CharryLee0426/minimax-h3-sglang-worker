"""Runpod Serverless entrypoint: MiniMax-H3 Ref2VA (+ optional LoRA) behind SGLang."""

from __future__ import annotations

import os
import traceback
from typing import Callable

from . import bootstrap
from .config import WorkerConfig, load_config
from .pipeline import GenerationError, UploadError, run_job
from .request import InputError, parse_job_input
from .sglang_server import SglangServer


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


def make_handler(
    cfg: WorkerConfig,
    server: SglangServer,
    send_progress: Callable[[dict, dict], None],
) -> Callable[[dict], dict]:
    suffixes = allowed_host_suffixes()

    def handler(job: dict) -> dict:
        def progress(update: dict) -> None:
            send_progress(job, update)

        try:
            spec = parse_job_input(job.get("input"), suffixes)
        except InputError as exc:
            return {"error": f"Invalid input: {exc}"}

        try:
            server.wait_ready(
                cfg.ready_timeout_s,
                on_tick=lambda elapsed: progress({"stage": "loading_model", "elapsed_s": round(elapsed)}),
            )
            return run_job(
                spec,
                base_url=cfg.base_url,
                output_dir=cfg.output_dir,
                progress=progress,
                job_timeout_s=cfg.job_timeout_s,
            )
        except (GenerationError, UploadError) as exc:
            # A timed-out generation keeps running inside SGLang, so recycle the worker.
            return {"error": str(exc), "refresh_worker": "exceeded" in str(exc)}
        except Exception as exc:
            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {exc}", "refresh_worker": True}

    return handler


def main() -> None:
    import runpod

    cfg = load_config()
    server = SglangServer(cfg, prepare=lambda: prepare_weights(cfg))
    server.start()
    runpod.serverless.start(
        {"handler": make_handler(cfg, server, runpod.serverless.progress_update)}
    )


if __name__ == "__main__":
    main()
