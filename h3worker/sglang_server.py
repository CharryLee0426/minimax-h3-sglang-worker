"""Launch and supervise the local SGLang diffusion server."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable

import requests

from .config import WorkerConfig, detect_gpu_count
from .lora_convert import ensure_native_lora
from .pipeline import JobCancelled


def build_command(
    cfg: WorkerConfig, num_gpus: int, model_path: str, lora_path: str | None = None
) -> list[str]:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--model-variant",
        cfg.variant,
        "--num-gpus",
        str(num_gpus),
        "--ulysses-degree",
        str(num_gpus),
        "--encoder-parallel",
        "auto",
        "--performance-mode",
        "speed",
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.port),
        "--output-path",
        cfg.output_dir,
    ]
    lora = cfg.lora
    if lora is not None:
        cmd += [
            "--lora-path",
            lora_path or lora.local_path,
            "--lora-nickname",
            "h3-custom",
            "--lora-scale",
            str(lora.scale),
            "--lora-merge-mode",
            lora.merge_mode,
        ]
        if lora.alpha is not None:
            cmd += ["--lora-alpha", str(lora.alpha)]
    cmd += list(cfg.extra_args)
    return cmd


LORA_APPLIED = re.compile(r"applied to (?P<n>\d+) layers")


class SglangServer:
    """Starts ``sglang serve`` in the background and reports readiness."""

    def __init__(self, cfg: WorkerConfig, prepare: Callable[[], str] | None = None):
        self.cfg = cfg
        self._prepare = prepare
        self._proc: subprocess.Popen | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, name="sglang-launcher", daemon=True)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            model_path = self._prepare() if self._prepare else self.cfg.model_dir
            num_gpus = detect_gpu_count(self.cfg.num_gpus)
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            lora = self.cfg.lora
            lora_path = ensure_native_lora(lora.local_path) if lora is not None else None
            cmd = build_command(self.cfg, num_gpus, model_path, lora_path)
            print("[worker] launching:", " ".join(cmd), flush=True)
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1
            )
            threading.Thread(
                target=self.pump_output, args=(self._proc.stdout,), name="sglang-output", daemon=True
            ).start()
            self._wait_healthy()
            self._ready.set()
            print("[worker] SGLang server is ready", flush=True)
        except Exception as exc:  # surfaced to jobs through wait_ready
            self._error = self._error or f"{type(exc).__name__}: {exc}"
            print(f"[worker] server startup failed: {self._error}", flush=True)

    def pump_output(self, lines) -> None:
        """Forward SGLang output to our log; a LoRA that matches no layer is a fatal misconfiguration."""
        for line in lines:
            print(line, end="", flush=True)
            if self.cfg.lora is not None and LORA_APPLIED.search(line) and int(LORA_APPLIED.search(line)["n"]) == 0:
                self._error = (
                    "The LoRA was loaded but applied to 0 layers (unsupported key layout); refusing to serve "
                    "base-model output as LoRA output."
                )
                print(f"[worker] {self._error}", flush=True)
                if self._proc is not None and self._proc.poll() is None:
                    self._proc.terminate()

    def _wait_healthy(self) -> None:
        deadline = time.time() + self.cfg.ready_timeout_s
        url = f"{self.cfg.base_url}/health"
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"sglang exited with code {self._proc.returncode}")
            try:
                if requests.get(url, timeout=3).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(f"SGLang not healthy after {self.cfg.ready_timeout_s}s")

    def alive(self) -> bool:
        return self._proc is None or self._proc.poll() is None

    def wait_ready(
        self,
        timeout: float,
        on_tick: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        """Block until the server is healthy; raise if startup failed, timed out or was cancelled."""
        started = time.time()
        while not self._ready.wait(timeout=5):
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            if self._error:
                raise RuntimeError(self._error)
            if not self.alive():
                raise RuntimeError("SGLang process exited during startup")
            if time.time() - started > timeout:
                raise TimeoutError(f"SGLang not ready after {timeout:.0f}s")
            if on_tick:
                on_tick(time.time() - started)
        if self._error:
            raise RuntimeError(self._error)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
