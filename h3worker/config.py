"""Environment-driven configuration for the MiniMax-H3 Ref2VA worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

VALID_GPU_COUNTS = (1, 2, 4, 8)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class LoraConfig:
    repo: str
    filename: str
    local_path: str
    scale: float
    merge_mode: str
    alpha: int | None


@dataclass(frozen=True)
class WorkerConfig:
    model_repo: str = "MiniMaxAI/MiniMax-H3"
    model_dir: str = "/runpod-volume/models/MiniMax-H3"
    variant: str = "ref2va"
    port: int = 30011
    num_gpus: int | None = None
    output_dir: str = "/tmp/h3-outputs"
    ready_timeout_s: int = 2400
    job_timeout_s: int = 3000
    auto_download: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    lora_repo: str = ""
    lora_file: str = ""
    lora_dir: str = "/runpod-volume/loras"
    lora_scale: float = 1.0
    lora_merge_mode: str = "auto"
    lora_alpha: int | None = None

    @property
    def lora(self) -> LoraConfig | None:
        if not self.lora_repo:
            return None
        if not self.lora_file:
            raise ValueError("H3_LORA_FILE is required when H3_LORA_REPO is set")
        return LoraConfig(
            repo=self.lora_repo,
            filename=self.lora_file,
            local_path=os.path.join(self.lora_dir, self.lora_file),
            scale=self.lora_scale,
            merge_mode=self.lora_merge_mode,
            alpha=self.lora_alpha,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def load_config() -> WorkerConfig:
    import shlex

    num_gpus_raw = _env("H3_NUM_GPUS")
    cfg = WorkerConfig(
        model_repo=_env("H3_MODEL_REPO", WorkerConfig.model_repo),
        model_dir=_env("H3_MODEL_DIR", WorkerConfig.model_dir),
        variant=_env("H3_VARIANT", WorkerConfig.variant).lower(),
        port=_env_int("H3_PORT", WorkerConfig.port),
        num_gpus=int(num_gpus_raw) if num_gpus_raw else None,
        output_dir=_env("H3_OUTPUT_DIR", WorkerConfig.output_dir),
        ready_timeout_s=_env_int("H3_READY_TIMEOUT_S", WorkerConfig.ready_timeout_s),
        job_timeout_s=_env_int("H3_JOB_TIMEOUT_S", WorkerConfig.job_timeout_s),
        auto_download=_env("H3_AUTO_DOWNLOAD", "1") not in ("0", "false", "no"),
        extra_args=tuple(shlex.split(_env("H3_EXTRA_ARGS"))),
        lora_repo=_env("H3_LORA_REPO"),
        lora_file=_env("H3_LORA_FILE"),
        lora_dir=_env("H3_LORA_DIR", WorkerConfig.lora_dir),
        lora_scale=_env_float("H3_LORA_SCALE", WorkerConfig.lora_scale),
        lora_merge_mode=_env("H3_LORA_MERGE_MODE", WorkerConfig.lora_merge_mode),
        lora_alpha=_env_int("H3_LORA_ALPHA", 0) or None,
    )
    if cfg.variant not in ("ref2va", "fl2va"):
        raise ValueError(f"H3_VARIANT must be ref2va or fl2va, got {cfg.variant!r}")
    cfg.lora  # validates LoRA settings eagerly
    return cfg


def detect_gpu_count(configured: int | None) -> int:
    if configured is not None:
        count = configured
    else:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
        count = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    if count not in VALID_GPU_COUNTS:
        raise RuntimeError(
            f"MiniMax-H3 serving needs 1, 2, 4 or 8 GPUs per worker, found {count}"
        )
    return count
