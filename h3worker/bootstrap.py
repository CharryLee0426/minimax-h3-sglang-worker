"""Download and verify model weights on the shared network volume.

Self-contained (needs only ``huggingface_hub``) so the same file can run inside the
worker image or on a cheap CPU pod that pre-populates the volume.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
MODEL_DIR = os.environ.get("H3_MODEL_DIR", "/runpod-volume/models/MiniMax-H3")
LORA_DIR = os.environ.get("H3_LORA_DIR", "/runpod-volume/loras")
PARTITIONS = {"ref2va": "Ref2VA", "fl2va": "FL2VA"}
MARKER = ".h3-complete"
# Components whose safetensors shards are listed in a model.safetensors.index.json.
SHARDED_COMPONENTS = ("transformer", "text_encoder")


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def missing_files(model_dir: str, variant: str) -> list[str]:
    """Return files the partition needs that are absent or empty."""
    root = Path(model_dir)
    partition = PARTITIONS[variant]
    required = [root / "model_index.json", root / partition / "model_index.json"]
    for component in SHARDED_COMPONENTS:
        index_path = root / partition / component / "model.safetensors.index.json"
        if not index_path.is_file():
            required.append(index_path)
            continue
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        required.extend(index_path.parent / shard for shard in shards)
    required.extend(
        [
            root / partition / "video_vae" / "source" / "model.safetensors",
            root / partition / "audio_vae" / "model.safetensors",
        ]
    )
    return [
        str(path.relative_to(root))
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]


def ensure_model(
    model_repo: str = MODEL_REPO, model_dir: str = MODEL_DIR, variant: str = "ref2va"
) -> str:
    """Make sure the partition's weights are on disk; return the model directory."""
    if variant not in PARTITIONS:
        raise ValueError(f"unknown variant {variant!r}")
    marker = Path(model_dir) / f"{MARKER}-{variant}"
    if marker.is_file() and not missing_files(model_dir, variant):
        log(f"{variant} weights already present in {model_dir}")
        return model_dir

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    started = time.time()
    log(f"downloading {model_repo} ({PARTITIONS[variant]}/*) to {model_dir}")
    snapshot_download(
        repo_id=model_repo,
        local_dir=model_dir,
        allow_patterns=["model_index.json", f"{PARTITIONS[variant]}/*"],
        max_workers=16,
    )
    missing = missing_files(model_dir, variant)
    if missing:
        raise RuntimeError(f"download finished but files are missing: {missing[:5]}")
    marker.write_text(str(int(time.time())))
    log(f"{variant} weights ready in {time.time() - started:.0f}s")
    return model_dir


def ensure_lora(repo: str, filename: str, lora_dir: str = LORA_DIR) -> str:
    """Download a single LoRA safetensors file; return its local path."""
    target = Path(lora_dir) / filename
    if target.is_file() and target.stat().st_size > 0:
        log(f"LoRA already present: {target}")
        return str(target)

    from huggingface_hub import hf_hub_download

    Path(lora_dir).mkdir(parents=True, exist_ok=True)
    log(f"downloading LoRA {repo}/{filename}")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=lora_dir)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"LoRA download did not produce {target}")
    return str(target)


def main() -> int:
    variant = os.environ.get("H3_VARIANT", "ref2va").lower()
    ensure_model(variant=variant)
    lora_repo = os.environ.get("H3_LORA_REPO", "").strip()
    lora_files = [
        name.strip()
        for name in os.environ.get("H3_LORA_FILES", os.environ.get("H3_LORA_FILE", "")).split(",")
        if name.strip()
    ]
    if lora_repo:
        for name in lora_files:
            ensure_lora(lora_repo, name)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
