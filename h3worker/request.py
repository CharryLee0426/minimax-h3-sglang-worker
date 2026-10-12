"""Validate a job payload and translate it into an SGLang ``/v1/videos`` request."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

ASPECT_RATIOS = ("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
MIN_SECONDS = 4
MAX_SECONDS = 15
MAX_REFERENCE_IMAGES = 9
MAX_PROMPT_CHARS = 12000
SHORT_EDGE = 768
PICTURE_TAG = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)


class InputError(ValueError):
    """The job payload is invalid; retrying cannot help."""


@dataclass(frozen=True)
class OutputTarget:
    put_url: str
    headers: dict[str, str]
    key: str | None


@dataclass(frozen=True)
class JobSpec:
    prompt: str
    references: tuple[str, ...]
    duration_seconds: int
    aspect_ratio: str
    seed: int
    num_inference_steps: int
    flow_shift: float
    audio_flow_shift: float
    output: OutputTarget


def _host_matches(host: str, suffix: str) -> bool:
    """True for the domain itself or any subdomain of it (dot-boundary match)."""
    domain = suffix.strip().lstrip(".").lower()
    return bool(domain) and (host == domain or host.endswith("." + domain))


def _require_https(url: object, label: str, allowed_suffixes: tuple[str, ...] = ()) -> str:
    if not isinstance(url, str) or not url:
        raise InputError(f"{label} must be a non-empty string")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise InputError(f"{label} must be an https URL")
    host = (parts.hostname or "").lower()
    if allowed_suffixes and not any(_host_matches(host, s) for s in allowed_suffixes):
        raise InputError(f"{label} host {host!r} is not allowed")
    return url


def _int_in_range(value: object, label: str, low: int, high: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise InputError(f"{label} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise InputError(f"{label} must be between {low} and {high}")
    return value


def _positive_float(value: object, label: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 100:
        raise InputError(f"{label} must be a number between 0 and 100")
    return float(value)


def parse_job_input(payload: object, allowed_host_suffixes: tuple[str, ...] = ()) -> JobSpec:
    if not isinstance(payload, dict):
        raise InputError("input must be an object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InputError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")

    raw_refs = payload.get("references")
    if not isinstance(raw_refs, list) or not raw_refs:
        raise InputError("at least one reference image is required")
    if len(raw_refs) > MAX_REFERENCE_IMAGES:
        raise InputError(f"at most {MAX_REFERENCE_IMAGES} reference images are supported")
    references = []
    for index, item in enumerate(raw_refs, start=1):
        url = item.get("url") if isinstance(item, dict) else item
        references.append(_require_https(url, f"references[{index}].url", allowed_host_suffixes))

    aspect_ratio = payload.get("aspect_ratio", "16:9")
    if aspect_ratio not in ASPECT_RATIOS:
        raise InputError(f"aspect_ratio must be one of {', '.join(ASPECT_RATIOS)}")

    output = payload.get("output")
    if not isinstance(output, dict):
        raise InputError("output.put_url is required")
    headers = output.get("headers") or {}
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise InputError("output.headers must map strings to strings")

    seed = payload.get("seed")
    seed = random.randint(0, 2**31 - 1) if seed is None else _int_in_range(seed, "seed", 0, 2**63 - 1, 0)

    return JobSpec(
        prompt=prompt.strip(),
        references=tuple(references),
        duration_seconds=_int_in_range(
            payload.get("duration_seconds"), "duration_seconds", MIN_SECONDS, MAX_SECONDS, 5
        ),
        aspect_ratio=aspect_ratio,
        seed=seed,
        num_inference_steps=_int_in_range(
            payload.get("num_inference_steps"), "num_inference_steps", 1, 100, 50
        ),
        flow_shift=_positive_float(payload.get("flow_shift"), "flow_shift", 12.0),
        audio_flow_shift=_positive_float(payload.get("audio_flow_shift"), "audio_flow_shift", 3.0),
        output=OutputTarget(
            put_url=_require_https(output.get("put_url"), "output.put_url", allowed_host_suffixes),
            headers=dict(headers),
            key=output.get("key") if isinstance(output.get("key"), str) else None,
        ),
    )


def effective_prompt(spec: JobSpec) -> str:
    """Make sure every reference image is addressable by a ``<Picture N>`` tag."""
    if PICTURE_TAG.search(spec.prompt):
        return spec.prompt
    tags = ", ".join(f"<Picture {i}>" for i in range(1, len(spec.references) + 1))
    noun = "image" if len(spec.references) == 1 else "images"
    return f"Use {tags} as the visual reference {noun}. {spec.prompt}"


def build_video_request(spec: JobSpec) -> dict:
    return {
        "prompt": effective_prompt(spec),
        "seconds": spec.duration_seconds,
        "task": "ref2va",
        "conditions": [
            {"type": "image", "uri": url, "role": "reference"} for url in spec.references
        ],
        "target": {
            "short_edge": SHORT_EDGE,
            "aspect_ratio": spec.aspect_ratio,
            "duration_seconds": float(spec.duration_seconds),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": spec.num_inference_steps,
        "flow_shift": spec.flow_shift,
        "audio_flow_shift": spec.audio_flow_shift,
        "seed": spec.seed,
    }
