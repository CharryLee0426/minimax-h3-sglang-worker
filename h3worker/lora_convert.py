"""Rename Kohya-style MiniMax-H3 LoRA keys to the native names SGLang v0.5.21 matches.

Trainers built on ``networks.lora_minimax_h3`` write ``lora_unet_blocks_<N>_attn_qkv_proj.lora_down.weight``.
SGLang only matches ``blocks.<N>.attn.qkv_proj.lora_A.weight``, so the unconverted file loads but applies to
zero layers. Only the safetensors header is rewritten; tensor bytes are copied as they are.
"""

from __future__ import annotations

import json
import os
import re
import struct

KOHYA_PREFIX = "lora_unet_blocks_"
_KEY = re.compile(
    r"^lora_unet_blocks_(?P<block>\d+)_(?P<module>attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)"
    r"\.(?P<part>lora_down\.weight|lora_up\.weight|alpha)$"
)
_MODULE = {
    "attn_qkv_proj": "attn.qkv_proj",
    "attn_out_proj": "attn.out_proj",
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
}
_PART = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight", "alpha": "alpha"}


class LoraFormatError(ValueError):
    """The file looks like a Kohya H3 LoRA but contains keys this converter does not understand."""


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length)), 8 + length


def is_kohya_h3(header: dict) -> bool:
    return any(key.startswith(KOHYA_PREFIX) for key in header if key != "__metadata__")


def convert_header(header: dict) -> dict:
    """Return a header with native key names; raises LoraFormatError on unknown keys."""
    converted: dict = {}
    for key, value in header.items():
        if key == "__metadata__":
            converted[key] = dict(value or {})
            continue
        match = _KEY.match(key)
        if match is None:
            raise LoraFormatError(f"unsupported LoRA key: {key}")
        name = f"blocks.{match['block']}.{_MODULE[match['module']]}.{_PART[match['part']]}"
        if name in converted:
            raise LoraFormatError(f"duplicate LoRA key after conversion: {name}")
        converted[name] = value
    converted.setdefault("__metadata__", {})["h3_converted_from"] = "kohya"
    return converted


def ensure_native_lora(path: str) -> str:
    """Return a path SGLang can apply: the input if already native, else a converted copy next to it."""
    header, data_start = read_header(path)
    if not is_kohya_h3(header):
        return path
    target = f"{path[: -len('.safetensors')] if path.endswith('.safetensors') else path}.native.safetensors"
    if os.path.isfile(target) and os.path.getmtime(target) >= os.path.getmtime(path):
        return target

    new_header = json.dumps(convert_header(header), separators=(",", ":")).encode()
    new_header += b" " * (-len(new_header) % 8)
    partial = f"{target}.partial"
    try:
        with open(path, "rb") as src, open(partial, "wb") as dst:
            dst.write(struct.pack("<Q", len(new_header)))
            dst.write(new_header)
            src.seek(data_start)
            while chunk := src.read(8 << 20):
                dst.write(chunk)
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return target
