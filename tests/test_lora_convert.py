import json
import struct

import pytest

from h3worker.config import WorkerConfig
from h3worker.lora_convert import LoraFormatError, convert_header, ensure_native_lora, read_header
from h3worker.sglang_server import SglangServer


def write_safetensors(path, tensors: dict[str, bytes], dtype="F32", shape=(2,), metadata=None):
    header, offset, blob = {}, 0, b""
    for name, data in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blob += data
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


KOHYA = {
    "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": b"AAAAAAAA",
    "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": b"BBBBBBBB",
    "lora_unet_blocks_0_attn_qkv_proj.alpha": b"CCCCCCCC",
    "lora_unet_blocks_12_mlp_fc2.lora_down.weight": b"DDDDDDDD",
    "lora_unet_blocks_12_mlp_fc2.lora_up.weight": b"EEEEEEEE",
    "lora_unet_blocks_7_attn_out_proj.lora_up.weight": b"FFFFFFFF",
    "lora_unet_blocks_7_mlp_fc1.lora_down.weight": b"GGGGGGGG",
}


def test_header_keys_are_renamed_to_native_names():
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in KOHYA}
    names = set(convert_header(header)) - {"__metadata__"}
    assert names == {
        "blocks.0.attn.qkv_proj.lora_A.weight",
        "blocks.0.attn.qkv_proj.lora_B.weight",
        "blocks.0.attn.qkv_proj.alpha",
        "blocks.12.mlp.fc2.lora_A.weight",
        "blocks.12.mlp.fc2.lora_B.weight",
        "blocks.7.attn.out_proj.lora_B.weight",
        "blocks.7.mlp.fc1.lora_A.weight",
    }


def test_conversion_keeps_tensor_bytes_and_metadata(tmp_path):
    src = tmp_path / "style.safetensors"
    write_safetensors(src, KOHYA, metadata={"ss_network_dim": "64"})
    out = ensure_native_lora(str(src))
    assert out.endswith("style.native.safetensors")
    header, start = read_header(out)
    assert header["__metadata__"]["ss_network_dim"] == "64"
    data = open(out, "rb").read()[start:]
    for old, new in [
        ("lora_unet_blocks_0_attn_qkv_proj.lora_up.weight", "blocks.0.attn.qkv_proj.lora_B.weight"),
        ("lora_unet_blocks_7_mlp_fc1.lora_down.weight", "blocks.7.mlp.fc1.lora_A.weight"),
    ]:
        a, b = header[new]["data_offsets"]
        assert data[a:b] == KOHYA[old]
        assert header[new]["dtype"] == "F32"


def test_native_files_are_left_alone_and_conversion_is_cached(tmp_path):
    native = tmp_path / "native.safetensors"
    write_safetensors(native, {"blocks.0.attn.qkv_proj.lora_A.weight": b"AAAAAAAA"})
    assert ensure_native_lora(str(native)) == str(native)

    src = tmp_path / "k.safetensors"
    write_safetensors(src, KOHYA)
    first = ensure_native_lora(str(src))
    mtime = (tmp_path / "k.native.safetensors").stat().st_mtime_ns
    assert ensure_native_lora(str(src)) == first
    assert (tmp_path / "k.native.safetensors").stat().st_mtime_ns == mtime
    assert not list(tmp_path.glob("*.partial"))


def test_unknown_keys_are_rejected(tmp_path):
    src = tmp_path / "bad.safetensors"
    write_safetensors(src, {**KOHYA, "lora_unet_blocks_0_attn_mystery.lora_up.weight": b"XXXXXXXX"})
    with pytest.raises(LoraFormatError, match="mystery"):
        ensure_native_lora(str(src))
    assert not list(tmp_path.glob("*.native*"))


def make_server(with_lora=True):
    cfg = WorkerConfig(lora_repo="a/b" if with_lora else "", lora_file="x.safetensors" if with_lora else "")
    server = SglangServer(cfg)
    return server


def test_zero_layer_lora_is_a_fatal_error(capsys):
    server = make_server()
    server.pump_output(iter([
        "[10-12 05:39:22] Rank 0: loaded LoRA adapter /x/y.safetensors\n",
        "[10-12 05:39:22] Rank 0: LoRA adapter(s) /x/y.safetensors applied to 0 layers (targets: all)\n",
    ]))
    assert "applied to 0 layers" in server.error
    assert "applied to 0 layers" in capsys.readouterr().out


def test_nonzero_layer_lora_and_no_lora_are_fine():
    server = make_server()
    server.pump_output(iter(["LoRA adapter(s) /x applied to 200 layers (targets: all)\n"]))
    assert server.error is None
    plain = make_server(with_lora=False)
    plain.pump_output(iter(["applied to 0 layers\n"]))
    assert plain.error is None
