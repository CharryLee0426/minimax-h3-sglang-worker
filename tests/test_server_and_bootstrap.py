import json

import pytest

from h3worker import bootstrap
from h3worker.config import WorkerConfig, detect_gpu_count, load_config
from h3worker.sglang_server import build_command


def test_command_without_lora():
    cmd = build_command(WorkerConfig(), 4, "/models/h3")
    assert cmd[:3] == ["sglang", "serve", "--model-path"]
    assert cmd[cmd.index("--model-variant") + 1] == "ref2va"
    assert cmd[cmd.index("--num-gpus") + 1] == "4"
    assert cmd[cmd.index("--ulysses-degree") + 1] == "4"
    assert "--lora-path" not in cmd


def test_command_with_lora():
    cfg = WorkerConfig(
        lora_repo="example/h3-lora",
        lora_file="style_ref2va_v1.safetensors",
        lora_dir="/vol/loras",
        lora_scale=0.9,
    )
    cmd = build_command(cfg, 2, "/models/h3")
    assert cmd[cmd.index("--lora-path") + 1] == (
        "/vol/loras/style_ref2va_v1.safetensors"
    )
    assert cmd[cmd.index("--lora-scale") + 1] == "0.9"
    assert cmd[cmd.index("--lora-merge-mode") + 1] == "auto"


def test_extra_args_are_appended(monkeypatch):
    monkeypatch.setenv("H3_EXTRA_ARGS", "--attention-backend fa")
    cmd = build_command(load_config(), 8, "/m")
    assert cmd[-2:] == ["--attention-backend", "fa"]


def test_lora_requires_filename(monkeypatch):
    monkeypatch.setenv("H3_LORA_REPO", "a/b")
    with pytest.raises(ValueError, match="H3_LORA_FILE"):
        load_config()


def test_invalid_variant(monkeypatch):
    monkeypatch.setenv("H3_VARIANT", "t2va")
    with pytest.raises(ValueError, match="H3_VARIANT"):
        load_config()


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_valid_gpu_counts(count):
    assert detect_gpu_count(count) == count


@pytest.mark.parametrize("count", [0, 3, 6])
def test_invalid_gpu_counts(count):
    with pytest.raises(RuntimeError, match="1, 2, 4 or 8"):
        detect_gpu_count(count)


def make_model_dir(root, complete=True):
    part = root / "Ref2VA"
    (root / "model_index.json").write_text("{}")
    (part / "model_index.json").parent.mkdir(parents=True)
    (part / "model_index.json").write_text("{}")
    for component, shards in (("transformer", ["a", "b"]), ("text_encoder", ["c"])):
        d = part / component
        d.mkdir()
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"w{i}": f"{s}.safetensors" for i, s in enumerate(shards)}})
        )
        for s in shards:
            (d / f"{s}.safetensors").write_bytes(b"x")
    (part / "video_vae" / "source").mkdir(parents=True)
    (part / "video_vae" / "source" / "model.safetensors").write_bytes(b"x")
    (part / "audio_vae").mkdir()
    (part / "audio_vae" / "model.safetensors").write_bytes(b"x")
    if not complete:
        (part / "transformer" / "b.safetensors").unlink()


def test_missing_files_detects_complete_and_partial(tmp_path):
    make_model_dir(tmp_path)
    assert bootstrap.missing_files(str(tmp_path), "ref2va") == []
    (tmp_path / "Ref2VA" / "transformer" / "b.safetensors").unlink()
    assert bootstrap.missing_files(str(tmp_path), "ref2va") == ["Ref2VA/transformer/b.safetensors"]


def test_ensure_model_skips_download_when_marker_and_files_present(tmp_path, monkeypatch):
    make_model_dir(tmp_path)
    (tmp_path / f"{bootstrap.MARKER}-ref2va").write_text("1")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)  # import would fail
    assert bootstrap.ensure_model("x/y", str(tmp_path), "ref2va") == str(tmp_path)


def test_ensure_model_downloads_when_partial(tmp_path, monkeypatch):
    make_model_dir(tmp_path, complete=False)
    (tmp_path / f"{bootstrap.MARKER}-ref2va").write_text("1")
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            (tmp_path / "Ref2VA" / "transformer" / "b.safetensors").write_bytes(b"x")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", FakeHub)
    bootstrap.ensure_model("MiniMaxAI/MiniMax-H3", str(tmp_path), "ref2va")
    assert calls[0]["allow_patterns"] == ["model_index.json", "Ref2VA/*"]
    assert calls[0]["repo_id"] == "MiniMaxAI/MiniMax-H3"


def test_ensure_lora_skips_when_present(tmp_path):
    (tmp_path / "x.safetensors").write_bytes(b"abc")
    assert bootstrap.ensure_lora("a/b", "x.safetensors", str(tmp_path)) == str(tmp_path / "x.safetensors")
