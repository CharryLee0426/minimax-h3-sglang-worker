import os

import pytest

from h3worker import pipeline
from h3worker.config import WorkerConfig
from h3worker.handler import make_handler
from h3worker.pipeline import GenerationError, UploadError, run_job
from h3worker.request import parse_job_input
from tests.fakes import FakeSglang, FakeUploadTarget


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)


def spec_for(target: FakeUploadTarget, **extra):
    payload = {
        "prompt": "A quiet street at dusk.",
        "references": [{"url": "https://refs.example.com/a.png"}],
        "output": {"put_url": target.url.replace("http://", "https://"), "key": "videos/j1.mp4",
                   "headers": {"x-test": "1"}},
        **extra,
    }
    return payload


def parse(target, **extra):
    payload = spec_for(target, **extra)
    # tests talk to plain-http fakes; swap the scheme after validation
    spec = parse_job_input(payload)
    object.__setattr__(spec.output, "put_url", target.url)
    return spec


def test_run_job_uploads_video_and_cleans_up(tmp_path):
    updates = []
    with FakeSglang(tmp_path) as sg, FakeUploadTarget() as up:
        out = run_job(
            parse(up, seed=7),
            base_url=sg.base_url,
            output_dir=str(tmp_path),
            progress=updates.append,
            job_timeout_s=60,
        )
        assert up.bodies and up.bodies[0].startswith(b"\x00\x00\x00\x18ftyp")
        assert up.headers[0]["Content-Type"] == "video/mp4"
        assert up.headers[0]["x-test"] == "1"
        assert sg.requests[0]["seed"] == 7
        assert sg.deleted == ["vid_1"]
    assert out["video_key"] == "videos/j1.mp4"
    assert out["size_bytes"] == len(up.bodies[0])
    assert out["resolution"] == "1344x768"
    stages = [u["stage"] for u in updates]
    assert stages[0] == "submitting" and "generating" in stages and stages[-1] == "uploading"
    assert not [f for f in os.listdir(tmp_path) if f.startswith("upload-")]
    # the SGLang-side file is removed as well
    assert not sg.video_path.exists()


def test_generation_failure_is_reported(tmp_path):
    with FakeSglang(tmp_path, fail_with="CUDA out of memory") as sg, FakeUploadTarget() as up:
        with pytest.raises(GenerationError, match="CUDA out of memory"):
            run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                    progress=lambda _u: None, job_timeout_s=60)
        assert up.bodies == []


def test_request_rejection_is_reported(tmp_path):
    with FakeSglang(tmp_path, reject_with="aspect_ratio invalid") as sg, FakeUploadTarget() as up:
        with pytest.raises(GenerationError, match="aspect_ratio invalid"):
            run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                    progress=lambda _u: None, job_timeout_s=60)


def test_upload_retries_transient_errors(tmp_path):
    with FakeSglang(tmp_path) as sg, FakeUploadTarget(fail_first=2) as up:
        run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                progress=lambda _u: None, job_timeout_s=60)
        assert up.attempts == 3 and len(up.bodies) == 1


def test_upload_gives_up_on_client_errors(tmp_path):
    with FakeSglang(tmp_path) as sg, FakeUploadTarget(fail_first=99, fail_status=403) as up:
        with pytest.raises(UploadError, match="403"):
            run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                    progress=lambda _u: None, job_timeout_s=60)
        assert up.attempts == 1


def test_generation_timeout(tmp_path, monkeypatch):
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(pipeline.time, "time", lambda: next(clock))
    with FakeSglang(tmp_path, polls_until_done=10_000) as sg, FakeUploadTarget() as up:
        with pytest.raises(GenerationError, match="exceeded"):
            run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                    progress=lambda _u: None, job_timeout_s=250)


class StubServer:
    def __init__(self, error=None):
        self.error = error
        self.waited = 0

    def wait_ready(self, timeout, on_tick=None):
        self.waited += 1
        if self.error:
            raise RuntimeError(self.error)
        if on_tick:
            on_tick(12.0)


def make(tmp_path, sg, server):
    cfg = WorkerConfig(port=int(sg.base_url.rsplit(":", 1)[1]), output_dir=str(tmp_path))
    sent = []
    return make_handler(cfg, server, lambda job, update: sent.append(update)), sent


def test_handler_happy_path_returns_output(tmp_path, monkeypatch):
    import h3worker.request as rq

    # the fakes speak plain http, which the https-only validator would reject
    monkeypatch.setattr(rq, "_require_https", lambda url, label, allowed=(): url)
    with FakeSglang(tmp_path) as sg, FakeUploadTarget() as up:
        handler, sent = make(tmp_path, sg, StubServer())
        payload = spec_for(up, seed=1)
        payload["output"]["put_url"] = up.url
        result = handler({"id": "job-1", "input": payload})
    assert result["video_key"] == "videos/j1.mp4"
    assert {"stage": "loading_model", "elapsed_s": 12} in sent


def test_handler_invalid_input_is_a_clean_error(tmp_path):
    with FakeSglang(tmp_path) as sg:
        handler, _ = make(tmp_path, sg, StubServer())
        result = handler({"id": "job-2", "input": {"prompt": ""}})
    assert result == {"error": "Invalid input: prompt is required"}


def test_handler_server_failure_recycles_worker(tmp_path):
    with FakeSglang(tmp_path) as sg, FakeUploadTarget() as up:
        handler, _ = make(tmp_path, sg, StubServer(error="CUDA error"))
        result = handler({"id": "job-3", "input": spec_for(up)})
    assert "CUDA error" in result["error"] and result["refresh_worker"] is True
