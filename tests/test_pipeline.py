import asyncio
import os

import pytest

from h3worker import pipeline
from h3worker.config import WorkerConfig
from h3worker.handler import IdleWatchdog, ProgressReporter, make_handler
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
        self.stopped = False

    def wait_ready(self, timeout, on_tick=None, cancel=None):
        self.waited += 1
        if self.error:
            raise RuntimeError(self.error)
        if on_tick:
            on_tick(12.0)

    def stop(self):
        self.stopped = True


def make(tmp_path, sg, server, exit_fn=lambda code: None):
    cfg = WorkerConfig(port=int(sg.base_url.rsplit(":", 1)[1]), output_dir=str(tmp_path))
    sent = []
    handler = make_handler(cfg, server, lambda job, update: sent.append(update), exit_fn)
    return handler, sent


def call(handler, job):
    return asyncio.run(handler(job))


def test_handler_happy_path_returns_output(tmp_path, monkeypatch):
    import h3worker.request as rq

    # the fakes speak plain http, which the https-only validator would reject
    monkeypatch.setattr(rq, "_require_https", lambda url, label, allowed=(): url)
    with FakeSglang(tmp_path) as sg, FakeUploadTarget() as up:
        handler, sent = make(tmp_path, sg, StubServer())
        payload = spec_for(up, seed=1)
        payload["output"]["put_url"] = up.url
        result = call(handler, {"id": "job-1", "input": payload})
    assert result["video_key"] == "videos/j1.mp4"
    assert {"stage": "loading_model", "elapsed_s": 12} in sent


def test_handler_invalid_input_is_a_clean_error(tmp_path):
    with FakeSglang(tmp_path) as sg:
        handler, _ = make(tmp_path, sg, StubServer())
        result = call(handler, {"id": "job-2", "input": {"prompt": ""}})
    assert result == {"error": "Invalid input: prompt is required"}


def test_handler_server_failure_recycles_worker(tmp_path):
    with FakeSglang(tmp_path) as sg, FakeUploadTarget() as up:
        handler, _ = make(tmp_path, sg, StubServer(error="CUDA error"))
        result = call(handler, {"id": "job-3", "input": spec_for(up)})
    assert "CUDA error" in result["error"] and result["refresh_worker"] is True


def test_handler_generation_error_keeps_worker(tmp_path):
    with FakeSglang(tmp_path, reject_with="bad aspect") as sg, FakeUploadTarget() as up:
        handler, _ = make(tmp_path, sg, StubServer())
        result = call(handler, {"id": "job-4", "input": spec_for(up)})
    assert "bad aspect" in result["error"] and "refresh_worker" not in result


def test_handler_generation_timeout_recycles_worker(tmp_path):
    with FakeSglang(tmp_path, polls_until_done=10_000) as sg, FakeUploadTarget() as up:
        cfg = WorkerConfig(port=int(sg.base_url.rsplit(":", 1)[1]), output_dir=str(tmp_path), job_timeout_s=0)
        handler = make_handler(cfg, StubServer(), lambda job, update: None)
        result = call(handler, {"id": "job-5", "input": spec_for(up)})
    assert "exceeded" in result["error"] and result["refresh_worker"] is True


def test_generation_failure_removes_sglang_output(tmp_path):
    with FakeSglang(tmp_path, fail_with="boom") as sg, FakeUploadTarget() as up:
        assert sg.video_path.exists()
        with pytest.raises(GenerationError):
            run_job(parse(up), base_url=sg.base_url, output_dir=str(tmp_path),
                    progress=lambda _u: None, job_timeout_s=60)
        assert not sg.video_path.exists()


def test_cancelling_a_job_stops_sglang_and_recycles_the_worker(tmp_path):
    exits = []
    with FakeSglang(tmp_path, polls_until_done=10_000) as sg, FakeUploadTarget() as up:
        server = StubServer()
        handler, _ = make(tmp_path, sg, server, exit_fn=exits.append)

        async def scenario():
            task = asyncio.create_task(handler({"id": "job-6", "input": spec_for(up)}))
            for _ in range(200):  # wait until SGLang has the request and the worker is polling
                if sg.requests and sg.polls:
                    break
                await asyncio.sleep(0.05)
            assert sg.requests, "job never reached SGLang"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
    assert server.stopped is True
    assert exits == [0]


class TestProgressReporter:
    def test_sends_on_stage_change_and_throttles_repeats(self, monkeypatch):
        sent = []
        clock = iter([0.0, 1.0, 2.0, 20.0, 21.0])
        monkeypatch.setattr("h3worker.handler.time.monotonic", lambda: next(clock))
        report = ProgressReporter(sent.append, min_interval_s=15.0)
        report({"stage": "generating", "elapsed_s": 0})   # first -> sent
        report({"stage": "generating", "elapsed_s": 1})   # same stage, too soon -> dropped
        report({"stage": "uploading"})                    # stage change -> sent
        report({"stage": "uploading", "elapsed_s": 18})   # same stage, 18s later -> sent
        report({"stage": "uploading", "elapsed_s": 19})   # too soon -> dropped
        assert [u["stage"] for u in sent] == ["generating", "uploading", "uploading"]


class TestIdleWatchdog:
    def make(self):
        state = {"now": 0.0, "ready": False, "exits": []}
        dog = IdleWatchdog(150, lambda: state["ready"], state["exits"].append, lambda: state["now"])
        return dog, state

    def test_never_exits_while_loading(self):
        dog, s = self.make()
        s["now"] = 10_000
        assert dog.check() is False and s["exits"] == []

    def test_exits_after_being_ready_and_idle_long_enough(self):
        dog, s = self.make()
        s["now"] = 1000; s["ready"] = False; dog.check()   # still loading: timer keeps resetting
        s["ready"] = True
        s["now"] = 1100; assert dog.check() is False
        s["now"] = 1149; assert dog.check() is False
        s["now"] = 1151; assert dog.check() is True
        assert s["exits"] == [0]

    def test_active_jobs_block_exit_and_restart_the_timer(self):
        dog, s = self.make()
        s["ready"] = True
        dog.job_started()
        s["now"] = 5000; assert dog.check() is False
        dog.job_finished()
        s["now"] = 5100; assert dog.check() is False
        s["now"] = 5151; assert dog.check() is True

    def test_handler_reports_jobs_to_the_watchdog(self, tmp_path):
        dog, s = self.make()
        s["ready"] = True
        with FakeSglang(tmp_path) as sg:
            cfg = WorkerConfig(port=int(sg.base_url.rsplit(":", 1)[1]), output_dir=str(tmp_path))
            handler = make_handler(cfg, StubServer(), lambda job, update: None, watchdog=dog)
            asyncio.run(handler({"id": "j", "input": {"prompt": ""}}))  # invalid input never counts as busy
            assert dog._active == 0
