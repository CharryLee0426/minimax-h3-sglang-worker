import pytest

from h3worker.request import (
    InputError,
    build_video_request,
    effective_prompt,
    parse_job_input,
)

OUT = {"put_url": "https://store.example.com/videos/a.mp4?sig=1", "key": "videos/a.mp4"}


def payload(**overrides):
    base = {
        "prompt": "She walks through the rain.",
        "references": [{"url": "https://store.example.com/ref/1.png"}],
        "output": dict(OUT),
    }
    base.update(overrides)
    return base


def test_defaults_and_request_shape():
    spec = parse_job_input(payload())
    req = build_video_request(spec)
    assert req["task"] == "ref2va"
    assert req["seconds"] == 5
    assert req["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}
    assert req["conditions"] == [
        {"type": "image", "uri": "https://store.example.com/ref/1.png", "role": "reference"}
    ]
    assert req["num_inference_steps"] == 50
    assert req["flow_shift"] == 12.0 and req["audio_flow_shift"] == 3.0
    assert req["num_outputs_per_prompt"] == 1
    assert 0 <= req["seed"] < 2**31


def test_picture_tags_added_when_missing():
    spec = parse_job_input(
        payload(references=[{"url": "https://a.example.com/1.png"}, {"url": "https://a.example.com/2.png"}])
    )
    assert effective_prompt(spec).startswith("Use <Picture 1>, <Picture 2> as the visual reference images.")


def test_existing_picture_tags_are_kept():
    spec = parse_job_input(payload(prompt="<Picture 1> dances."))
    assert effective_prompt(spec) == "<Picture 1> dances."


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prompt": "  "}, "prompt is required"),
        ({"references": []}, "at least one reference"),
        ({"references": [{"url": "https://a.example.com/x.png"}] * 10}, "at most 9"),
        ({"references": [{"url": "http://a.example.com/x.png"}]}, "https"),
        ({"duration_seconds": 3}, "between 4 and 15"),
        ({"duration_seconds": 16}, "between 4 and 15"),
        ({"duration_seconds": 5.5}, "integer"),
        ({"aspect_ratio": "2:1"}, "aspect_ratio"),
        ({"num_inference_steps": 0}, "between 1 and 100"),
        ({"output": {"put_url": "ftp://x"}}, "https"),
        ({"output": None}, "output.put_url is required"),
    ],
)
def test_rejects_invalid_input(overrides, message):
    with pytest.raises(InputError, match=message):
        parse_job_input(payload(**overrides))


def test_rejects_non_object():
    with pytest.raises(InputError):
        parse_job_input("nope")


def test_host_allowlist():
    parse_job_input(payload(), (".example.com",))
    with pytest.raises(InputError, match="not allowed"):
        parse_job_input(payload(), (".blob.vercel-storage.com",))


def test_host_allowlist_requires_a_dot_boundary():
    ok = payload(references=[{"url": "https://a.b.example.com/1.png"}])
    parse_job_input(ok, (".example.com",))
    parse_job_input(ok, ("example.com",))
    evil = payload(references=[{"url": "https://evilexample.com/1.png"}])
    with pytest.raises(InputError, match="not allowed"):
        parse_job_input(evil, (".example.com",))
    with pytest.raises(InputError, match="not allowed"):
        parse_job_input(evil, ("example.com",))
