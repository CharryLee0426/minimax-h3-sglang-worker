"""In-process stand-ins for the SGLang video API and a presigned upload target."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeSglang:
    """Minimal ``/v1/videos`` API. ``polls_until_done`` controls queued->completed."""

    def __init__(self, tmp_path, polls_until_done: int = 2, fail_with: str | None = None,
                 reject_with: str | None = None, healthy: bool = True):
        self.requests: list[dict] = []
        self.deleted: list[str] = []
        self.polls: dict[str, int] = {}
        self.polls_until_done = polls_until_done
        self.fail_with = fail_with
        self.reject_with = reject_with
        self.healthy = healthy
        self.video_path = tmp_path / "sglang-output.mp4"
        self.video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 4096)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, code, body=b"", ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200 if outer.healthy else 503, b"{}")
                if self.path.startswith("/v1/videos/") and self.path.endswith("/content"):
                    return self._send(200, outer.video_path.read_bytes(), "video/mp4")
                if self.path.startswith("/v1/videos/"):
                    vid = self.path.rsplit("/", 1)[1]
                    outer.polls[vid] = outer.polls.get(vid, 0) + 1
                    if outer.fail_with:
                        body = {"id": vid, "status": "failed", "error": {"message": outer.fail_with}}
                    elif outer.polls[vid] > outer.polls_until_done:
                        body = {"id": vid, "status": "completed", "progress": 100,
                                "size": "1344x768", "file_path": str(outer.video_path)}
                    else:
                        body = {"id": vid, "status": "queued", "progress": 0}
                    return self._send(200, json.dumps(body).encode())
                self._send(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                if outer.reject_with:
                    return self._send(400, json.dumps({"detail": outer.reject_with}).encode())
                self._send(200, json.dumps({"id": "vid_1", "status": "queued",
                                            "file_path": str(outer.video_path)}).encode())

            def do_DELETE(self):
                outer.deleted.append(self.path.rsplit("/", 1)[1])
                self._send(200, b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeUploadTarget:
    """Records PUT bodies; can answer a fixed number of failures first."""

    def __init__(self, fail_first: int = 0, fail_status: int = 503):
        self.bodies: list[bytes] = []
        self.headers: list[dict] = []
        self.attempts = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_PUT(self):
                outer.attempts += 1
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if outer.attempts <= fail_first:
                    self.send_response(fail_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.bodies.append(body)
                outer.headers.append(dict(self.headers))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/put"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
