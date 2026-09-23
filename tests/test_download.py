import httpx

from entertainer.data import download


def _serve(monkeypatch, body: bytes, etag: str) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        headers = {"etag": etag, "content-length": str(len(body))}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        rng = request.headers.get("range")
        if rng and request.headers.get("if-range") == etag:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=body[start:], headers={"etag": etag})
        return httpx.Response(200, content=body, headers={"etag": etag})

    real = httpx.Client
    monkeypatch.setattr(
        download.httpx, "Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


def test_resumes_the_same_version(tmp_path, monkeypatch):
    dest = tmp_path / "f.gz"
    dest.write_bytes(b"hello ")
    download._validator_path(dest).write_text('"v1"')
    _serve(monkeypatch, b"hello world", '"v1"')

    download.fetch("https://example.test/f.gz", dest)

    assert dest.read_bytes() == b"hello world"


def test_a_new_version_replaces_a_partial_copy(tmp_path, monkeypatch):
    dest = tmp_path / "f.gz"
    dest.write_bytes(b"old pa")
    download._validator_path(dest).write_text('"v1"')
    _serve(monkeypatch, b"new version!", '"v2"')

    download.fetch("https://example.test/f.gz", dest)

    assert dest.read_bytes() == b"new version!"
    assert download._validator_path(dest).read_text() == '"v2"'


def test_a_partial_copy_of_unknown_version_is_not_spliced(tmp_path, monkeypatch):
    dest = tmp_path / "f.gz"
    dest.write_bytes(b"old pa")
    seen = _serve(monkeypatch, b"new version!", '"v2"')

    download.fetch("https://example.test/f.gz", dest)

    assert dest.read_bytes() == b"new version!"
    assert "range" not in seen[-1].headers


def test_a_complete_current_copy_is_skipped(tmp_path, monkeypatch):
    dest = tmp_path / "f.gz"
    dest.write_bytes(b"same")
    download._validator_path(dest).write_text('"v1"')
    seen = _serve(monkeypatch, b"same", '"v1"')

    download.fetch("https://example.test/f.gz", dest)

    assert [r.method for r in seen] == ["HEAD"]
