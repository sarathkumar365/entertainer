"""Binding and token rules for `ent rate`.

These are security decisions that lived in a Typer callback and had no test
at all — `ent rate` starts a server, so it is not something the suite can
invoke. Whether a token is mandatory depends on which interface is bound, and
getting that wrong hands somebody on the same network write access to the
verdict log.
"""

from __future__ import annotations

import pytest

from entertainer.web import serve


def test_localhost_needs_no_token():
    binding = serve.resolve()
    assert binding.host == "127.0.0.1"
    assert binding.token is None
    assert binding.off_loopback is False
    assert binding.url() == "http://127.0.0.1:8756"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_every_loopback_spelling_counts_as_loopback(host):
    assert serve.resolve(host=host).off_loopback is False


def test_lan_binds_every_interface_and_mints_a_token():
    binding = serve.resolve(lan=True)
    assert binding.host == "0.0.0.0"  # noqa: S104
    assert binding.off_loopback is True
    assert binding.token, "a LAN binding must not be unauthenticated"
    assert len(binding.token) >= 16


def test_an_explicit_off_loopback_host_also_requires_a_token():
    """--lan is not the only way off the loopback interface."""
    binding = serve.resolve(host="192.168.1.50")
    assert binding.off_loopback is True
    assert binding.token


def test_a_supplied_token_is_respected_rather_than_replaced():
    assert serve.resolve(lan=True, token="mine").token == "mine"


def test_tokens_are_not_reused_between_bindings():
    assert serve.resolve(lan=True).token != serve.resolve(lan=True).token


def test_the_url_carries_the_token_so_it_can_be_opened_directly():
    binding = serve.resolve(lan=True, token="abc")
    assert binding.url().endswith("?token=abc")


def test_a_lan_binding_shows_a_reachable_address_not_the_wildcard():
    """0.0.0.0 is not something anyone can type into a phone."""
    assert serve.resolve(lan=True).display_host != "0.0.0.0"  # noqa: S104


def test_the_address_lookup_falls_back_rather_than_raising(monkeypatch):
    """Offline, the probe socket fails; the server is still listening."""
    def boom(*_a, **_kw):
        raise OSError("no network")

    monkeypatch.setattr(serve.socket, "socket", boom)
    assert serve.lan_address() == "localhost"
    assert serve.resolve(lan=True).display_host == "localhost"
