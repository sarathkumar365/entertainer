"""Deciding how and where to serve the rating interface.

The binding rules, the token requirement and the LAN-address discovery lived
in the `ent rate` command body. They are security decisions, not argument
parsing: whether a token is mandatory depends on which interface is being
bound, and getting that wrong hands somebody else write access to a taste
profile.

Nothing here imports typer or touches a console. `Binding` is a decision, and
the caller reports it.
"""

from __future__ import annotations

import secrets
import socket
from dataclasses import dataclass

LOOPBACK = ("127.0.0.1", "localhost", "::1")
ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - deliberate, and gated behind a token

#: Long enough that guessing is hopeless, short enough to retype from a phone.
TOKEN_BYTES = 12


@dataclass
class Binding:
    """Where the server will listen, and on what terms."""

    host: str
    port: int
    token: str | None
    off_loopback: bool
    display_host: str

    def url(self) -> str:
        url = f"http://{self.display_host}:{self.port}"
        return f"{url}?token={self.token}" if self.token else url


def resolve(
    host: str = "127.0.0.1",
    port: int = 8756,
    lan: bool = False,
    token: str = "",
) -> Binding:
    """Work out the binding, minting a token when one is required.

    A token is mandatory off the loopback interface and cannot be opted out
    of. The page writes to the verdict log, so an unauthenticated copy on a
    shared network is somebody else's write access to your taste profile —
    which is why this mints one rather than warning and continuing.
    """
    if lan:
        host = ALL_INTERFACES
    off_loopback = host not in LOOPBACK
    if off_loopback and not token:
        token = secrets.token_urlsafe(TOKEN_BYTES)
    return Binding(
        host=host,
        port=port,
        token=token or None,
        off_loopback=off_loopback,
        display_host=lan_address() if host == ALL_INTERFACES else host,
    )


def lan_address() -> str:
    """The address another device on this network would use.

    Found by opening a UDP socket toward a public address and reading back
    which local interface the routing table chose. No packet is sent — UDP
    connect only fixes the peer — so this works offline and costs nothing.
    Falls back to "localhost", which is wrong for the stated purpose but
    harmless: the server is still listening on every interface.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        return "localhost"
