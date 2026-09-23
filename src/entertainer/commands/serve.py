"""Opening the browser interfaces."""

from __future__ import annotations

import typer
from rich.panel import Panel

from ..config import has_tmdb
from ..errors import EntertainerError
from ..render import console
from ..render import fail as _fail
from ._apps import app


@app.command()
def rate(
    port: int = typer.Option(8756, help="Port to serve on."),
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    lan: bool = typer.Option(
        False, help="Bind to all interfaces so another device on the network can reach it."
    ),
    token: str = typer.Option("", help="Shared secret. Generated automatically when --lan."),
    live: bool = typer.Option(
        None,
        "--live/--catalogue",
        help="Serve titles from TMDB instead of a local catalogue. "
        "Chosen automatically when there is no catalogue.",
    ),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Open the rating interface — a grid of posters you click through.

    The engine learns from verdicts and nothing else, so how fast you can give
    verdicts is the rate-limiting step in making it good. Recognising a poster
    is far quicker than recalling and typing a transliterated title, and the
    friction compounds over the hundred-odd ratings the model needs.

    Everything written here goes into the same event log the CLI uses, so
    `ent recs`, `ent taste` and `ent audit` see it immediately.
    """
    try:
        import uvicorn
    except ImportError:
        _fail("install the web extra: uv pip install -e '.[web]'")

    from ..web import serve
    from ..web.app import EMPTY_CATALOGUE, _catalogue_size, create_app

    # A running build owns the database, so this is the normal way to find
    # out — and a lock error here must read as "a build is running", not as
    # an unhandled traceback on startup.
    try:
        size = _catalogue_size()
    except EntertainerError as exc:
        _fail(str(exc))
    use_live = live if live is not None else size < EMPTY_CATALOGUE
    if use_live and not has_tmdb():
        _fail(
            "no catalogue on this machine and no TMDB credentials.\n"
            "Either put TMDB_BEARER in .env, or import a bundle with "
            "`ent bundle import <file>`."
        )
    if not use_live and size < EMPTY_CATALOGUE:
        _fail("no catalogue — run `ent setup`, `ent bundle import <file>`, or use --live")

    binding = serve.resolve(host=host, port=port, lan=lan, token=token)
    # Built before the panel is printed: a failure here must not follow a
    # banner announcing a URL that will never answer.
    try:
        application = create_app(token=binding.token, live=use_live)
    except EntertainerError as exc:
        _fail(str(exc))
    off_loopback = binding.off_loopback
    url = binding.url()
    console.print(
        Panel.fit(
            f"[bold]{url}[/bold]\n\n"
            "[bold]♥[/bold] loved   [bold]+[/bold] liked   [bold]~[/bold] fine   "
            "[bold]−[/bold] disliked   [bold]?[/bold] not seen\n"
            + (
                "[yellow]live mode — titles come from TMDB, no local catalogue[/yellow]\n"
                if use_live
                else f"[dim]{size:,} titles in the local catalogue[/dim]\n"
            )
            + "search finds anything TMDB knows, even outside the catalogue\n\n"
            + (
                "[yellow]reachable on your local network — the link above "
                "contains its access token[/yellow]\n"
                if off_loopback
                else ""
            )
            + "[dim]ctrl-c to stop · nothing leaves this machine[/dim]",
            title="rate what you have seen",
            border_style="cyan",
        )
    )
    if open_browser and not off_loopback:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        application, host=binding.host, port=binding.port, log_level="warning",
    )


@app.command()
def studio(
    port: int = typer.Option(8757, help="Port to serve on."),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Open the read-only Build Studio, then run ``ent setup`` separately."""
    try:
        import uvicorn
    except ImportError:
        _fail("install the web extra: uv pip install -e '.[web]'")
    from ..web.studio import create_studio_app

    url = f"http://127.0.0.1:{port}"
    console.print(Panel.fit(
        f"[bold]{url}[/bold]\n\nOpen this page, then run [cyan]ent setup[/cyan] in another terminal.\n"
        "[dim]Studio observes saved local build progress; it cannot alter the build.[/dim]",
        title="Build Studio", border_style="cyan",
    ))
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_studio_app(), host="127.0.0.1", port=port, log_level="warning")
