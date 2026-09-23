"""Referring to a recommendation by position.

`ent loved 3` means "the third thing you just showed me". The writer (recs)
and the readers (the verdict commands) sat 800 lines apart in cli.py with
nothing connecting them but a bare "last_slate" string, and the lookup query
was a verbatim copy of resolve._SELECT.
"""

from __future__ import annotations

from test_cli import app_env  # noqa: F401

from entertainer import resolve, store


def ids(con, n=3):
    return [
        int(r[0])
        for r in con.execute("SELECT item_id FROM titles ORDER BY item_id LIMIT ?", [n]).fetchall()
    ]


def test_a_position_resolves_to_the_title_at_that_index(app_env):  # noqa: F811
    with store.session() as con:
        items = ids(con)
        store.set_last_slate(con, items)
        assert resolve.from_slate(con, "1").item_id == items[0]
        assert resolve.from_slate(con, "3").item_id == items[2]


def test_positions_are_one_based_because_that_is_how_slates_print(app_env):  # noqa: F811
    with store.session() as con:
        store.set_last_slate(con, ids(con))
        assert resolve.from_slate(con, "0") is None


def test_a_position_past_the_end_falls_through_rather_than_raising(app_env):  # noqa: F811
    """It must be treated as a title instead — a film may be called "12"."""
    with store.session() as con:
        store.set_last_slate(con, ids(con))
        assert resolve.from_slate(con, "99") is None


def test_a_non_numeric_query_is_not_a_position(app_env):  # noqa: F811
    with store.session() as con:
        store.set_last_slate(con, ids(con))
        assert resolve.from_slate(con, "Parasite") is None


def test_no_slate_yet_is_not_an_error(app_env):  # noqa: F811
    with store.session() as con:
        assert store.last_slate(con) == []
        assert resolve.from_slate(con, "1") is None


def test_surrounding_whitespace_is_tolerated(app_env):  # noqa: F811
    with store.session() as con:
        items = ids(con)
        store.set_last_slate(con, items)
        assert resolve.from_slate(con, "  2  ").item_id == items[1]


def test_by_item_id_returns_the_same_shape_as_every_other_lookup(app_env):  # noqa: F811
    """The CLI kept its own copy of the SELECT, so a new Match field meant
    editing a query in another file 1,200 lines away."""
    with store.session() as con:
        item_id = ids(con, 1)[0]
        direct = resolve.by_item_id(con, item_id)
        hit = next(m for m in resolve.search(con, direct.title, limit=20) if m.item_id == item_id)
        assert (direct.title, direct.year, direct.kind, direct.language) == (
            hit.title, hit.year, hit.kind, hit.language,
        )


def test_by_item_id_returns_none_for_an_unknown_id(app_env):  # noqa: F811
    with store.session() as con:
        assert resolve.by_item_id(con, 10**9) is None
