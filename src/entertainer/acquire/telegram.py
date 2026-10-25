"""Telegram channels as a source, through a user-account client.

The Bot API cannot do this job: a bot cannot search a channel's history and
cannot download files over 20 MB. MTProto as a user account (Telethon) can do
both, and is what the "paste a name, get a file" bots are themselves built on —
they index channel posts and resend by ``file_id``. Searching the channels
directly uses the same data without depending on a bot's UI.

Channels searched come from ``ENTERTAINER_TG_CHANNELS`` and request groups
posted in from ``ENTERTAINER_TG_GROUPS``, and nowhere else. Which ones are
acceptable is the user's decision, made in configuration.

Login is interactive once (phone number and code, typed by the person), after
which the session file is reused. Every source in a job shares one client
behind one lock (``connect``): two clients on one session file is how a session
gets revoked.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from ..config import AcquireSettings
from ..errors import AcquireBusy, MissingCredentials, MissingDependency
from .match import VIDEO_EXTS, Candidate, Target


@contextmanager
def session_lock(settings: AcquireSettings) -> Iterator[None]:
    """Hold the Telegram session for this process, or refuse at once.

    Held for one job, not for the life of the app, so the CLI can still
    acquire whenever the web app is idle. ``flock`` is released by the kernel
    if the process dies, so a crash never leaves a stale lock behind.
    """
    settings.tg_lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(settings.tg_lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AcquireBusy(
                "Another process is using the Telegram session, usually the app running a job. "
                "Wait for it to finish, or use the web page."
            ) from None
        yield
    finally:
        os.close(fd)


def client(settings: AcquireSettings):
    try:
        from telethon import TelegramClient
    except ImportError as e:
        raise MissingDependency("Install the acquire extra: uv pip install -e '.[acquire]'") from e
    if not (settings.tg_api_id and settings.tg_api_hash):
        raise MissingCredentials(
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env (from my.telegram.org)."
        )
    settings.tg_session.parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(settings.tg_session), int(settings.tg_api_id), settings.tg_api_hash)


@asynccontextmanager
async def connect(settings: AcquireSettings) -> AsyncIterator:
    """One locked, logged-in client, shared by every Telegram source in a job."""
    with session_lock(settings):
        c = client(settings)
        await c.connect()
        try:
            if not await c.is_user_authorized():
                raise MissingCredentials("Telegram is not logged in. Run: ent acquire login")
            yield c
        finally:
            await c.disconnect()
            # The session holds full control of the account.
            for f in settings.tg_session.parent.glob(settings.tg_session.name + "*"):
                f.chmod(0o600)


async def leave(settings: AcquireSettings, chats: list[str]) -> tuple[list[str], dict[str, str]]:
    """Leave the channels the agent joined. Returns those left and why any were not.

    Only ever called with chat ids from the join log, so a channel the user
    joined themselves is never touched.
    """
    left: list[str] = []
    failed: dict[str, str] = {}
    async with connect(settings) as c:
        for chat in chats:
            try:
                await c.delete_dialog(await c.get_entity(int(chat)))
                left.append(chat)
            except Exception as e:  # noqa: BLE001 — one stubborn channel must not keep the rest
                failed[chat] = f"{type(e).__name__}: {e}"
    return left, failed


class TelegramSource:
    name = "telegram"

    def __init__(self, client, chans: tuple[str, ...] | list[str], per_query: int = 40):
        self._client = client
        self.chans = list(chans)
        self.per_query = per_query

    async def __aenter__(self):
        self.chans = await self._resolve(self.chans)
        return self

    async def __aexit__(self, *exc):
        pass  # the client belongs to whoever connected it

    async def _resolve(self, names: list[str]) -> list:
        """Accept @usernames, t.me links, or a channel's display title.

        Many of these channels are private or have no username, so the title
        as it appears in the app is often the only name a person knows. Titles
        are matched against the dialogs this account has joined.
        """
        by_title = [n for n in names if not n.startswith(("@", "http", "t.me", "-100"))]
        found = {}
        if by_title:
            async for d in self._client.iter_dialogs():
                if d.is_channel and d.name in by_title:
                    found[d.name] = d.entity
        missing = [n for n in by_title if n not in found]
        if missing:
            raise MissingCredentials(
                f"This account has not joined {', '.join(missing)}. Join it in Telegram, "
                "or name it by @username or t.me link."
            )
        return [found.get(n, n) for n in names]

    async def search(self, target: Target) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[tuple[int, int]] = set()
        for query in _queries(target):
            for chan in self.chans:
                async for msg in self._client.iter_messages(
                    chan, search=query, limit=self.per_query
                ):
                    key = (msg.chat_id, msg.id)
                    if key in seen or not msg.file:
                        continue
                    seen.add(key)
                    if (c := _candidate(msg)) is not None:
                        out.append(c)
        return out

    async def fetch(self, cand: Candidate, dest: Path, progress: Callable | None = None) -> Path:
        await self._client.download_media(
            cand.ref,
            file=str(dest),
            progress_callback=(lambda d, t: progress(d, t)) if progress else None,
        )
        return dest


def _queries(target: Target) -> list[str]:
    qs = [target.title, *target.alt_titles]
    return list(dict.fromkeys(q for q in qs if q))


def _candidate(msg) -> Candidate | None:
    f = msg.file
    name = f.name or ""
    ext = _ext(f)
    is_video = (f.mime_type or "").startswith("video/") or ext in VIDEO_EXTS
    if not is_video:
        return None
    # Channels often post files named "@channel_1234.mkv" and put the real
    # title in the caption, so the caption is the fallback label.
    caption = (msg.message or "").splitlines()[0] if msg.message else ""
    label = name if name and not name.startswith("@") else (caption or name)
    return Candidate(source="telegram", label=label, size=f.size, ref=msg, ext=ext)


class BotGroupSource:
    """A request group run by auto-filter bots, driven the way a person would.

    Measured against a real group on 29 September 2026. The group holds no
    files: a member posts a title, a filter bot replies to that message with
    one callback button per file (``file#<file_id>#<user_id>``, only pressable
    by the requester), pressing one returns a ``t.me/<bot>?start=...`` link,
    and ``/start <param>`` in the bot's private chat makes it send the file.
    Before sending, the bot usually demands a join of another channel; the
    join (or join request) is made and ``/start`` sent again.

    Only three kinds of button are ever pressed: a file button on the reply to
    our own query, a next-page button there, and a ``t.me`` join link the
    file bot sends in private. Everything else — the 18+ ad buttons these
    groups carry, shorteners, external sites — is ignored.
    """

    name = "telegram-bot"
    _SIZE = {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}

    def __init__(self, client, groups: tuple[str, ...] | list[str], pages: int = 2,
                 wait: float = 30.0, on_join=None):
        self._client = client
        self.inner = TelegramSource(client, groups)
        self.pages, self.wait = pages, wait
        # Every join is recorded against its job so "leave joined channels" can undo it.
        self.on_join = on_join

    async def __aenter__(self):
        await self.inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        pass

    async def search(self, target: Target) -> list[Candidate]:
        out: list[Candidate] = []
        queries = [f"{target.title} {target.year or ''} {target.language or ''}".strip(),
                   f"{target.title} {target.year or ''}".strip()]
        for group in self.inner.chans:
            for q in dict.fromkeys(" ".join(q.split()) for q in queries):
                out.extend(await self._query(group, q))
                if any(c.label for c in out):
                    break
        return out

    async def _query(self, group, query: str) -> list[Candidate]:
        import asyncio

        from telethon import functions

        sent = await self._client.send_message(group, query)
        reply = await self._await(lambda: self._reply_to(group, sent.id))
        if reply is None:
            return []
        out = _file_buttons(group, reply)
        for _ in range(self.pages - 1):
            nxt = _next_page(reply)
            if nxt is None:
                break
            await self._client(functions.messages.GetBotCallbackAnswerRequest(
                peer=group, msg_id=reply.id, data=nxt))
            await asyncio.sleep(2)
            reply = await self._client.get_messages(group, ids=reply.id)
            out.extend(_file_buttons(group, reply))
        # A next-page press that the bot ignores leaves page one in place.
        return list({c.ref[2]: c for c in out}.values())

    async def _reply_to(self, group, msg_id: int):
        async for m in self._client.iter_messages(group, min_id=msg_id, limit=30):
            if m.reply_to_msg_id == msg_id and _file_buttons(group, m):
                return m
        return None

    async def fetch(self, cand: Candidate, dest: Path, progress: Callable | None = None) -> Path:


        group, msg_id, data, bot = cand.ref
        bot, param = await self._start_link(group, msg_id, data, bot)
        for _ in range(3):
            before = await self._last_id(bot)
            await self._client.send_message(bot, f"/start {param}")
            msg = await self._await(lambda b=before: self._delivery(bot, b))
            if msg is None:
                raise RuntimeError(f"@{bot} sent nothing within {self.wait:.0f}s")
            if msg.file:
                # The button label often carries no extension; the delivered
                # file does, and it decides what the library copy is called.
                cand.ext = _ext(msg.file) or cand.ext
                await self._client.download_media(
                    msg, file=str(dest),
                    progress_callback=(lambda d, t: progress(d, t)) if progress else None)
                return dest
            links = _join_links(msg)
            if not links:
                raise RuntimeError(f"@{bot} replied without a file: {(msg.message or '')[:120]}")
            for link in links:
                await self._join(link)
        raise RuntimeError(f"@{bot} kept asking for joins; gave up after three rounds")

    async def _start_link(
        self, group, msg_id: int, data: bytes, bot: str | None
    ) -> tuple[str, str]:
        """Where the file is delivered: ``(bot username, /start parameter)``.

        Pressing the button returns ``t.me/<bot>?start=<file_id>_<user_id>``,
        which is the button data with ``#`` turned into ``_``. Deriving it
        avoids the press, which fails with DATA_INVALID once the reply has
        been paged or auto-deleted — both routine in these groups.
        """
        from telethon import errors, functions

        _, file_id, user_id = data.decode().split("#", 2)
        if bot:
            return bot, f"{file_id}_{user_id}"
        try:
            ans = await self._client(functions.messages.GetBotCallbackAnswerRequest(
                peer=group, msg_id=msg_id, data=data))
        except errors.RPCError as e:
            raise RuntimeError(f"file button no longer answers ({e})") from e
        m = re.match(r"https://t\.me/(\w+)\?start=(\S+)", ans.url or "")
        if not m:
            raise RuntimeError(f"bot answered without a start link: {ans.message or ans.url}")
        return m.group(1), m.group(2)

    async def _delivery(self, bot: str, after: int):
        """The file the bot sent after ``after``, else its join demand.

        Read oldest first, and a file always wins: the bot measured here
        sends the file and then, a second later, a fresh join demand for a
        different channel. Newest first saw the demand, joined, asked again,
        and never looked at the file already sitting above it.
        """
        gate = None
        async for m in self._client.iter_messages(bot, min_id=after, reverse=True):
            if m.out:
                continue
            if _is_video(m):
                return m
            if gate is None and _join_links(m):
                gate = m
        return gate

    async def _join(self, link: str) -> None:
        from telethon import errors, functions

        m = re.match(r"https?://t\.me/(?:\+|joinchat/)([\w-]+)", link)
        chat = ""
        try:
            if m:
                res = await self._client(functions.messages.ImportChatInviteRequest(m.group(1)))
            else:
                res = await self._client(
                    functions.channels.JoinChannelRequest(link.rsplit("/", 1)[-1]))
            chat = _joined_chat(res)
        except errors.UserAlreadyParticipantError:
            return  # joined before, by the user or an earlier job; not ours to leave
        except errors.InviteRequestSentError:
            pass  # a pending request is what these bots check for
        except errors.FloodWaitError as e:
            if e.seconds > 600:
                raise
            import asyncio

            await asyncio.sleep(e.seconds + 1)
            await self._join(link)
            return
        if self.on_join:
            self.on_join(link, chat)

    async def _last_id(self, bot: str) -> int:
        last = await self._client.get_messages(bot, limit=1)
        return last[0].id if last else 0

    async def _await(self, probe):
        import asyncio

        loop = asyncio.get_running_loop()
        end = loop.time() + self.wait
        while loop.time() < end:
            await asyncio.sleep(2)
            if (got := await probe()) is not None:
                return got
        return None


import re  # noqa: E402 — module-level helpers below


def _joined_chat(res) -> str:
    """The id of the channel a join opened, as ``leave`` needs it; empty if unknown."""
    chats = getattr(res, "chats", None) or []
    return str(chats[0].id) if chats else ""


def _buttons(msg):
    rows = getattr(getattr(msg, "reply_markup", None), "rows", None) or []
    for r in rows:
        for b in r.buttons:
            yield b.text or "", b.to_dict().get("type") or b.to_dict()


def _file_buttons(group, msg) -> list[Candidate]:
    # Taken now: the group auto-deletes the reply, and with it the sender.
    bot = getattr(getattr(msg, "sender", None), "username", None)
    out = []
    for text, t in _buttons(msg):
        data = t.get("data") or b""
        if not data.startswith(b"file#"):
            continue
        size, _, label = text.strip().partition("◾")
        out.append(Candidate(source="telegram-bot", label=label.strip() or text.strip(),
                             size=_size(size), ref=(group, msg.id, data, bot)))
    return out


def _next_page(msg) -> bytes | None:
    for _, t in _buttons(msg):
        data = t.get("data") or b""
        if data.startswith(b"nxt_pg"):
            return data
    return None


def _ext(f) -> str:
    ext = (f.ext or "").lstrip(".").lower()
    if not ext and "." in (f.name or ""):
        ext = f.name.rsplit(".", 1)[-1].lower()
    return ext


def _is_video(msg) -> bool:
    f = getattr(msg, "file", None)
    if not f:
        return False
    return (f.mime_type or "").startswith("video/") or _ext(f) in VIDEO_EXTS


def _join_links(msg) -> list[str]:
    """Join links on a join demand. Never on a file message: those carry
    promotional channel buttons that are not a condition of anything."""
    if getattr(msg, "file", None) or "join" not in (getattr(msg, "message", "") or "").lower():
        return []
    return [t["url"] for _, t in _buttons(msg)
            if re.match(r"https?://t\.me/", t.get("url") or "") and "?start=" not in t["url"]]


def _size(s: str) -> int | None:
    m = re.match(r"\s*([\d.]+)\s*(KB|MB|GB)", s)
    return int(float(m.group(1)) * BotGroupSource._SIZE[m.group(2)]) if m else None
