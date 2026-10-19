# Acquisition

How the agent gets a title it has decided on onto disk, in the shape Jellyfin expects.
Spiked on 29 September 2026 on branch `spike/acquire`; code in `src/entertainer/acquire/`.

**Result: the Telegram route works end to end.** Given only `"Vikram" --year 2022`, one
command resolved the film, found it through a Telegram request group, downloaded it, verified
it and filed it:

```
data/library/Movies/Vikram (2022)/Vikram (2022).mkv
903 MB · HEVC · Tamil audio · English subtitles · 171 min against TMDB's 174
```

```bash
ent acquire login                                    # once, interactive
ent acquire get "Vikram 2022" --dry-run
ent acquire get "Vikram 2022" --sources telegram-bot
ent acquire get --imdb <tt-id>                       # a film the catalogue lacks
```

---

## The decision

The source is Telegram. That is the user's decision, recorded here because the README asks
for acquisition to be decided explicitly rather than inherited. The user has confirmed the
channels used serve copies they are entitled to hold. The code encodes that as configuration:
nothing is searched that is not named in `ENTERTAINER_TG_CHANNELS`.

Internet Archive is kept as a second source. It is small (mostly pre-1970, public domain) but
has no licensing question and no account risk, and it is what the pipeline was first verified
against.

---

## How the Telegram route works

### The client

A **user-account** client (Telethon, MTProto), not the Bot API. The Bot API cannot search a
channel's history and cannot download files over 20 MB, which rules it out for films.

- `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` come from my.telegram.org, into `.env`.
- Login is interactive once: phone number **with country code**, then the code Telegram sends.
  The session is stored under `data/telegram/`, which is gitignored.
- Use a secondary account. Automated user clients can be rate-limited or flagged, and this
  one posts in public groups and joins channels.

### The group is not a file channel

The spike assumed the named channel held files and could be searched. It does not. Cinemas
Terminal is a **request group run by auto-filter bots**: 13 bots are members, and no film is
ever posted in it. The flow, measured:

1. A member posts a title in the group (`Vikram 2022 Tamil`).
2. A filter bot (`@Lt_Moviebot` in the test) replies to that message with one inline button
   per file, labelled `<size> ◾ <file name>`, plus paging buttons. The reply is auto-deleted
   after a while.
3. Each file button carries callback data `file#<file_id>#<user_id>`. Only the member who
   asked can press it.
4. Pressing it answers with a URL: `https://t.me/<bot>?start=<file_id>_<user_id>` — the button
   data with `#` turned into `_`.
5. `/start <param>` in the bot's private chat makes the bot send the file.
6. Before sending, the bot usually demands a join of another channel ("Please join below
   channel to get file!", a `t.me/+<hash>` button). A join *request* is enough; the bot then
   delivers on the next `/start`.
7. The bot sends the file and a note that it will be deleted in five minutes.

The bot does not answer queries sent to it privately — `/start` there only returns a welcome.
The query has to go to the group, under the user's name.

### What the adapter does

`BotGroupSource` in `acquire/telegram.py`:

- Posts `<title> <year> <language>`, then `<title> <year>` if the first finds nothing.
- Waits for the bot's reply to *that* message, reads the file buttons, and records the bot's
  username at once, because the reply may be gone by the time a file is fetched.
- Builds the `/start` parameter straight from the button data instead of pressing the button.
  Pressing fails with `DATA_INVALID` once the reply has been paged or auto-deleted, both of
  which are routine.
- Sends `/start`, then reads the bot's private chat **oldest first**. A file always wins. If
  there is no file but there is a join demand, it joins and asks again, up to three rounds.
- Downloads straight to a staging directory.

### Matching and verification

Shared with every source, in `acquire/match.py` and `acquire/core.py`:

- The title is resolved through TMDB first, so every source is matched against the canonical
  title, year, runtime, original title and original language rather than what was typed.
- File names are parsed with `guessit`. Title similarity must be at least 0.85 and the year
  within one. Samples, trailers, episodes and non-video files are refused.
- Ranking: original language present (a Tamil film's Hindi dub matches on title and year and
  is still the wrong file), then 1080p over 2160p and 720p, then HEVC. Cam and telesync are
  marked down hard.
- Before filing, `ffprobe` must find a video stream and a duration within `max(8 min, 20%)` of
  the TMDB runtime. A file that fails is deleted and the next candidate is tried, three at most.
- Free disk is checked before each download against the file size plus a reserve (5 GB).
- The file lands as `Movies/<Title> (<Year>)/<Title> (<Year>).<ext>`, and a Jellyfin library
  refresh is requested if `JELLYFIN_URL` and `JELLYFIN_API_KEY` are set.

---

## What the spike got wrong first

Each of these was hit against the real group and is fixed in the code.

- **Searching the channel found nothing.** It is a request group, not a file channel. The
  answer was to drive the bot, not to search harder.
- **Pressing a file button failed with `DATA_INVALID`.** Fetching a second page of results
  edits the reply in place, so page-one buttons no longer exist on it. Deriving the start
  link from the button data removed the press entirely.
- **The file arrived and was ignored.** The bot sends the file and then, a second later, a
  fresh join demand. Reading newest first saw the demand, joined, asked again, and never
  looked at the file above it. Read oldest first; a file wins.
- **Promotional buttons were followed as join demands.** File messages carry buttons to the
  bot's other channels. Before the fix these were joined, which is how the test account ended
  up in several ad and trading channels. Join links are now taken only from a message that
  has no file and says "join".
- **Joining too fast is rate-limited.** Telegram returned a wait of about 280 seconds on
  `ImportChatInviteRequest`. Waits up to ten minutes are now slept through.
- **A bare local phone number fails login** with `PhoneNumberInvalidError`. The prompt now
  refuses anything not starting with `+`.

---

## Measured

| Measure | Value |
| --- | --- |
| Results for `Vikram 2022` in the group | 297 across 33 pages, most of them other films and dubs |
| Results for `Vikram 2022 Tamil` | 99; 9 of 18 on the first two pages matched |
| Download speed, user account | about 1.5 MB/s steady |
| Time for a 903 MB file | about 610 s |
| Bot's deletion window for the sent file | 5 minutes — a download that has started completes regardless |

At 1.5 MB/s a 2 GB 1080p file takes over 20 minutes. That is fine for an agent that acts
ahead of time, and a reason not to make acquisition interactive.

---

## Open

- **Tie-breaking.** A dual-audio "Hindi or Tamil" file scored the same as Tamil-only files and
  was picked. Ties should prefer a file carrying only the original language.
- **One bot layout is parsed.** `file#` and `nxt_pg` are this bot family's format. Another
  group's bot needs its own parser; a registry of layouts per group is the likely shape.
- **The query is public.** It is posted in the group under the account's name. Another reason
  for a secondary account.
- **Channel clean-up.** The account joined several promotional channels during the spike and
  should leave them.
- **Disk.** The development machine had about 17 GB free. The disk budget and eviction rule
  are the next thing that bites.
- ~~**Not yet a real command.**~~ `ent acquire` replaced `python -m entertainer.acquire` in
  phase 1 of `ARCHITECTURE.md`; the web route is phase 4.

## Telegram MCP servers

Checked on 29 September 2026. There is no official Telegram MCP server and none in the
connector registry. Community servers exist — `mcp-telegram/mcp-telegram`,
`resccrew/telegram-mcp`, `Telegram-mcp/telegram-mcp`, `sparfenyuk/mcp-telegram` — all built on
Telethon as a user account, several able to press inline buttons and download media.

Not used for the pipeline. An MCP server lets a model drive Telegram step by step, which turns
every acquisition into a chain of model calls, and a gigabyte download through tool calls is
awkward where Telethon streams to disk. Each such server also holds the full account session,
so a third-party one has to be audited, and a hosted one should not be used at all. The one
place it could help is exploring a new group's bot by hand before writing its parser.
