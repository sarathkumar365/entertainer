# The agent

Part two of the platform: turn *what to watch* into *what is ready to watch*. The specification
is in the top-level README under "Part two — the agent"; backlog item 14 tracks it.

This folder holds the agent's design, one document per piece, written as each piece is
decided or proven.

| Document | Holds | State |
| --- | --- | --- |
| [DESIGN.md](DESIGN.md) | Agent-wide decisions, phase by phase | Settled for the manual piece, 29 Sep 2026 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the manual piece is built: packages, threads, session ownership, jobs, phases | Decided 29 Sep 2026 |
| [ACQUISITION.md](ACQUISITION.md) | Getting a judged title onto disk: the Telegram route, how it works, what the spike found | Spike done, 29 Sep 2026 |

The manual piece — the user asks, the agent gets the film into Jellyfin — is designed. Still
to design, as their own pieces: the autonomous watcher (and the engine threshold it acts on),
the disk budget and eviction rule, and turning watch outcomes back into verdicts.
