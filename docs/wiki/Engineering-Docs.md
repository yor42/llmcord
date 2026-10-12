Working notes for contributors stay in the repository next to the code, under [`docs/engineering/`](https://github.com/yor42/llmcord/tree/main/docs/engineering). They change with almost every item, so they are not copied here.

| Document | Contents |
| --- | --- |
| [Backlog](https://github.com/yor42/llmcord/blob/main/docs/engineering/backlog.md) | Current maintenance (`MNT-*`), UI polish (`UI-*`) and feature (`FEAT-*`) items, and the design decisions from D13 on |
| [System map](https://github.com/yor42/llmcord/blob/main/docs/engineering/system-map.md) | Processes, component diagram, turn and dashboard-auth sequences, shared state, external dependencies |
| [UI guide](https://github.com/yor42/llmcord/blob/main/docs/engineering/ui-guide.md) | Conventions for dashboard and Discord-reply wording and layout |
| [Claude Code workflow](https://github.com/yor42/llmcord/blob/main/docs/engineering/claude-workflow.md) | How changes are made with Claude Code agents: model routing, delegation and verification |
| [Dashboard performance baseline](https://github.com/yor42/llmcord/blob/main/docs/engineering/perf-baseline.md) | Benchmark method and recorded timings |
| [Channels plan](https://github.com/yor42/llmcord/blob/main/docs/engineering/channels-plan.md), [Items plan](https://github.com/yor42/llmcord/blob/main/docs/engineering/items-plan.md), [Minigames plan](https://github.com/yor42/llmcord/blob/main/docs/engineering/minigames-plan.md) | Design plans for those features |
| [Research](https://github.com/yor42/llmcord/tree/main/docs/engineering/research) | Spoken-move prototype, model choices for reading game moves, local model benchmarks |
| [History](https://github.com/yor42/llmcord/tree/main/docs/engineering/history) | The closed R1–R6 hardening rework: audit findings (`BUG-*`, `PERF-*`, `SEC-*`) that test docstrings cite, the roadmap and decisions D1–D12, and the first verification baseline |

The repository's [`CLAUDE.md`](https://github.com/yor42/llmcord/blob/main/CLAUDE.md) has the module map, the commands, and the invariants every change must keep (server isolation, lore isolation, memory consent, branches, preset and model snapshots).
