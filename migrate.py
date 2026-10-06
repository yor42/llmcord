"""Prepare the shared SQLite database before bot and web start."""
from __future__ import annotations

import os

from llmcord_core.store import Store


def main() -> None:
    database = os.environ.get("LLMCORD_DATABASE_PATH", "data/llmcord.sqlite3")
    Store(database).close()


if __name__ == "__main__":
    main()
