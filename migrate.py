"""Prepare the shared SQLite database before bot and web start."""
from __future__ import annotations

from llmcord_core.config import resolve_database_path
from llmcord_core.store import Store


def main() -> None:
    Store(resolve_database_path()).close()


if __name__ == "__main__":
    main()
