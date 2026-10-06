"""Run the private administration service behind Tailscale Serve."""
from __future__ import annotations

import os

import uvicorn

from llmcord_core.web import create_app


def main() -> None:
    base_url = os.environ.get("WEB_BASE_URL", "")
    client_id = os.environ.get("DISCORD_CLIENT_ID", "")
    client_secret = os.environ.get("DISCORD_CLIENT_SECRET", "")
    bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
    database_path = os.environ.get("LLMCORD_DATABASE_PATH", "data/llmcord.sqlite3")
    app = create_app(database_path, base_url, client_id, client_secret, bot_token)
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)


if __name__ == "__main__":
    main()
