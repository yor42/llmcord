"""Run the private administration service behind Tailscale Serve."""
from __future__ import annotations

import os

import uvicorn

from llmcord_core.config import operator_ids_from_env, resolve_database_path
from llmcord_core.web import create_app


def main() -> None:
    base_url = os.environ.get("WEB_BASE_URL", "")
    client_id = os.environ.get("DISCORD_CLIENT_ID", "")
    client_secret = os.environ.get("DISCORD_CLIENT_SECRET", "")
    bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
    database_path = resolve_database_path()
    app = create_app(database_path, base_url, client_id, client_secret, bot_token, operator_ids=operator_ids_from_env())
    # Keep-alive must outlast Tailscale Serve's idle upstream connections (Go's default is 90 s), so the proxy never reuses a socket uvicorn has just closed (MNT-13).
    uvicorn.run(app, host=os.environ.get("WEB_HOST", "127.0.0.1"),
                port=int(os.environ.get("WEB_PORT", "8080")), access_log=False,
                timeout_keep_alive=int(os.environ.get("WEB_KEEP_ALIVE", "95")))


if __name__ == "__main__":
    main()
