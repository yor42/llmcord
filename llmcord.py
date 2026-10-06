"""Run the llmcord Discord bot with config.yaml and environment credentials."""

import logging

from llmcord_core.config import load_settings
from llmcord_core.discord_bot import SkitBot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    SkitBot(settings).run(settings.token, log_handler=None)


if __name__ == "__main__":
    main()
