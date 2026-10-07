"""BUG-05: database path resolution across migrate.py, web_main.py and the bot.

Every test runs with cwd set to a fresh temp dir holding its own (optional)
config.yaml. ``migrate.Store`` and ``web_main.create_app`` / ``uvicorn.run``
are patched, so no database file or server is ever created.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import migrate
import web_main
from llmcord_core.config import load_settings, resolve_database_path

DEFAULT_PATH = "data/llmcord.sqlite3"
ENV_PATH = "env/override.sqlite3"
YAML_PATH = "custom/db.sqlite3"


class DatabasePathTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        old_cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, old_cwd)

    def write_config(self, text: str) -> None:
        (self.dir / "config.yaml").write_text(text, encoding="utf-8")

    def env(self, database_path: str | None, **extra: str):
        """Patch os.environ, removing LLMCORD_DATABASE_PATH unless given."""
        patcher = patch.dict(os.environ, extra)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("LLMCORD_DATABASE_PATH", None)
        if database_path is not None:
            os.environ["LLMCORD_DATABASE_PATH"] = database_path

    def migrate_path(self) -> str:
        with patch.object(migrate, "Store") as store:
            migrate.main()
        store.assert_called_once()
        store.return_value.close.assert_called_once()
        return str(store.call_args.args[0])

    def web_path(self) -> str:
        app = MagicMock(name="app")
        with patch.object(web_main, "create_app", return_value=app) as create_app, \
                patch.object(web_main.uvicorn, "run") as run:
            web_main.main()
        create_app.assert_called_once()
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], app)
        return str(create_app.call_args.args[0])

    def assert_no_files_created(self) -> None:
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            ["config.yaml"] if (self.dir / "config.yaml").exists() else [],
        )

    # --- migrate.main() ---------------------------------------------------

    def test_migrate_uses_env_path_over_config(self) -> None:
        """Characterization (BUG-05): migrate uses LLMCORD_DATABASE_PATH even when config.yaml sets database_path."""
        self.write_config(f"database_path: {YAML_PATH}\n")
        self.env(ENV_PATH)
        self.assertEqual(self.migrate_path(), ENV_PATH)
        self.assert_no_files_created()

    def test_migrate_defaults_without_env_or_config(self) -> None:
        """Characterization (BUG-05): with no env var and no config.yaml, migrate uses data/llmcord.sqlite3."""
        self.env(None)
        self.assertEqual(self.migrate_path(), DEFAULT_PATH)
        self.assert_no_files_created()

    def test_migrate_defaults_when_config_lacks_key(self) -> None:
        """Characterization (BUG-05): a config.yaml without database_path leaves migrate on the default path."""
        self.write_config("history_retention_days: 30\n")
        self.env(None)
        self.assertEqual(self.migrate_path(), DEFAULT_PATH)

    def test_migrate_reads_database_path_from_config(self) -> None:
        """BUG-05 (fixed): with no env var, migrate uses config.yaml database_path (as the bot does), without needing bot secrets."""
        self.write_config(f"database_path: {YAML_PATH}\n")
        self.env(None)
        self.assertEqual(self.migrate_path(), YAML_PATH)

    # --- web_main.main() --------------------------------------------------

    def test_web_uses_env_path_over_config(self) -> None:
        """Characterization (BUG-05): web_main uses LLMCORD_DATABASE_PATH even when config.yaml sets database_path."""
        self.write_config(f"database_path: {YAML_PATH}\n")
        self.env(ENV_PATH)
        self.assertEqual(self.web_path(), ENV_PATH)
        self.assert_no_files_created()

    def test_web_defaults_without_env_or_config(self) -> None:
        """Characterization (BUG-05): with no env var and no config.yaml, web_main uses data/llmcord.sqlite3."""
        self.env(None)
        self.assertEqual(self.web_path(), DEFAULT_PATH)
        self.assert_no_files_created()

    def test_web_defaults_when_config_lacks_key(self) -> None:
        """Characterization (BUG-05): a config.yaml without database_path leaves web_main on the default path."""
        self.write_config("history_retention_days: 30\n")
        self.env(None)
        self.assertEqual(self.web_path(), DEFAULT_PATH)

    def test_web_reads_database_path_from_config(self) -> None:
        """BUG-05 (fixed): with no env var, web_main uses config.yaml database_path (as the bot does), without needing bot secrets."""
        self.write_config(f"database_path: {YAML_PATH}\n")
        self.env(None)
        self.assertEqual(self.web_path(), YAML_PATH)

    # --- resolve_database_path() -----------------------------------------

    def test_resolver_empty_env_falls_through_to_config(self) -> None:
        """BUG-05 (fixed): an empty LLMCORD_DATABASE_PATH is ignored in favour of config.yaml database_path."""
        self.write_config(f"database_path: {YAML_PATH}\n")
        self.env("")
        self.assertEqual(resolve_database_path(), Path(YAML_PATH))

    def test_resolver_raises_on_invalid_config_yaml(self) -> None:
        """BUG-05 (fixed): an unparseable config.yaml raises instead of silently falling back to the default DB."""
        import yaml
        self.write_config("database_path: [unclosed\n")
        self.env(None)
        with self.assertRaises(yaml.YAMLError):
            resolve_database_path()

    def test_resolver_env_skips_invalid_config_yaml(self) -> None:
        """BUG-05 (fixed): with LLMCORD_DATABASE_PATH set, config.yaml is never read, so invalid YAML does not raise."""
        self.write_config("database_path: [unclosed\n")
        self.env(ENV_PATH)
        self.assertEqual(resolve_database_path(), Path(ENV_PATH))

    def test_resolver_rejects_non_mapping_config_yaml(self) -> None:
        """BUG-05 (fixed): a config.yaml that is a list or scalar raises ValueError instead of defaulting."""
        self.env(None)
        for text in ("- a\n- b\n", "just a string\n"):
            with self.subTest(text=text):
                self.write_config(text)
                with self.assertRaises(ValueError):
                    resolve_database_path()

    # --- bot: load_settings() ---------------------------------------------

    BOT_CONFIG = (
        "models:\n"
        "  dialogue: local\n"
        "  profiles:\n"
        "    local:\n"
        "      provider: compatible\n"
        "      model: fake-model\n"
        "      context_tokens: 8192\n"
        "      base_url: http://127.0.0.1:9/v1\n"
    )

    def test_bot_settings_resolve_env_then_config_then_default(self) -> None:
        """Characterization (BUG-05): load_settings already applies env > config.yaml database_path > default."""
        token = {"DISCORD_BOT_TOKEN": "fake-token"}
        self.write_config(self.BOT_CONFIG + f"database_path: {YAML_PATH}\n")
        with self.subTest("env wins"):
            with patch.dict(os.environ, {**token, "LLMCORD_DATABASE_PATH": ENV_PATH}):
                self.assertEqual(load_settings().database_path, Path(ENV_PATH))
        with self.subTest("config when env unset"):
            with patch.dict(os.environ, token):
                os.environ.pop("LLMCORD_DATABASE_PATH", None)
                self.assertEqual(load_settings().database_path, Path(YAML_PATH))
        self.write_config(self.BOT_CONFIG)
        with self.subTest("default when key absent"):
            with patch.dict(os.environ, token):
                os.environ.pop("LLMCORD_DATABASE_PATH", None)
                self.assertEqual(load_settings().database_path, Path(DEFAULT_PATH))

    def test_check_host_accepts_yaml_only_database_path(self) -> None:
        """BUG-05 (fixed): check_host does not report differing paths when only config.yaml sets database_path."""
        from scripts import check_host
        self.write_config(self.BOT_CONFIG + f"database_path: {YAML_PATH}\n")
        self.env(None, DISCORD_BOT_TOKEN="fake-token")
        with patch("sys.argv", ["check_host"]), patch("builtins.print") as printed:
            check_host.main()
        self.assertIn("PASS: shared SQLite location is writable", [c.args[0] for c in printed.call_args_list])


if __name__ == "__main__":
    unittest.main()
