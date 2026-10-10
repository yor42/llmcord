"""Test web_main.py configuration and startup."""
import unittest
from unittest import mock


class TestWebMainKeepAlive(unittest.TestCase):
    """Test that timeout_keep_alive is configured correctly."""

    @mock.patch.dict("os.environ", {}, clear=True)
    @mock.patch("web_main.operator_ids_from_env")
    @mock.patch("web_main.resolve_database_path")
    @mock.patch("web_main.create_app")
    @mock.patch("web_main.uvicorn.run")
    def test_timeout_keep_alive_default(self, mock_run, mock_create_app, mock_resolve_db, mock_ops, ):
        """Assert timeout_keep_alive defaults to 95 seconds."""
        import web_main

        web_main.main()

        # Verify uvicorn.run was called with timeout_keep_alive=95
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["timeout_keep_alive"], 95)

    @mock.patch.dict("os.environ", {"WEB_KEEP_ALIVE": "30"}, clear=True)
    @mock.patch("web_main.operator_ids_from_env")
    @mock.patch("web_main.resolve_database_path")
    @mock.patch("web_main.create_app")
    @mock.patch("web_main.uvicorn.run")
    def test_timeout_keep_alive_custom(self, mock_run, mock_create_app, mock_resolve_db, mock_ops, ):
        """Assert timeout_keep_alive respects WEB_KEEP_ALIVE environment variable."""
        import web_main

        web_main.main()

        # Verify uvicorn.run was called with timeout_keep_alive=30
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["timeout_keep_alive"], 30)


if __name__ == "__main__":
    unittest.main()
