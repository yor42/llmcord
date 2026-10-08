"""AdminService.run authorization (SEC-03 / roadmap R2.2).

These pin what must survive dropping the redundant csrf/origin arguments on live calls: every action still
re-checks the session and the guild's admin permission (cheap now that guild lists are cached), and only an
authorized action runs and is audited. Assertions are on observable behavior only (op ran or not, exception
status, audit rows), never on the arguments passed to ``guard``.
"""
import json
import unittest
from collections import Counter

import httpx
from fastapi import HTTPException

from helpers import discord_transport, install_session

from llmcord_core.web import create_app


async def run_action(service, ident, guild_id, operation, action=None, detail=None):
    # Single call site for AdminService.run (no csrf parameter since R2.2 / SEC-03).
    return await service.run(ident, guild_id, operation, action, detail)


class AdminServiceRunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = Counter()
        self.admin_guilds = [1]
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls)
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.service = self.app.state.admin
        self.store = self.app.state.store
        self.session = install_session(self.app, user_id="4")
        self.ran = []

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    def operation(self):
        self.ran.append(True)
        return "done"

    def audit_rows(self):
        return [dict(row) for row in self.store.all(
            "SELECT guild_id, actor_id, action, detail_json FROM admin_audit ORDER BY id")]

    async def assertRejected(self, status, ident, guild_id):
        with self.assertRaises(HTTPException) as caught:
            await run_action(self.service, ident, guild_id, self.operation, "space.create", {"name": "X"})
        self.assertEqual(caught.exception.status_code, status)
        self.assertEqual(self.ran, [])
        self.assertEqual(self.audit_rows(), [])

    async def test_admin_session_runs_operation_and_audits(self):
        """SEC-03: an admin session's action runs, returns its result and writes one audit row."""
        result = await run_action(self.service, self.session, 1, self.operation, "space.create", {"name": "X"})
        self.assertEqual(result, "done")
        self.assertEqual(self.ran, [True])
        self.assertEqual(self.audit_rows(), [{"guild_id": 1, "actor_id": 4, "action": "space.create",
                                              "detail_json": json.dumps({"name": "X"})}])

    async def test_callable_detail_is_audited_from_the_result(self):
        """SEC-02 (audit detail kept after legacy retirement): a callable detail is called with the operation result."""
        await run_action(self.service, self.session, 1, self.operation, "x.y", lambda result: {"got": result})
        self.assertEqual([json.loads(r["detail_json"]) for r in self.audit_rows()], [{"got": "done"}])

    async def test_failed_operation_does_not_call_detail_or_audit(self):
        """SEC-02: an operation that raises audits nothing and never calls the detail callable."""
        called = []
        def boom():
            raise ValueError("no")
        with self.assertRaises(ValueError):
            await run_action(self.service, self.session, 1, boom, "x.y", lambda result: called.append(result) or {})
        self.assertEqual(called, [])
        self.assertEqual(self.audit_rows(), [])

    async def test_async_operation_is_awaited_and_unaudited_without_action(self):
        """SEC-03: an awaitable operation is awaited; no action name means no audit row."""
        async def operation():
            self.ran.append(True)
            return 7
        self.assertEqual(await run_action(self.service, self.session, 1, operation), 7)
        self.assertEqual(self.ran, [True])
        self.assertEqual(self.audit_rows(), [])

    async def test_non_admin_guild_is_forbidden_and_op_not_run(self):
        """SEC-03: a guild the session does not administer is rejected with 403 per action."""
        await self.assertRejected(403, self.session, 2)

    async def test_expired_session_is_rejected_and_op_not_run(self):
        """SEC-03: an expired session is rejected with 401 per action, even with a warm guild cache."""
        await run_action(self.service, self.session, 1, lambda: None)
        self.app.state.sessions[self.session]["expires"] = 0
        await self.assertRejected(401, self.session, 1)

    async def test_unknown_session_is_rejected_and_op_not_run(self):
        """SEC-03: an unknown or empty session ident is rejected with 401."""
        await self.assertRejected(401, "unknown-session", 1)
        await self.assertRejected(401, "", 1)

    async def test_signed_out_session_is_rejected_and_op_not_run(self):
        """SEC-03: a session removed after a successful action (sign-out) is rejected with 401 on the next one."""
        await run_action(self.service, self.session, 1, lambda: None)
        self.app.state.sessions.pop(self.session)
        await self.assertRejected(401, self.session, 1)

    async def test_revoked_admin_rejected_after_forget_guilds(self):
        """SEC-03 / PERF-01: after Discord revokes admin and the guild cache is dropped, the next action is 403."""
        await run_action(self.service, self.session, 1, lambda: None)
        self.admin_guilds.remove(1)
        self.app.state.auth.forget_guilds()
        await self.assertRejected(403, self.session, 1)


if __name__ == "__main__":
    unittest.main()
