"""FEAT-08 step 2: operator guard (AuthService.guard_operator) and AdminService.run_operator."""
import unittest
from collections import Counter

import httpx
from fastapi import HTTPException

from helpers import discord_transport, install_session

from llmcord_core.admin import OPERATOR_AUDIT_GUILD
from llmcord_core.web import create_app

GUILDS = "GET /users/@me/guilds"


class OperatorGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = Counter()
        self.ran = []
        transport, _ = discord_transport(admin_guilds=[1], calls=self.calls)
        self.make(transport, frozenset({4}))
        self.session = install_session(self.app, user_id="4")

    def make(self, transport, operator_ids):
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml", operator_ids=operator_ids)
        self.auth, self.service, self.store = self.app.state.auth, self.app.state.admin, self.app.state.store

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    async def assertStatus(self, status, awaitable):
        with self.assertRaises(HTTPException) as caught:
            await awaitable
        self.assertEqual(caught.exception.status_code, status)

    def operation(self):
        self.ran.append(True)
        return "done"

    def audit_rows(self):
        return [dict(r) for r in self.store.all("SELECT guild_id, actor_id, action, detail_json FROM admin_audit ORDER BY id")]

    async def test_operator_passes_and_returns_session(self):
        session = await self.auth.guard_operator(self.session, "csrf", "https://pi.test")
        self.assertIs(session, self.app.state.sessions[self.session])
        self.assertEqual(self.calls[GUILDS], 0)

    async def test_non_operator_refused_without_guild_fetch(self):
        other = install_session(self.app, ident="other", user_id="5")
        await self.assertStatus(403, self.auth.guard_operator(other))
        self.assertEqual(self.calls[GUILDS], 0)

    async def test_unknown_session_401(self):
        await self.assertStatus(401, self.auth.guard_operator("nope"))

    async def test_expired_session_401(self):
        self.app.state.sessions[self.session]["expires"] = 0
        await self.assertStatus(401, self.auth.guard_operator(self.session))

    async def test_bad_csrf_403(self):
        await self.assertStatus(403, self.auth.guard_operator(self.session, "wrong"))

    async def test_bad_origin_403(self):
        await self.assertStatus(403, self.auth.guard_operator(self.session, None, "https://evil.test"))

    async def test_empty_allowlist_refuses_everyone(self):
        self.app.state.operator_ids = frozenset()
        await self.assertStatus(403, self.auth.guard_operator(self.session))
        await self.assertStatus(403, self.auth.guard_operator(self.session, "csrf", "https://pi.test"))

    async def test_malformed_session_user_refused(self):
        for label, user in (("empty", {}), ("none id", {"id": None}), ("non-numeric id", {"id": "x"}), ("no user", ...)):
            with self.subTest(user=label):
                ident = install_session(self.app, ident="bad", user_id="4")
                if user is ...:
                    del self.app.state.sessions[ident]["user"]
                else:
                    self.app.state.sessions[ident]["user"] = user
                await self.assertStatus(403, self.auth.guard_operator(ident))

    async def test_run_operator_runs_and_audits_guild_zero(self):
        result = await self.service.run_operator(self.session, self.operation, "operator.test", {"k": 1})
        self.assertEqual(result, "done")
        rows = self.audit_rows()
        self.assertEqual([(r["guild_id"], r["actor_id"], r["action"]) for r in rows],
                         [(OPERATOR_AUDIT_GUILD, 4, "operator.test")])
        self.assertEqual(OPERATOR_AUDIT_GUILD, 0)

    async def test_run_operator_awaitable_and_callable_detail(self):
        async def op():
            self.ran.append(True)
            return 7
        result = await self.service.run_operator(self.session, op, "operator.test", lambda r: {"n": r})
        self.assertEqual(result, 7)
        self.assertIn('"n": 7', self.audit_rows()[0]["detail_json"])

    async def test_run_operator_refuses_non_operator(self):
        other = install_session(self.app, ident="other", user_id="5")
        await self.assertStatus(403, self.service.run_operator(other, self.operation, "operator.test", {}))
        self.assertEqual(self.ran, [])
        self.assertEqual(self.audit_rows(), [])

    async def test_server_guard_with_guild_zero_refuses_operator(self):
        await self.assertStatus(403, self.auth.guard(self.session, 0))
        await self.assertStatus(403, self.service.run(self.session, 0, self.operation, "x", {}))
        self.assertEqual(self.ran, [])


if __name__ == "__main__":
    unittest.main()
