"""Dashboard link/unlink/bind keep the legacy routes' audit detail (SEC-02, audit detail kept after legacy retirement).

Seam: the module-level operation and detail builders in ``dashboard.py`` that the buttons are wired to (the closures
need a live NiceGUI page); detail is applied through ``AdminService.run``'s callable-detail path, covered in
``test_admin_service_run.py``.
"""
import json
import unittest
from collections import Counter

import httpx

from helpers import discord_transport, install_session

from llmcord_core.dashboard import bind_detail, bind_operation, link_detail, link_operation
from llmcord_core.store import Store
from llmcord_core.web import create_app


class DashboardAuditDetailTests(unittest.TestCase):
    def setUp(self):
        self.store = s = Store(":memory:")
        self.harbor, self.forest = s.create_space(1, "Harbor", "world"), s.create_space(1, "Forest", "world")
        self.plaza = s.create_space(1, "Plaza", "hub")
        s.link_world(1, self.plaza, self.harbor)
        s.link_world(1, self.plaza, self.forest)
        add = lambda world, name: s.add_character(1, world, name, {"name": name}, None, [])  # noqa: E731
        self.alice, self.carol = add(self.harbor, "Alice"), add(self.forest, "Carol")
        s.bind_channel(1, 10, self.plaza)
        s.set_cast(10, None, [self.alice, self.carol], default=True)

    def tearDown(self):
        self.store.close()

    def test_unlink_audits_hub_world_enabled_and_pruned(self):
        """SEC-02: unlink detail has hub, world, enabled=False and the pruned count (> 0)."""
        result = link_operation(self.store, 1, self.plaza, self.forest, False)
        self.assertGreater(result["pruned"], 0)
        self.assertEqual(link_detail(result),
                         {"hub": self.plaza, "world": self.forest, "enabled": False, "pruned": result["pruned"]})

    def test_link_audits_hub_world_enabled_and_zero_pruned(self):
        """SEC-02: link detail has enabled=True and pruned 0."""
        other = self.store.create_space(1, "Other", "world")
        result = link_operation(self.store, 1, self.plaza, other, True)
        self.assertEqual(link_detail(result),
                         {"hub": self.plaza, "world": other, "enabled": True, "pruned": 0})

    def test_bind_audits_channel_space_and_dropped_ids(self):
        """SEC-02: bind detail has channel, space and the dropped character ids; names are kept for the notification."""
        result = bind_operation(self.store, 1, 10, self.harbor)
        self.assertEqual(result["names"], ["Carol"])
        self.assertEqual(bind_detail(result), {"channel": 10, "space": self.harbor, "dropped": [self.carol]})


class AdminServiceLinkDetailTests(unittest.IsolatedAsyncioTestCase):
    """SEC-02: link_operation and link_detail are wired through AdminService.run and audited correctly."""

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
        # Set up hub and worlds for linking, with forest linked to plaza
        self.plaza = self.store.create_space(1, "Plaza", "hub")
        self.forest = self.store.create_space(1, "Forest", "world")
        self.store.link_world(1, self.plaza, self.forest)
        add = lambda world, name: self.store.add_character(1, world, name, {"name": name}, None, [])  # noqa: E731
        self.carol = add(self.forest, "Carol")
        self.store.bind_channel(1, 10, self.plaza)
        self.store.set_cast(10, None, [self.carol], default=True)

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    def audit_rows(self):
        return [dict(row) for row in self.store.all(
            "SELECT guild_id, actor_id, action, detail_json FROM admin_audit ORDER BY id")]

    async def test_unlink_operation_with_detail_audited_end_to_end(self):
        """SEC-02: link_operation (unlink) and link_detail flow through AdminService.run and produce correct audit detail with pruned > 0."""
        operation = lambda: link_operation(self.store, 1, self.plaza, self.forest, False)
        await self.service.run(self.session, 1, operation, "hub.unlink", link_detail)
        rows = self.audit_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "hub.unlink")
        detail = json.loads(rows[0]["detail_json"])
        self.assertEqual(detail, {"hub": self.plaza, "world": self.forest, "enabled": False, "pruned": detail["pruned"]})
        self.assertGreater(detail["pruned"], 0)


if __name__ == "__main__":
    unittest.main()
