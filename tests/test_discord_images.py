"""Discord CDN image URL helpers and the admin img-src policy (MNT-18)."""
import unittest

import httpx
from fastapi.testclient import TestClient

from helpers import discord_transport, install_session, shared_dashboard

from llmcord_core import auth
from llmcord_core.web import create_app

CDN = "https://cdn.discordapp.com"
HASH = "0123456789abcdef0123456789abcdef"
UID = "80351110224678912"  # (id >> 22) % 6 == 5
UID2 = 1234567890123456789  # (id >> 22) % 6 == 1


class UserAvatarUrlTests(unittest.TestCase):
    def test_valid_hash_default_and_custom_size(self):
        """MNT-18: a valid avatar hash yields the exact CDN png URL; size defaults to 64 and can be changed."""
        user = {"id": UID, "avatar": HASH}
        self.assertEqual(auth.user_avatar_url(user), f"{CDN}/avatars/{UID}/{HASH}.png?size=64")
        self.assertEqual(auth.user_avatar_url(user, size=128), f"{CDN}/avatars/{UID}/{HASH}.png?size=128")

    def test_animated_hash_uses_png(self):
        """MNT-18: animated a_ hashes still get a static .png URL."""
        user = {"id": UID, "avatar": "a_" + HASH}
        self.assertEqual(auth.user_avatar_url(user), f"{CDN}/avatars/{UID}/a_{HASH}.png?size=64")

    def test_missing_avatar_uses_default_embed_avatar(self):
        """MNT-18: None, missing or empty avatar falls back to the default avatar index (id >> 22) % 6."""
        self.assertEqual(auth.user_avatar_url({"id": UID, "avatar": None}), f"{CDN}/embed/avatars/5.png")
        self.assertEqual(auth.user_avatar_url({"id": UID}), f"{CDN}/embed/avatars/5.png")
        self.assertEqual(auth.user_avatar_url({"id": UID, "avatar": ""}), f"{CDN}/embed/avatars/5.png")
        self.assertEqual(auth.user_avatar_url({"id": UID2}), f"{CDN}/embed/avatars/1.png")

    def test_invalid_hash_never_interpolated(self):
        """MNT-18: malformed hashes fall back to the default avatar and never appear in the URL."""
        bad = ["../x", "abc", "A" * 32, "a" * 31 + "?", "a" * 31 + '"', "a" * 31 + "/", HASH + "/..", 12345, ["x"], b"x"]
        for value in bad:
            url = auth.user_avatar_url({"id": UID, "avatar": value})
            self.assertEqual(url, f"{CDN}/embed/avatars/5.png", value)
            self.assertNotIn(str(value), url)

    def test_bad_id_returns_none(self):
        """MNT-18: a non-numeric or missing id gives None."""
        for user in ({"avatar": HASH}, {"id": "abc", "avatar": HASH}, {"id": None}, {"id": "12/3"}, {"id": "../1"}):
            self.assertIsNone(auth.user_avatar_url(user), user)

    def test_invalid_size_raises(self):
        """MNT-18: sizes outside Discord's powers of two 16..4096 raise ValueError."""
        for size in (100, 0, 8192, "64", 8, -64):
            with self.assertRaises(ValueError, msg=repr(size)):
                auth.user_avatar_url({"id": UID, "avatar": HASH}, size=size)


class GuildIconUrlTests(unittest.TestCase):
    def test_valid_icon(self):
        """MNT-18: a valid icon hash yields the exact CDN png URL; custom size is honored."""
        guild = {"id": "42", "icon": HASH}
        self.assertEqual(auth.guild_icon_url(guild), f"{CDN}/icons/42/{HASH}.png?size=64")
        self.assertEqual(auth.guild_icon_url(guild, size=256), f"{CDN}/icons/42/{HASH}.png?size=256")

    def test_animated_icon_uses_png(self):
        """MNT-18: animated icon hashes get .png."""
        self.assertEqual(auth.guild_icon_url({"id": "42", "icon": "a_" + HASH}), f"{CDN}/icons/42/a_{HASH}.png?size=64")

    def test_missing_or_invalid_icon_is_none(self):
        """MNT-18: no icon or an invalid hash yields None."""
        for icon in (None, "", "../x", "abc", "A" * 32, HASH + "?x", 5):
            self.assertIsNone(auth.guild_icon_url({"id": "42", "icon": icon}), icon)
        self.assertIsNone(auth.guild_icon_url({"id": "42"}))

    def test_bad_id_is_none(self):
        """MNT-18: non-numeric or missing guild id yields None."""
        for guild in ({"icon": HASH}, {"id": "x1", "icon": HASH}, {"id": "../1", "icon": HASH}, {"id": None, "icon": HASH}):
            self.assertIsNone(auth.guild_icon_url(guild), guild)

    def test_invalid_size_raises(self):
        """MNT-18: invalid sizes raise ValueError."""
        for size in (100, 0, 8192, "64"):
            with self.assertRaises(ValueError, msg=repr(size)):
                auth.guild_icon_url({"id": "42", "icon": HASH}, size=size)


class CspImgSrcTests(unittest.TestCase):
    def test_admin_policy_allows_discord_cdn_only_in_admin(self):
        """MNT-18: /admin HTML allows the Discord CDN in img-src; a non-/admin response has no img-src directive."""
        app, client = shared_dashboard()
        client.cookies.set("llmcord_session", install_session(app))
        admin = client.get("/admin/guild/1", headers={"accept-encoding": "identity"})
        self.assertIn("img-src 'self' data: blob: https://cdn.discordapp.com;", admin.headers["content-security-policy"])

        transport, _ = discord_transport(admin_guilds=(1,))
        plain_app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                               httpx.AsyncClient(transport=transport), enable_dashboard=False,
                               config_path="tests/nonexistent-config.yaml")
        with TestClient(plain_app, base_url="https://pi.test") as plain:
            response = plain.get("/login", follow_redirects=False)
        self.assertNotIn("img-src", response.headers["content-security-policy"])


if __name__ == "__main__":
    unittest.main()
