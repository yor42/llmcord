import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from llmcord_core.spoken import onnx_embedder
from llmcord_core.spoken.onnx_embedder import ModelError, OnnxEmbedder, SpokenRuntime, check_vectors, verify_files


class ThreadEmbedder:
    prefix = "query: "

    def __init__(self, vec=None):
        self.vec, self.threads = vec or [0.5, 0.5], []

    def embed(self, texts):
        self.threads.append(threading.current_thread())
        return [list(self.vec) for _ in texts]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RuntimeTests(unittest.TestCase):
    def test_import_does_not_load_onnxruntime(self):
        code = "import sys; import llmcord_core.spoken.onnx_embedder; print(any(m in sys.modules for m in ('onnxruntime', 'numpy', 'tokenizers')))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent)
        self.assertEqual(out.stdout.strip(), "False")

    def test_missing_files_unavailable_with_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            rt = SpokenRuntime(model_dir=d)
            self.assertIsNone(rt.available)
            with self.assertRaisesRegex(ModelError, "missing"):
                OnnxEmbedder(d).load()
            with self.assertLogs(onnx_embedder.log, "ERROR") as logs:
                self.assertFalse(asyncio.run(rt.ready()))
                self.assertFalse(rt.available)
                with self.assertRaises(ModelError):
                    asyncio.run(rt.embed("hit me"))
                with self.assertRaises(ModelError):
                    asyncio.run(rt.embed("hit me"))
            self.assertEqual(len(logs.records), 1)
            self.assertFalse(rt.available)
            rt.close()

    def test_sha_mismatch_rejected_and_match_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.bin").write_bytes(b"one")
            (Path(d) / "b.bin").write_bytes(b"two")
            good = {"a.bin": digest(b"one"), "b.bin": digest(b"two")}
            verify_files(d, good)
            with self.assertRaisesRegex(ModelError, "b.bin"):
                verify_files(d, {**good, "b.bin": digest(b"other")})
            with self.assertRaisesRegex(ModelError, "pinned sha256"):
                OnnxEmbedder(d, hashes={**good, "b.bin": digest(b"other")}).load()

    def test_embed_runs_on_single_worker_thread_not_caller(self):
        emb = ThreadEmbedder()
        rt = SpokenRuntime(embedder=emb)

        async def go():
            caller = threading.current_thread()
            a, b = await rt.embed("x"), await rt.embed("y")
            return caller, a, b

        caller, a, b = asyncio.run(go())
        self.assertEqual(a, [0.5, 0.5])
        self.assertEqual(len(emb.threads), 2)
        self.assertNotIn(caller, emb.threads)
        self.assertEqual(emb.threads[0], emb.threads[1])
        rt.close()
        self.assertIsNone(rt._executor)

    def test_ready_verifies_once_off_loop_and_available_is_cached_state(self):
        threads = []

        def fake_verify(*a, **k):
            threads.append(threading.current_thread())

        class Loaded:
            def load(self, verify=True):
                threads.append(verify)
                return ThreadEmbedder()

        async def go():
            rt = SpokenRuntime(model_dir="/nonexistent")
            self.assertIsNone(rt.available)
            self.assertTrue(await rt.ready())
            self.assertTrue(await rt.ready())
            self.assertTrue(rt.available)
            rt.close()
            return threading.current_thread()

        with mock.patch.object(onnx_embedder, "verify_files", side_effect=fake_verify) as v, \
                mock.patch.object(onnx_embedder, "OnnxEmbedder", return_value=Loaded()):
            caller = asyncio.run(go())
        self.assertEqual(v.call_count, 1)
        self.assertNotEqual(threads[0], caller)
        self.assertIs(threads[1], False)

    def test_runtime_non_finite_logged_once(self):
        rt = SpokenRuntime(embedder=ThreadEmbedder([float("nan")]))
        rt._embedder.embed = lambda texts: check_vectors([[float("nan")]])

        async def go():
            for _ in range(2):
                with self.assertRaises(ModelError):
                    await rt.embed("x")

        with self.assertLogs(onnx_embedder.log, "ERROR") as logs:
            asyncio.run(go())
        self.assertEqual(len(logs.records), 1)
        rt.close()

    def test_non_finite_vectors_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ModelError):
                check_vectors([[0.1, bad]])
        self.assertEqual(check_vectors([[0.1, 0.2]]), [[0.1, 0.2]])


@unittest.skipUnless(os.environ.get("LLMCORD_SPOKEN_MODEL_DIR"), "LLMCORD_SPOKEN_MODEL_DIR not set")
class RealModelSmoke(unittest.TestCase):
    def test_hit_me_decides_hit(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        from llmcord_core.spoken.head import blackjack_head
        from llmcord_core.spoken.interpret import decide

        emb = OnnxEmbedder(os.environ["LLMCORD_SPOKEN_MODEL_DIR"]).load()
        vec = emb.embed(["hit me"])[0]
        self.assertEqual(len(vec), 384)
        d = decide(vec, ("hit", "stand"), blackjack_head())
        self.assertIn(d.band, ("sure", "unsure"))
        self.assertEqual(d.moves[0], "hit")


if __name__ == "__main__":
    unittest.main()
