"""Real multilingual-e5-small embedder (int8 ONNX) and the lazy runtime that serves it off the event loop."""
import asyncio
import hashlib
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PREFIX = "query: "
MAX_TOKENS = 128
ENV_DIR = "LLMCORD_SPOKEN_MODEL_DIR"
DEFAULT_DIR = "/usr/src/app/models/e5-small"
LOCAL_DIR = Path.home() / ".cache" / "llmcord-spoken" / "intfloat__multilingual-e5-small"
# sha256 of the files in the model dir (int8 is derived from the pinned fp32 model at build time)
FILE_HASHES = {
    "model_int8.onnx": "8da4c9ba0ad59f58e8566839425d7fd6339d31414d0ce5cba2d7d0afb75dd8b6",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
}
SOURCE_HASHES = {"onnx/model.onnx": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665", "tokenizer.json": FILE_HASHES["tokenizer.json"]}


class ModelError(RuntimeError):
    pass


def default_model_dir():
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env)
    return Path(DEFAULT_DIR) if Path(DEFAULT_DIR).is_dir() else LOCAL_DIR


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(model_dir, hashes=FILE_HASHES):
    model_dir = Path(model_dir)
    for name, want in hashes.items():
        path = model_dir / name
        if not path.is_file():
            raise ModelError(f"spoken model file missing: {path}")
        if _sha256(path) != want:
            raise ModelError(f"spoken model file {name} does not match its pinned sha256")


def check_vectors(vectors):
    for vec in vectors:
        if not all(math.isfinite(x) for x in vec):
            raise ModelError("embedder produced a non-finite vector")
    return vectors


class OnnxEmbedder:
    prefix = PREFIX

    def __init__(self, model_dir, threads=1, hashes=FILE_HASHES):
        self.model_dir, self.threads, self.hashes = Path(model_dir), threads, hashes
        self.session = self.tok = self.np = None
        self.inputs = frozenset()

    def load(self, verify=True):
        if verify:
            verify_files(self.model_dir, self.hashes)
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ModelError(f"spoken model runtime is not installed: {exc}") from exc
        self.np = np
        self.tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = self.threads, 1
        self.session = ort.InferenceSession(str(self.model_dir / "model_int8.onnx"), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        return self

    def embed(self, texts):
        if self.session is None:
            self.load()
        np = self.np
        encs = self.tok.encode_batch([self.prefix + t for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        m = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        out = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return check_vectors(out.astype(float).tolist())


class SpokenRuntime:
    """Lazy: nothing is hashed or loaded until `ready()` or the first `embed`, always on the worker thread.

    A failure (missing/mismatched files, load error) is logged once and the feature stays off.
    Callers treat ModelError from `embed` (including a non-finite vector) as "no reading".
    """

    def __init__(self, model_dir=None, embedder=None):
        self._dir, self._embedder = model_dir, embedder
        self._executor = None
        self._state = True if embedder is not None else None
        self._vector_logged = False

    @property
    def available(self):
        """Cached state only (None = not checked yet); never touches the disk."""
        return self._state

    def _prepare(self):
        if self._state is None:
            try:
                model_dir = Path(self._dir) if self._dir else default_model_dir()
                verify_files(model_dir)
                self._embedder = OnnxEmbedder(model_dir).load(verify=False)
                self._state = True
            except Exception as exc:
                self._state = False
                log.error("spoken moves disabled: %s", exc)
        return self._state

    def _pool(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spoken")
        return self._executor

    async def ready(self):
        if self._state is None:
            await asyncio.get_running_loop().run_in_executor(self._pool(), self._prepare)
        return bool(self._state)

    def _run(self, text):
        if not self._prepare():
            raise ModelError("spoken model unavailable")
        try:
            return self._embedder.embed([text])[0]
        except ModelError as exc:
            if not self._vector_logged:
                self._vector_logged = True
                log.error("spoken embedder rejected a vector: %s", exc)
            raise

    async def embed(self, text):
        return await asyncio.get_running_loop().run_in_executor(self._pool(), self._run, text)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = SpokenRuntime()
    return _runtime
