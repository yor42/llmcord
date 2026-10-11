"""Build-time: download the pinned e5-small ONNX model, quantize to int8, verify hashes (FEAT-37, D35).

usage: fetch_spoken_model.py TARGET_DIR   (needs onnx + onnxruntime; not used at runtime)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmcord_core.spoken.onnx_embedder import FILE_HASHES, MODEL_ID, MODEL_REVISION, SOURCE_HASHES  # noqa: E402

HF = "https://huggingface.co"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path, dest):
    url = f"{HF}/{MODEL_ID}/resolve/{MODEL_REVISION}/{path}"
    print(f"downloading {path} @ {MODEL_REVISION[:8]}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main(target):
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    fp32, tok, int8 = out / "model.onnx", out / "tokenizer.json", out / "model_int8.onnx"
    for remote, dest in (("onnx/model.onnx", fp32), ("tokenizer.json", tok)):
        download(remote, dest)
        got = sha256(dest)
        if got != SOURCE_HASHES[remote]:
            sys.exit(f"{remote}: sha256 {got} does not match the pin {SOURCE_HASHES[remote]}")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print("quantizing to int8 (dynamic, QInt8)", flush=True)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    got = sha256(int8)
    print(f"model_int8.onnx sha256 {got}")
    if got != FILE_HASHES["model_int8.onnx"]:
        sys.exit(f"model_int8.onnx sha256 {got} does not match the pin {FILE_HASHES['model_int8.onnx']}")
    fp32.unlink()
    print(f"ok: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
