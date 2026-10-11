FROM python:3.12-slim AS spoken-model

WORKDIR /build
COPY requirements-spoken-build.txt .
RUN pip install --no-cache-dir -r requirements-spoken-build.txt onnxruntime==1.22.1 numpy==2.3.2

COPY llmcord_core/__init__.py ./llmcord_core/
COPY llmcord_core/spoken/__init__.py llmcord_core/spoken/moves.py llmcord_core/spoken/onnx_embedder.py ./llmcord_core/spoken/
COPY scripts/fetch_spoken_model.py ./scripts/
RUN python scripts/fetch_spoken_model.py /models/e5-small

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLMCORD_SPOKEN_MODEL_DIR=/usr/src/app/models/e5-small

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --from=spoken-model /models/e5-small ./models/e5-small
COPY llmcord.py web_main.py migrate.py config-example.yaml config-gemini.yaml LICENSE.md ./
COPY llmcord_core/ ./llmcord_core/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

CMD ["python", "llmcord.py"]
