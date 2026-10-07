FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY llmcord.py web_main.py migrate.py config-example.yaml config-gemini.yaml LICENSE.md ./
COPY llmcord_core/ ./llmcord_core/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

CMD ["python", "llmcord.py"]
