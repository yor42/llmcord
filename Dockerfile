FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app

RUN pip install --no-cache-dir discord.py httpx openai

CMD [ "python", "./llmcord.py" ]
