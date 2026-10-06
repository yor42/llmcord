from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .config import Settings


@dataclass(frozen=True)
class ImageInput:
    media_type: str
    data: bytes

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode('ascii')}"


@dataclass(frozen=True)
class TurnMessage:
    role: str
    text: str
    images: list[ImageInput] = field(default_factory=list)


class ModelGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.clients: dict[str, Any] = {}

    def _client(self, profile_name: str):
        if profile_name not in self.clients:
            profile = self.settings.profiles[profile_name]
            key = os.environ.get(profile.api_key_env or "", "local-no-key")
            if profile.provider == "anthropic":
                from anthropic import AsyncAnthropic
                self.clients[profile_name] = AsyncAnthropic(api_key=key)
            else:
                from openai import AsyncOpenAI
                kwargs = {"api_key": key}
                if profile.base_url:
                    kwargs["base_url"] = profile.base_url
                self.clients[profile_name] = AsyncOpenAI(**kwargs)
        return self.clients[profile_name]

    @staticmethod
    def _openai_input(messages: list[TurnMessage]) -> list[dict]:
        result = []
        for message in messages:
            if message.images:
                content: Any = [{"type": "input_text", "text": message.text}]
                content += [{"type": "input_image", "image_url": image.data_url()} for image in message.images]
            else:
                content = message.text
            result.append({"role": message.role, "content": content})
        return result

    @staticmethod
    def _chat_input(messages: list[TurnMessage]) -> list[dict]:
        result = []
        for message in messages:
            if message.images:
                content: Any = [{"type": "text", "text": message.text}]
                content += [{"type": "image_url", "image_url": {"url": image.data_url()}} for image in message.images]
            else:
                content = message.text
            result.append({"role": message.role, "content": content})
        return result

    @staticmethod
    def _anthropic_input(messages: list[TurnMessage]) -> list[dict]:
        result = []
        for message in messages:
            if message.images:
                content: Any = [{"type": "text", "text": message.text}]
                content += [
                    {"type": "image", "source": {"type": "base64", "media_type": image.media_type,
                      "data": base64.b64encode(image.data).decode("ascii")}}
                    for image in message.images
                ]
            else:
                content = message.text
            result.append({"role": message.role, "content": content})
        return result

    async def text(self, role: str, system: str, messages: list[TurnMessage], max_tokens: int | None = None) -> str:
        profile_name = getattr(self.settings, role)
        profile = self.settings.profiles[profile_name]
        client = self._client(profile_name)
        limit = max_tokens or self.settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            response = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit)
            return response.output_text or ""
        if profile.provider == "anthropic":
            response = await client.messages.create(model=profile.model, system=system,
                messages=self._anthropic_input(messages), max_tokens=limit)
            return "".join(block.text for block in response.content if block.type == "text")
        response = await client.chat.completions.create(model=profile.model,
            messages=[{"role": "system", "content": system}, *self._chat_input(messages)], max_tokens=limit)
        return response.choices[0].message.content or ""

    async def stream_text(self, role: str, system: str, messages: list[TurnMessage], max_tokens: int | None = None) -> AsyncIterator[str]:
        profile_name = getattr(self.settings, role)
        profile = self.settings.profiles[profile_name]
        client = self._client(profile_name)
        limit = max_tokens or self.settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            stream = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit, stream=True)
            async for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    yield event.delta
        elif profile.provider == "anthropic":
            async with client.messages.stream(model=profile.model, system=system,
                messages=self._anthropic_input(messages), max_tokens=limit) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        else:
            stream = await client.chat.completions.create(model=profile.model,
                messages=[{"role": "system", "content": system}, *self._chat_input(messages)],
                max_tokens=limit, stream=True)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

    async def structured(self, role: str, system: str, messages: list[TurnMessage], schema_name: str, schema: dict) -> dict:
        profile_name = getattr(self.settings, role)
        profile = self.settings.profiles[profile_name]
        client = self._client(profile_name)
        limit = self.settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            response = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit,
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}})
            return json.loads(response.output_text)
        if profile.provider == "anthropic":
            response = await client.messages.create(model=profile.model, system=system,
                messages=self._anthropic_input(messages), max_tokens=limit,
                tools=[{"name": schema_name, "description": "Return the requested structured result", "input_schema": schema}],
                tool_choice={"type": "tool", "name": schema_name})
            for block in response.content:
                if block.type == "tool_use" and block.name == schema_name:
                    return block.input
            raise ValueError("Model returned no structured result")
        instruction = f"{system}\nReturn only JSON matching this schema: {json.dumps(schema)}"
        for attempt in range(2):
            raw = await self.text(role, instruction, messages, limit)
            try:
                value = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
            instruction += "\nYour previous result was not a JSON object. Try again."
        raise ValueError("Compatible model returned invalid JSON")

    async def close(self) -> None:
        for client in self.clients.values():
            await client.close()


DIRECTOR_SCHEMA = {
    "type": "object", "properties": {
        "speakers": {"type": "array", "items": {"type": "integer"}},
    }, "required": ["speakers"], "additionalProperties": False,
}
MEMORY_SCHEMA = {
    "type": "object", "properties": {
        "shared_facts": {"type": "array", "items": {"type": "string"}},
        "personal_facts": {"type": "array", "items": {"type": "string"}},
        "encounter_facts": {"type": "array", "items": {"type": "string"}},
    }, "required": ["shared_facts", "personal_facts", "encounter_facts"],
    "additionalProperties": False,
}
