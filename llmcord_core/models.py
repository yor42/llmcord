from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from . import budget
from .backend import settings_source
from .config import ModelProfile, Settings, check_key_pin
from .errors import ModelConfigError, error_detail
from .usage import collect_usage, mask_for_log


def validate_result(value, schema):
    """Validate the object/array/scalar schema used by our structured calls."""
    kind = schema.get('type')
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
        'string': isinstance(value, str), 'integer': type(value) is int,
        'boolean': type(value) is bool, 'number': type(value) in {int, float}, 'null': value is None}
    if kind and not valid.get(kind, False):
        raise ValueError('Model result does not match the required output schema')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        if set(schema.get('required', [])) - set(value) or (schema.get('additionalProperties') is False and set(value) - set(properties)):
            raise ValueError('Model result has missing or unexpected fields')
        for key in set(value) & set(properties):
            validate_result(value[key], properties[key])
    elif isinstance(value, list) and 'items' in schema:
        for item in value:
            validate_result(item, schema['items'])
    return value


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


def AsyncOpenAI(**kwargs):
    from openai import AsyncOpenAI as sdk  # imported on first use: the web process imports this module too
    return sdk(**kwargs)


def AsyncAnthropic(**kwargs):
    from anthropic import AsyncAnthropic as sdk
    return sdk(**kwargs)


class ModelGateway:
    def __init__(self, settings, usage_sink=None, budget_gate=None, log_sink=None):
        """`settings` is a Settings or a BackendResolver (anything with `.current()`)."""
        self.source = settings_source(settings)
        self.fingerprints: dict[str, tuple] = {}
        self.retired: list[Any] = []
        self.log_sink = log_sink
        self.clients: dict[str, Any] = {}
        self.usage_sink = usage_sink
        self.budget_gate = budget_gate

    @property
    def settings(self) -> Settings:
        return self.source.current()

    @settings.setter
    def settings(self, value: Settings) -> None:
        self.source = settings_source(value)

    def _check_budget(self):
        if self.budget_gate is None or budget.admitted():
            return
        try:
            state = self.budget_gate()
        except Exception as error:
            logging.error('Budget gate failed: %s', type(error).__name__)
            return
        if state is not None and state.hard_reached:
            raise budget.BudgetExceeded(state)

    def _entry(self, role, system, messages, settings):
        profile_name = getattr(settings, role)
        return {'role': role, 'profile': profile_name, 'model': settings.profiles[profile_name].model,
                'system': system, 'messages': messages, 'usage': None, 'raw': None, 'ctx': contextvars.copy_context()}

    @staticmethod
    def _render(system, messages):
        parts = [system] if system else []
        for message in messages:
            images = ''.join(f'\n[image: {image.media_type}, {len(image.data)} bytes]' for image in message.images)
            parts.append(f'[{message.role}]\n{message.text}{images}')
        return '\n\n'.join(parts)

    def _log(self, entry, response='', error=None):
        if self.log_sink is None:
            return
        try:
            usage = entry['usage']
            entry['ctx'].run(self.log_sink, {
                'role': entry['role'], 'profile': entry['profile'], 'model': entry['model'],
                'render_request': lambda: self._render(entry['system'], entry['messages']), 'response_text': response or '',
                'input_tokens': usage.input_tokens if usage else None, 'output_tokens': usage.output_tokens if usage else None,
                'status': 'error' if error else 'ok', 'error_detail': error_detail(error) if error else ''})
        except Exception as sink_error:
            logging.warning('Turn log entry could not be saved: %s', type(sink_error).__name__)

    def _usage(self, profile_name, role, usage, entry=None, profile=None):
        record = collect_usage(profile_name, profile or self.settings.profiles[profile_name], role, usage)
        if entry is not None:
            entry['usage'] = record
        if self.usage_sink:
            try:
                self.usage_sink(record)
            except Exception as error:
                logging.warning('Model usage could not be saved: %s', type(error).__name__)

    def _client(self, profile_name: str, profile: ModelProfile | None = None):
        profile = profile or self.settings.profiles[profile_name]
        dashboard = profile.source == 'dashboard'
        environ = getattr(self.source, 'environ', os.environ)  # only the dashboard's one-off connection test supplies its own
        if dashboard:
            try:
                check_key_pin(profile, self.source.key_hosts)
            except ValueError as error:
                raise ModelConfigError(f'Profile {profile_name}: {error}') from None
            if profile.api_key_env and not environ.get(profile.api_key_env):
                raise ModelConfigError(f"{profile.api_key_env} is not set in the bot's environment (profile {profile_name})")
        fingerprint = (profile.provider, profile.base_url, profile.api_key_env, profile.timeout_seconds, profile.max_retries, profile.source)
        if profile_name in self.clients and self.fingerprints.get(profile_name, fingerprint) == fingerprint:
            return self.clients[profile_name]
        key = environ.get(profile.api_key_env or "", "local-no-key")
        if profile.provider == "anthropic":
            kwargs = {"api_key": key, "timeout": profile.timeout_seconds, "max_retries": profile.max_retries}
            if dashboard:
                kwargs["base_url"] = 'https://api.anthropic.com'
            client = AsyncAnthropic(**kwargs)
        else:
            kwargs = {"api_key": key, "timeout": profile.timeout_seconds, "max_retries": profile.max_retries}
            if profile.base_url:
                kwargs["base_url"] = profile.base_url
            elif dashboard and profile.provider == "openai":
                kwargs["base_url"] = 'https://api.openai.com/v1'
            client = AsyncOpenAI(**kwargs)
        if profile_name in self.clients:
            self.retired.append(self.clients[profile_name])
        self.clients[profile_name], self.fingerprints[profile_name] = client, fingerprint
        return client

    def compiled_input(self, role, request, settings=None):
        if (settings or self.settings).profile(role).provider == 'anthropic':
            return '\n\n'.join(m.text for m in request.messages if m.role == 'system'), [m for m in request.messages if m.role != 'system']
        return '', request.messages

    async def text_compiled(self, role, request, max_tokens=None):
        settings = self.settings
        system, messages = self.compiled_input(role, request, settings)
        return await self.text(role, system, messages, max_tokens, settings)

    async def stream_compiled(self, role, request):
        settings = self.settings
        system, messages = self.compiled_input(role, request, settings)
        async with contextlib.aclosing(self.stream_text(role, system, messages, None, settings)) as stream:
            async for delta in stream:
                yield delta

    async def structured_compiled(self, role, request, schema_name, schema):
        settings = self.settings
        system, messages = self.compiled_input(role, request, settings)
        return await self.structured(role, system, messages, schema_name, schema, settings)

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

    async def text(self, role: str, system: str, messages: list[TurnMessage], max_tokens: int | None = None, settings: Settings | None = None) -> str:
        self._check_budget()
        settings = settings or self.settings
        entry = self._entry(role, system, messages, settings)
        try:
            result = await self._text(role, system, messages, max_tokens, entry, settings)
        except Exception as error:
            self._log(entry, entry['raw'], error)
            raise
        self._log(entry, result)
        return result

    async def _text(self, role, system, messages, max_tokens, entry, settings) -> str:
        profile_name = getattr(settings, role)
        profile = settings.profiles[profile_name]
        client = self._client(profile_name, profile)
        limit = max_tokens or settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            response = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit)
            self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
            return response.output_text or ""
        if profile.provider == "anthropic":
            response = await client.messages.create(model=profile.model, system=system,
                messages=self._anthropic_input(messages), max_tokens=limit)
            self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
            return "".join(block.text for block in response.content if block.type == "text")
        response = await client.chat.completions.create(model=profile.model,
            messages=([{"role": "system", "content": system}] if system else []) + self._chat_input(messages), max_tokens=limit,
            **({"reasoning_effort": profile.reasoning_effort} if profile.reasoning_effort else {}))
        self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
        return response.choices[0].message.content or ""

    async def stream_text(self, role: str, system: str, messages: list[TurnMessage], max_tokens: int | None = None, settings: Settings | None = None) -> AsyncIterator[str]:
        self._check_budget()
        settings = settings or self.settings
        entry = self._entry(role, system, messages, settings)
        pieces, failure = [], None
        inner = self._stream(role, system, messages, max_tokens, entry, settings)
        try:
            async for delta in inner:
                pieces.append(delta)
                yield delta
        except (asyncio.CancelledError, GeneratorExit) as error:
            failure = RuntimeError(f'stream closed before completion ({type(error).__name__})')
            raise
        except Exception as error:
            failure = error
            raise
        finally:
            await inner.aclose()  # runs the usage recording so the entry carries this call's tokens
            self._log(entry, ''.join(pieces), failure)

    async def _stream(self, role, system, messages, max_tokens, entry, settings) -> AsyncIterator[str]:
        profile_name = getattr(settings, role)
        profile = settings.profiles[profile_name]
        client = self._client(profile_name, profile)
        limit = max_tokens or settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            stream = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit, stream=True)
            usage = None
            try:
                async for event in stream:
                    if event.type in {'response.completed', 'response.incomplete', 'response.failed'}:
                        usage = getattr(getattr(event, 'response', None), 'usage', None)
                    if event.type == "response.output_text.delta" and event.delta:
                        yield event.delta
            finally:
                self._usage(profile_name, role, usage, entry, profile)
        elif profile.provider == "anthropic":
            usage = None
            try:
                async with client.messages.stream(model=profile.model, system=system,
                    messages=self._anthropic_input(messages), max_tokens=limit) as stream:
                    async for chunk in stream.text_stream:
                        yield chunk
                    usage = (await stream.get_final_message()).usage
            finally:
                self._usage(profile_name, role, usage, entry, profile)
        else:
            stream = await client.chat.completions.create(model=profile.model,
                messages=([{"role": "system", "content": system}] if system else []) + self._chat_input(messages),
                max_tokens=limit, stream=True,
                **({'stream_options': {'include_usage': True}} if profile.stream_usage else {}),
                **({"reasoning_effort": profile.reasoning_effort} if profile.reasoning_effort else {}))
            usage = None
            try:
                async for chunk in stream:
                    if getattr(chunk, 'usage', None) is not None:
                        usage = chunk.usage
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
            finally:
                self._usage(profile_name, role, usage, entry, profile)

    async def structured(self, role: str, system: str, messages: list[TurnMessage], schema_name: str, schema: dict, settings: Settings | None = None) -> dict:
        self._check_budget()
        settings = settings or self.settings
        entry = self._entry(role, system, messages, settings)
        try:
            result = await self._structured(role, system, messages, schema_name, schema, entry, settings)
        except Exception as error:
            if not entry.get('logged'):
                self._log(entry, entry['raw'], error)
            raise
        if not entry.get('logged'):
            self._log_result(entry, result)
        return result

    def _log_result(self, entry, result):
        if isinstance(result, dict) and isinstance(result.get('personal_facts'), list):
            mask_for_log(*result['personal_facts'])
        self._log(entry, json.dumps(result, ensure_ascii=False))

    async def _structured(self, role, system, messages, schema_name, schema, entry, settings) -> dict:
        profile_name = getattr(settings, role)
        profile = settings.profiles[profile_name]
        client = self._client(profile_name, profile)
        limit = settings.limits["max_output_tokens"]
        if profile.provider == "openai":
            response = await client.responses.create(model=profile.model, instructions=system,
                input=self._openai_input(messages), max_output_tokens=limit,
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}})
            self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
            entry['raw'] = response.output_text
            return validate_result(json.loads(response.output_text), schema)
        if profile.provider == "anthropic":
            response = await client.messages.create(model=profile.model, system=system,
                messages=self._anthropic_input(messages), max_tokens=limit,
                tools=[{"name": schema_name, "description": "Return the requested structured result", "input_schema": schema}],
                tool_choice={"type": "tool", "name": schema_name})
            self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
            for block in response.content:
                if block.type == "tool_use" and block.name == schema_name:
                    entry['raw'] = json.dumps(block.input, ensure_ascii=False, default=str)
                    return validate_result(block.input, schema)
            raise ValueError("Model returned no structured result")
        if profile.structured_outputs:
            response = await client.chat.completions.create(model=profile.model,
                messages=([{"role": "system", "content": system}] if system else []) + self._chat_input(messages),
                max_tokens=limit,
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
                **({"reasoning_effort": profile.reasoning_effort} if profile.reasoning_effort else {}))
            self._usage(profile_name, role, getattr(response, 'usage', None), entry, profile)
            choice = response.choices[0]
            entry['raw'] = choice.message.content
            try:
                return validate_result(json.loads(choice.message.content or ''), schema)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Model returned invalid {schema_name} JSON (finish reason: {choice.finish_reason})") from error
        instruction = f"{system}\nReturn only JSON matching this schema: {json.dumps(schema)}"
        for attempt in range(2):
            entry['logged'] = True
            self._check_budget()
            attempt_entry = self._entry(role, instruction, messages, settings)
            try:
                raw = await self._text(role, instruction, messages, limit, attempt_entry, settings)
            except Exception as error:
                self._log(attempt_entry, attempt_entry['raw'], error)
                raise
            try:
                value = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
                if isinstance(value, dict):
                    result = validate_result(value, schema)
                    self._log_result(attempt_entry, result)
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
            self._log(attempt_entry, raw, ValueError("Compatible model returned invalid JSON") if attempt else None)
            instruction += "\nYour previous result did not match the required JSON schema. Try again."
        raise ValueError("Compatible model returned invalid JSON")

    async def close(self) -> None:
        for client in [*self.clients.values(), *self.retired]:
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
