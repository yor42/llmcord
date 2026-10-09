"""Provider-reported usage, task-local attribution, and list-rate estimates."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

_capture = ContextVar('llmcord_model_usage', default=None)


@contextmanager
def capture_usage(guild_id=None, channel_id=None, feature=None):
    parent = _capture.get()
    records = []
    token = _capture.set((guild_id if guild_id is not None else parent[0] if parent else None, records,
                          channel_id if channel_id is not None else parent[2] if parent else None,
                          feature if feature is not None else parent[3] if parent else ''))
    try:
        yield records
    finally:
        _capture.reset(token)


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def count(value):
    return value if type(value) is int and value >= 0 else None


def rates(profile, timestamp):
    if profile.billing_tier == 'free':
        return (0.0, 0.0, 0.0), 'free'
    defaults = (None, None, None)
    if profile.prompt_provider == 'gemini' and profile.model == 'gemini-3.8-flash':
        # Standard paid list rates: https://ai.google.dev/gemini-api/docs/pricing
        factor = 1 if datetime.fromtimestamp(timestamp, timezone.utc).year < 2027 else 2
        defaults = (0.75 * factor, 3.75 * factor, 0.075 * factor)
    inputs = profile.input_cost_per_million if profile.input_cost_per_million is not None else defaults[0]
    outputs = profile.output_cost_per_million if profile.output_cost_per_million is not None else defaults[1]
    cached = profile.cached_input_cost_per_million if profile.cached_input_cost_per_million is not None else defaults[2] if defaults[2] is not None else inputs
    basis = 'configured' if profile.input_cost_per_million is not None or profile.output_cost_per_million is not None else 'paid-list' if defaults[0] is not None else 'unconfigured'
    return (inputs, outputs, cached), basis


@dataclass(frozen=True)
class ModelUsage:
    guild_id: int | None
    profile: str
    model: str
    role: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    cost_basis: str
    created_at: float
    channel_id: int | None = None
    feature: str = ''

    def as_dict(self):
        return asdict(self)


def collect_usage(profile_name, profile, role, usage):
    captured = _capture.get()
    inputs = count(field(usage, 'prompt_tokens', field(usage, 'input_tokens')))
    outputs = count(field(usage, 'completion_tokens', field(usage, 'output_tokens')))
    cached = count(field(field(usage, 'prompt_tokens_details', field(usage, 'input_tokens_details')), 'cached_tokens')) or 0
    reasoning = count(field(field(usage, 'completion_tokens_details', field(usage, 'output_tokens_details')), 'reasoning_tokens')) or 0
    if profile.provider == 'anthropic' and inputs is not None:
        cached = count(field(usage, 'cache_read_input_tokens')) or 0
        inputs += cached + (count(field(usage, 'cache_creation_input_tokens')) or 0)
    cached = min(cached, inputs or 0)
    now = time.time()
    (input_rate, output_rate, cached_rate), basis = rates(profile, now)
    cost = None
    if inputs is not None and outputs is not None and input_rate is not None and output_rate is not None:
        cost = ((inputs - cached) * input_rate + cached * cached_rate + outputs * output_rate) / 1_000_000
    record = ModelUsage(captured[0] if captured else None, profile_name, profile.model, role, inputs, outputs, cached, reasoning, cost, basis, now,
                        captured[2] if captured else None, captured[3] if captured else '')
    if captured:
        captured[1].append(record)
    return record


def reply_footer(model, usage):
    # The caller supplies a markdown-escaped, bounded model label.
    inputs = f'{usage.input_tokens:,}' if usage and usage.input_tokens is not None else 'unavailable'
    outputs = f'{usage.output_tokens:,}' if usage and usage.output_tokens is not None else 'unavailable'
    if usage and usage.output_tokens is not None and usage.reasoning_tokens:
        outputs += f' (incl. {usage.reasoning_tokens:,} thinking)'
    cost = f'${usage.cost_usd:.6f}' if usage and usage.cost_usd is not None else 'unavailable'
    label = 'Est. paid cost' if usage and usage.cost_basis == 'paid-list' else 'Est. cost'
    return f'\n\n-# {model} · Reply input: {inputs} · Output: {outputs} · {label}: {cost}'
