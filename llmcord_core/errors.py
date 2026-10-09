"""Actionable error details without credentials or full provider response bodies."""
from __future__ import annotations

import os
import re
import secrets
import sys
import traceback

TIMEOUT_TYPES = (('httpx', 'TimeoutException'), ('openai', 'APITimeoutError'), ('anthropic', 'APITimeoutError'))


LOOSE_CREDENTIAL = re.compile(r'(?i)\b(?:Bearer|Bot)\s+[^\s,;]+')
STRICT_CREDENTIAL = re.compile(r'(?i:Authorization)\s*:\s*(?:(?:Bearer|Bot|Basic)\s+)?[^\s,;]+|\b(?:Bearer|Bot) [A-Za-z0-9._~+/=-]{20,}')


def redact(text: str, extra_values=(), strict: bool = False) -> str:
    # Secret env values, caller-supplied values and credentials; strict only matches token-shaped ones so prose survives.
    values = [v for n, v in os.environ.items() if len(v) >= 8 and n.endswith(('_TOKEN', '_KEY', '_SECRET'))]
    values += [v for v in extra_values if v]
    for value in sorted(set(values), key=len, reverse=True):
        text = text.replace(value, '[redacted]')
    return (STRICT_CREDENTIAL if strict else LOOSE_CREDENTIAL).sub('[credential omitted]', text)


def mask_facts(text: str, facts) -> str:
    facts = sorted({f.strip() for f in facts if f and f.strip()}, key=len, reverse=True)
    if not facts:
        return text
    return re.sub('|'.join(map(re.escape, facts)), '[personal fact hidden]', text, flags=re.IGNORECASE)


def error_detail(error: Exception, limit: int = 1000) -> str:
    body = getattr(error, 'body', None)
    if isinstance(body, dict):
        payload = body.get('error', body)
        message = payload.get('message') if isinstance(payload, dict) else None
        message = message or 'Provider request failed'
    else:
        message = getattr(error, 'text', None) or str(error)
    message = str(message)
    message = redact(message)
    message = re.sub(r'https?://\S+', '[URL omitted]', message)
    message = ' '.join(message.split())
    status = getattr(error, 'status_code', None) or getattr(error, 'status', None)
    code = getattr(error, 'code', None)
    label = type(error).__name__
    if status:
        label += f' HTTP {status}'
    if code:
        label += f' code {code}'
    return f'{label}: {message}'[:limit]


def error_stack(error: Exception) -> str:
    # Frame locations aid debugging; omit exception bodies and local variables.
    return ''.join(traceback.format_list(traceback.extract_tb(error.__traceback__)))


def reference_id() -> str:
    # Short token shown to users and logged with the detail so an admin can match the two.
    return secrets.token_hex(3)


def is_timeout(error: Exception) -> bool:
    # Only modules already loaded can have raised their timeout types; avoids importing optional SDKs.
    for module, name in TIMEOUT_TYPES:
        kind = getattr(sys.modules.get(module), name, None)
        if isinstance(kind, type) and isinstance(error, kind):
            return True
    return False


def user_detail(error: Exception) -> str:
    return 'The model provider did not respond in time' if is_timeout(error) else error_detail(error)
