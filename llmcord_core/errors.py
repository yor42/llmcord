"""Actionable error details without credentials or full provider response bodies."""
from __future__ import annotations

import os
import re
import traceback


def error_detail(error: Exception, limit: int = 1000) -> str:
    body = getattr(error, 'body', None)
    if isinstance(body, dict):
        payload = body.get('error', body)
        message = payload.get('message') if isinstance(payload, dict) else None
        message = message or 'Provider request failed'
    else:
        message = getattr(error, 'text', None) or str(error)
    message = str(message)
    for name, value in os.environ.items():
        if value and name.endswith(('_TOKEN', '_KEY', '_SECRET')):
            message = message.replace(value, '[redacted]')
    message = re.sub(r'https?://\S+', '[URL omitted]', message)
    message = re.sub(r'(?i)\b(?:Bearer|Bot)\s+[^\s,;]+', '[credential omitted]', message)
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
