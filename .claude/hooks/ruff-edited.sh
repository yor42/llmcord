#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): run ruff check on the edited file only, if it is a .py file.
# Exit 2 feeds ruff's findings back to Claude; anything else is silent and non-blocking.
set -u

project="${CLAUDE_PROJECT_DIR:-$(pwd)}"
python="$project/.venv/bin/python"
[ -x "$python" ] || python=python3

file=$("$python" -c 'import json, sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    pass' 2>/dev/null)

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

cd "$project" || exit 0
# --force-exclude keeps ruff.toml's extend-exclude (data/, .venv, ...) in effect for explicit paths.
if ! out=$(env -u FORCE_COLOR -u CLICOLOR_FORCE NO_COLOR=1 "$python" -m ruff check --force-exclude --quiet --output-format=concise --no-cache "$file" 2>&1); then
  printf 'ruff check failed for %s:\n%s\n' "$file" "$out" >&2
  exit 2
fi
exit 0
