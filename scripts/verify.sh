#!/usr/bin/env bash
# Single verification entry point. Default stages are fast and offline.
#   scripts/verify.sh            lint, repository hygiene, compile, offline unit tests
#   scripts/verify.sh --browser  also the Playwright dashboard suite (~2 min on a Pi 5)
#   scripts/verify.sh --bench    also the dashboard performance bench (~1.5 min)
# A Gitleaks secret scan also runs when $GITLEAKS or `gitleaks` on PATH exists (CI always scans).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
BROWSER=0 BENCH=0
for arg in "$@"; do
  case "$arg" in
    --browser) BROWSER=1 ;;
    --bench) BENCH=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

declare -a SUMMARY
FAILED=0
stage() {
  local name=$1; shift
  echo "==> $name"
  if "$@"; then SUMMARY+=("PASS  $name"); else SUMMARY+=("FAIL  $name"); FAILED=1; fi
}

secret_scan() {
  local gl=$1 rc=0 f
  "$gl" git --redact --no-banner -v || rc=1
  "$gl" git --pre-commit --redact --no-banner -v || rc=1
  "$gl" git --pre-commit --staged --redact --no-banner -v || rc=1
  while IFS= read -r -d '' f; do
    "$gl" dir --redact --no-banner -v "./$f" || rc=1
  done < <(git ls-files --others --exclude-standard -z)
  return $rc
}

stage "ruff (bug-class lint)" "$PY" -m ruff check .
stage "repository hygiene" "$PY" scripts/check_repository.py
stage "compile" "$PY" -m compileall -q llmcord_core scripts tests llmcord.py web_main.py migrate.py
stage "offline unit tests" "$PY" -m unittest discover -s tests
GL=${GITLEAKS:-$(command -v gitleaks || true)}
if [ -n "$GL" ] && [ -x "$GL" ]; then
  stage "secret scan (gitleaks)" secret_scan "$GL"
elif [ -n "${GITLEAKS:-}" ]; then
  SUMMARY+=("FAIL  secret scan (GITLEAKS=$GITLEAKS is not executable)"); FAILED=1
else
  SUMMARY+=("SKIP  secret scan (gitleaks not installed)")
fi
if [ "$BROWSER" = 1 ]; then
  stage "dashboard browser suite" env LLMCORD_BROWSER_TESTS=1 "$PY" -m unittest discover -s tests -p test_dashboard_browser.py
fi
if [ "$BENCH" = 1 ]; then
  stage "dashboard perf bench" "$PY" scripts/bench_dashboard.py
fi

echo
echo "Summary (expected failures in the unit output are documented known defects):"
printf '  %s\n' "${SUMMARY[@]}"
exit $FAILED
