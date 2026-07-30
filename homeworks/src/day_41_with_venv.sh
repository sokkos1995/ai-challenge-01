#!/usr/bin/env bash
# Activate project venv, then run the given command from repo root.
# Usage:
#   ./homeworks/src/day_41_with_venv.sh python homeworks/src/day_41_baseline.py --n 10
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
cd "${ROOT}"
exec "$@"
