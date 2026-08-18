#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-.}"
exec python -m uvicorn services.fusion_service.app:app --host 0.0.0.0 --port 8000 "$@"
