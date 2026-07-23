#!/usr/bin/env bash
set -euo pipefail

if pgrep -f "uvicorn insurance_rag.api:app" >/dev/null 2>&1; then
  exit 0
fi

nohup python -m uvicorn insurance_rag.api:app --host 0.0.0.0 --port 8000 > /tmp/insurance-rag.log 2>&1 &
