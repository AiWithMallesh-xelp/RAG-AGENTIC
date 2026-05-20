#!/usr/bin/env bash
# Run CLI from repo root
cd "$(dirname "$0")/agentic_rag"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
if [[ "${FRONTEND:-}" == "true" ]] && [[ $# -eq 0 ]]; then
  exec .venv/bin/python app.py
fi
exec .venv/bin/python main.py "$@"
