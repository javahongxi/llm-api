#!/usr/bin/env bash
# One-command start for the llm-api proxy using the shipped template config.
# All api_key values in config.example.yaml are os.environ/VAR references, so
# export the provider keys you need first, e.g.:
#   export OPENAI_API_KEY=sk-...
#   export ANTHROPIC_API_KEY=sk-...
#   export LLM_API_MASTER_KEY=sk-my-master-key   # optional: enables auth/budgets
#
# Extra args are forwarded to the binary, e.g.:
#   ./run.sh --port 8080 --host 0.0.0.0
set -euo pipefail
cd "$(dirname "$0")"
exec go run ./cmd/llm-api --config config.example.yaml "$@"
