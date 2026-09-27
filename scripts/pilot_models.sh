#!/usr/bin/env bash
# Phase-1 pilot: every model on the same 200 clips (100/sport) x 6 cells.
# Resumable: re-running skips done items. Logs in runs/<items>/logs/<model>.log
#   bash scripts/pilot_models.sh runs/pilot-strict 200
set -u
cd "$(dirname "$0")/.."
ITEMS=${1:-runs/pilot-strict}
LIMIT=${2:-200}
WORKERS=${WORKERS:-6}
MODELS=${MODELS:-"azure-openai:gpt-5.6-sol azure-openai:gpt-5.6-terra vertex-anthropic:claude-sonnet-5 vertex:gemini-2.5-pro vertex:gemini-3.1-pro-preview"}
CELLS="motion:sheet motion_shuffled:sheet formation:sheet kinematics:sheet kinematics_solo:sheet motion:text"

# Azure OpenAI: key read from a file (AZURE_OPENAI_KEY_FILE), never printed
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://australiaeast.api.cognitive.microsoft.com}
[ -z "${AZURE_OPENAI_KEY:-}" ] && [ -f "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}" ] && export AZURE_OPENAI_KEY="$(tr -d '\n' < "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}")"
# Vertex
export GCLOUD_BIN=${GCLOUD_BIN:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
export VERTEX_PROJECT=${VERTEX_PROJECT:?set VERTEX_PROJECT to your GCP project}

mkdir -p "$ITEMS/logs"
for m in $MODELS; do
  (
    for cell in $CELLS; do
      .venv/bin/motion-sport run --items "$ITEMS" --model "$m" --condition "${cell%%:*}" \
        --repr "${cell##*:}" --limit "$LIMIT" --workers "$WORKERS"
    done
    echo "DONE $m"
  ) > "$ITEMS/logs/${m//[:\/]/_}.log" 2>&1 &
done
wait
echo "ALL DONE"
