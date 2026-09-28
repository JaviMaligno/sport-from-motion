#!/usr/bin/env bash
# Final model run, pre-registered in docs/preregistration.md. Do not change the cells,
# N or the models after the first call without amending (and dating) that document.
#
# One background subshell per model; cells in pre-registered order, primary first.
# Resumable: re-running skips answered items and retries errored / unparseable ones
# (every cell is run PASSES times; later passes only redo what failed).
# Logs in $ITEMS/logs/<model>.log; $ITEMS/plan.json holds the planned rows per prediction
# file (report and the error summary flag cells with missing rows); an error summary at the end.
#
#   DRY_RUN=1 bash scripts/final_run.sh        # print the commands, touch nothing
#   bash scripts/final_run.sh                  # the run
#   MODELS="vertex:gemini-3.1-pro-preview" bash scripts/final_run.sh   # one model
set -u
cd "$(dirname "$0")/.."
ITEMS=${ITEMS:-runs/final}
N=${N:-400}            # first N clips of the interleaved order (100 per sport)
NREP=${NREP:-200}      # replicates 2 and 3: first NREP clips
WORKERS=${WORKERS:-6}
GEMINI_WORKERS=${GEMINI_WORKERS:-12}
PASSES=${PASSES:-2}
DRY_RUN=${DRY_RUN:-0}
MODELS=${MODELS:-"azure-openai:gpt-5.6-sol azure-openai:gpt-5.6-terra vertex-anthropic:claude-sonnet-5 vertex-anthropic:claude-opus-5-5 vertex:gemini-3.1-pro-preview jev-openrouter:~typesafe/jev-latest"}

# cell = condition:repr:prompt_style:replicate:limit  (keep in sync with scripts/estimate_run.py)
PRIMARY_CELLS="motion:sheet:neutral:1:$N motion_shuffled:sheet:neutral:1:$N formation:sheet:neutral:1:$N motion:text:neutral:1:$N motion:sheet:informed:1:$N"
SECONDARY_CELLS="kinematics:sheet:neutral:1:$N kinematics_solo:sheet:neutral:1:$N motion:trails:neutral:1:$N"
VIDEO_CELLS="motion:video:neutral:1:$N motion_shuffled:video:neutral:1:$N"   # vertex: (Gemini) only
REPLICATE_CELLS="motion:sheet:neutral:2:$NREP motion_shuffled:sheet:neutral:2:$NREP motion:sheet:neutral:3:$NREP motion_shuffled:sheet:neutral:3:$NREP"
JEV_CELLS="motion:text:neutral:1:$N motion_shuffled:text:neutral:1:$N formation:text:neutral:1:$N motion:text:informed:1:$N kinematics:text:neutral:1:$N kinematics_solo:text:neutral:1:$N"

# Azure OpenAI: key read from a file (AZURE_OPENAI_KEY_FILE), never printed
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://australiaeast.api.cognitive.microsoft.com}
# Vertex
export GCLOUD_BIN=${GCLOUD_BIN:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
export VERTEX_PROJECT=${VERTEX_PROJECT:?set VERTEX_PROJECT to your GCP project}
if [ "$DRY_RUN" != 1 ]; then  # keys are read only for a real run, and never echoed
  [ -z "${AZURE_OPENAI_KEY:-}" ] && [ -f "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}" ] && export AZURE_OPENAI_KEY="$(tr -d '\n' < "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}")"
  [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f ~/.openrouter-key ] && export OPENROUTER_API_KEY="$(tr -d '\n' < ~/.openrouter-key)"
fi

cells_for() {  # $1 = model id
  case "$1" in
    jev*|laya*) echo "$JEV_CELLS" ;;
    vertex:*) echo "$PRIMARY_CELLS $SECONDARY_CELLS $VIDEO_CELLS $REPLICATE_CELLS" ;;
    *) echo "$PRIMARY_CELLS $SECONDARY_CELLS $REPLICATE_CELLS" ;;
  esac
}

commands_for() {  # one `motion-sport run` per line, in run order
  local m=$1 w=$WORKERS fmts="-" cell cond rep style k lim f
  case "$m" in vertex:*) w=$GEMINI_WORKERS ;; esac
  case "$m" in jev*|laya*) fmts="text json" ;; esac
  for cell in $(cells_for "$m"); do
    IFS=: read -r cond rep style k lim <<< "$cell"
    for f in $fmts; do
      printf '%s' ".venv/bin/motion-sport run --items $ITEMS --model $m --condition $cond --repr $rep --limit $lim --workers $w"
      [ "$style" != neutral ] && printf ' --prompt-style %s' "$style"
      [ "$k" != 1 ] && printf ' --replicate %s' "$k"
      [ "$f" != - ] && printf ' --state-format %s' "$f"
      printf '\n'
    done
  done
}

preflight() {  # the items must be the pre-registered set (scripts/run_plan.py preflight):
  # strict_smooth, N=10 random players, teleport ceiling set, NFL at a random phase (D1-D2),
  # >= N/4 clips per sport, informed prompts and every planned representation
  .venv/bin/python scripts/run_plan.py preflight "$ITEMS" "$N" "$MODELS" || exit 1
}

error_summary() {  # unrecovered errors per file (> 2 % flagged) + planned cells with missing rows
  .venv/bin/python scripts/run_plan.py check "$ITEMS"
}

write_plan() {  # $ITEMS/plan.json: planned rows per prediction file, from these very commands
  for m in $MODELS; do commands_for "$m"; done | .venv/bin/python scripts/run_plan.py write "$ITEMS"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: items=$ITEMS N=$N NREP=$NREP WORKERS=$WORKERS GEMINI_WORKERS=$GEMINI_WORKERS PASSES=$PASSES"
  [ -f "$ITEMS/items.jsonl" ] || echo "# note: $ITEMS/items.jsonl does not exist yet (preflight runs only for real)"
  for m in $MODELS; do
    echo "# --- $m -> $ITEMS/logs/${m//[:\/]/_}.log ($(commands_for "$m" | wc -l | tr -d ' ') commands x $PASSES passes)"
    commands_for "$m"
  done
  exit 0
fi

preflight
write_plan
mkdir -p "$ITEMS/logs"
for m in $MODELS; do
  (
    for pass in $(seq "$PASSES"); do
      echo "=== pass $pass"
      while IFS= read -r cmd; do
        echo "+ $cmd"
        $cmd < /dev/null
      done < <(commands_for "$m")
    done
    echo "DONE $m"
  ) > "$ITEMS/logs/${m//[:\/]/_}.log" 2>&1 &
done
wait
echo "ALL DONE"
error_summary
