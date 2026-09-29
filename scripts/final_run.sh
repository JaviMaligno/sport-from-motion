#!/usr/bin/env bash
# Final model run, pre-registered in docs/preregistration.md. Do not change the cells,
# N or the models after the first call without amending (and dating) that document.
#
# One background subshell per model. Order per model: the 5 primary cells, then the
# pre-registered secondaries (kinematics, kinematics_solo, trails, video for Gemini,
# replicates), then A7b last. If the run is cut, what is missing is the least important.
# Resumable: re-running skips answered items and retries errored / unparseable ones
# (every cell is run PASSES times; later passes only redo what failed).
# Logs in $ITEMS/logs/<model>.log; $ITEMS/plan.json and $D8_ITEMS/plan.json hold the planned
# rows per prediction file (report and the error summary flag cells with missing rows); an
# error summary at the end.
#
# Bookkeeping (preflight, plan.json, logs dir) runs under `set -e -o pipefail`: if any of it
# fails the script exits non-zero before the first model call. The per-model subshells run
# with `set +e`: a failing command there is a cell with errors, which the next pass
# (PASSES) or a relaunch retries, and the error summary reports.
#
# A7b (deviation D7, exploratory, outside the Holm family): every chat model also answers
# motion/sheet and motion_shuffled/sheet on the 8 s items ($D8_ITEMS, 3 sports, first ND8
# clips), after all its other cells. Jev (text only) has no A7b cell.
#
#   DRY_RUN=1 bash scripts/final_run.sh        # print the commands, touch nothing
#   bash scripts/final_run.sh                  # the run
#   MODELS="vertex:gemini-3.1-pro-preview" bash scripts/final_run.sh   # one model
set -euo pipefail
cd "$(dirname "$0")/.."
ITEMS=${ITEMS:-runs/final}
N=${N:-400}            # first N clips of the interleaved order (100 per sport)
NREP=${NREP:-200}      # replicates 2 and 3: first NREP clips
D8_ITEMS=${D8_ITEMS:-runs/final-d8}   # A7b: 8 s clips, 3 sports
ND8=${ND8:-300}        # A7b: first ND8 clips of its interleaved order (100 per sport)
WORKERS=${WORKERS:-6}
GEMINI_WORKERS=${GEMINI_WORKERS:-12}
PASSES=${PASSES:-2}
DRY_RUN=${DRY_RUN:-0}
MODELS=${MODELS:-"azure-openai:gpt-5.6-sol azure-openai:gpt-5.6-terra vertex-anthropic:claude-sonnet-5 vertex-anthropic:claude-opus-5-5 vertex:gemini-3.1-pro-preview jev-openrouter:~typesafe/jev-latest"}

# cell = condition:repr:prompt_style:replicate:limit[:items]  (keep in sync with
# scripts/estimate_run.py); items defaults to $ITEMS
PRIMARY_CELLS="motion:sheet:neutral:1:$N motion_shuffled:sheet:neutral:1:$N formation:sheet:neutral:1:$N motion:text:neutral:1:$N motion:sheet:informed:1:$N"
A7B_CELLS="motion:sheet:neutral:1:$ND8:$D8_ITEMS motion_shuffled:sheet:neutral:1:$ND8:$D8_ITEMS"   # chat models only
SECONDARY_CELLS="kinematics:sheet:neutral:1:$N kinematics_solo:sheet:neutral:1:$N motion:trails:neutral:1:$N"
VIDEO_CELLS="motion:video:neutral:1:$N motion_shuffled:video:neutral:1:$N"   # vertex: (Gemini) only
REPLICATE_CELLS="motion:sheet:neutral:2:$NREP motion_shuffled:sheet:neutral:2:$NREP motion:sheet:neutral:3:$NREP motion_shuffled:sheet:neutral:3:$NREP"
JEV_CELLS="motion:text:neutral:1:$N motion_shuffled:text:neutral:1:$N formation:text:neutral:1:$N motion:text:informed:1:$N kinematics:text:neutral:1:$N kinematics_solo:text:neutral:1:$N"

# Azure OpenAI: key read from a file (AZURE_OPENAI_KEY_FILE), never printed
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://australiaeast.api.cognitive.microsoft.com}
# Vertex
export GCLOUD_BIN=${GCLOUD_BIN:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
export VERTEX_PROJECT=${VERTEX_PROJECT:-}  # your GCP project; Vertex routes fail clearly if unset
if [ "$DRY_RUN" != 1 ]; then  # keys are read only for a real run, and never echoed
  [ -z "${AZURE_OPENAI_KEY:-}" ] && [ -f "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}" ] && export AZURE_OPENAI_KEY="$(tr -d '\n' < "${AZURE_OPENAI_KEY_FILE:-$HOME/.azure-openai-key}")"
  [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f ~/.openrouter-key ] && export OPENROUTER_API_KEY="$(tr -d '\n' < ~/.openrouter-key)"
fi

cells_for() {  # $1 = model id
  case "$1" in
    jev*|laya*) echo "$JEV_CELLS" ;;
    vertex:*) echo "$PRIMARY_CELLS $SECONDARY_CELLS $VIDEO_CELLS $REPLICATE_CELLS $A7B_CELLS" ;;
    *) echo "$PRIMARY_CELLS $SECONDARY_CELLS $REPLICATE_CELLS $A7B_CELLS" ;;
  esac
}

commands_for() {  # one `motion-sport run` per line, in run order
  local m=$1 w=$WORKERS fmts="-" cell cond rep style k lim items f
  case "$m" in vertex:*) w=$GEMINI_WORKERS ;; esac
  case "$m" in jev*|laya*) fmts="text json" ;; esac
  for cell in $(cells_for "$m"); do
    IFS=: read -r cond rep style k lim items <<< "$cell"
    for f in $fmts; do
      printf '%s' ".venv/bin/motion-sport run --items ${items:-$ITEMS} --model $m --condition $cond --repr $rep --limit $lim --workers $w"
      [ "$style" != neutral ] && printf ' --prompt-style %s' "$style"
      [ "$k" != 1 ] && printf ' --replicate %s' "$k"
      [ "$f" != - ] && printf ' --state-format %s' "$f"
      printf '\n'
    done
  done
}

preflight() {  # the items must be the pre-registered set (scripts/run_plan.py preflight):
  # strict_smooth, N=10 random players, 20 frames, smooth=2.0, max_speed_ms=12.0 exactly,
  # the 4 sports as candidates, NFL at a random phase (D1-D2), >= N/4 clips per sport,
  # informed prompts, every planned representation with all its conditions
  .venv/bin/python scripts/run_plan.py preflight "$ITEMS" "$N" "$MODELS" || exit 1
  # A7b (D7): 8 s items (40 frames), 3 sports, same controls pinned exactly,
  # motion/motion_shuffled in sheet
  .venv/bin/python scripts/run_plan.py preflight-a7b "$D8_ITEMS" "$ND8" || exit 1
}

error_summary() {  # unrecovered errors per file (> 2 % flagged) + planned cells with missing rows
  for d in "$ITEMS" "$D8_ITEMS"; do
    echo "# $d"
    .venv/bin/python scripts/run_plan.py check "$d"
  done
}

write_plan() {  # <items>/plan.json: planned rows per prediction file, from these very commands
  # (each directory keeps only the commands that run on it). Any failure -> non-zero.
  # $D8_ITEMS is marked exploratory: its report has no primary contrast and no Holm (D7).
  local d m
  for d in "$ITEMS" "$D8_ITEMS"; do
    for m in $MODELS; do commands_for "$m"; done \
      | .venv/bin/python scripts/run_plan.py write "$d" \
          $([ "$d" = "$D8_ITEMS" ] && echo --exploratory) || return 1
  done
}

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: items=$ITEMS N=$N NREP=$NREP d8_items=$D8_ITEMS ND8=$ND8 WORKERS=$WORKERS GEMINI_WORKERS=$GEMINI_WORKERS PASSES=$PASSES"
  for d in "$ITEMS" "$D8_ITEMS"; do
    [ -f "$d/items.jsonl" ] || echo "# note: $d/items.jsonl does not exist yet (preflight runs only for real)"
  done
  for m in $MODELS; do
    echo "# --- $m -> $ITEMS/logs/${m//[:\/]/_}.log ($(commands_for "$m" | wc -l | tr -d ' ') commands x $PASSES passes)"
    commands_for "$m"
  done
  echo "# calls per pass: $(for m in $MODELS; do commands_for "$m"; done | sed -n 's/.*--limit \([0-9]*\).*/\1/p' | paste -sd+ - | bc)"
  exit 0
fi

preflight
write_plan || { echo "write_plan failed: no model was called" >&2; exit 1; }
mkdir -p "$ITEMS/logs"
for m in $MODELS; do
  (
    set +e  # a failing cell is retried by the next pass / a relaunch, not fatal
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
