#!/usr/bin/env bash
# The whole OGBench pipeline (data -> world models + DP -> planning evaluation -> figures) for one
# environment, with the per-environment settings in one place.
#
#   bash scripts/ogb_suite.sh ENV STAGE [GPU]
#
# ENV    cube-single | cube-double | cube-triple | scene | puzzle-3x3 | puzzle-4x4 | puzzle-4x5 | puzzle-4x6
#        (optionally with a dataset type: cube-double:noisy; default play)
# STAGE  data      download the OGBench dataset and convert it to our cache
#        viz       look at the cache (frames, episode GIF, stats, timeline, the 5 evaluation tasks)
#        wm-sig | wm-gcb | wm-sf    train one world model (run the three in parallel, one GPU each)
#        dp        train the goal-conditioned Diffusion Policy
#        eval      the 5 tasks, nominal conditions (policy, consensus, each world model, SF value)
#        disturb   the 5 tasks under execution disturbances (push only where there is a cube)
#        video     side-by-side GIFs per task + frame strips for SF (value)
#        analysis  latent-space analyses (PC error curves, decoded state, open-loop prediction)
#        all-wm    wm-sig, wm-gcb, wm-sf and dp one after the other on one GPU
# GPU    CUDA device index (default 0)
#
# Paths (override with environment variables): OUT_ROOT (root of runs/, eval/, figures/; default
# the repo), DATA_DIR (OGBench .npz files), CACHE_ROOT (caches),
# RUNS (runs/ogb/<ENV>), EVAL (eval/ogb_<ENV>), FIG (figures/ogb_<ENV>), EPISODES (per task, 20),
# EXTRA_SET="training.lambda_state=0.5 ..." for more config overrides,
# PROP=1 for the proprio variant (configs/ogb/*_prop.yaml, runs and outputs suffixed _prop).
set -euo pipefail
ENVKEY=${1:?ENV}; STAGE=${2:?STAGE}; GPU=${3:-0}
NAME=${ENVKEY%%:*}; DTYPE=play; [[ $ENVKEY == *:* ]] && DTYPE=${ENVKEY#*:}
export MUJOCO_GL=${MUJOCO_GL:-egl} CUDA_VISIBLE_DEVICES=$GPU

ENV=visual-$NAME-v0
DATASET=visual-$NAME-$DTYPE-v0
DATA_DIR=${DATA_DIR:-/mnt/t7shield/ogbench_data}
CACHE_ROOT=${CACHE_ROOT:-/mnt/t7shield}
TAG=${NAME//-/_}$([[ $DTYPE != play ]] && echo _$DTYPE || true)
CACHE=$CACHE_ROOT/cache_ogb_$TAG
SUF=$([[ ${PROP:-0} == 1 ]] && echo _prop || true)
# STATE=robot | all: auxiliary readout of the state from z (effector + gripper; "all" adds the
# objects, privileged in training only); runs and outputs suffixed _state<robot|all>
STATE=${STATE:-none}
[[ $STATE != none ]] && SUF=${SUF}_state$STATE
OUT_ROOT=${OUT_ROOT:-.}       # e.g. /mnt/t7shield to keep runs, evaluations and figures off the home disk
RUNS=${RUNS:-$OUT_ROOT/runs/ogb/$TAG$SUF}
# the DP always sees proprio, so PROP=1 reuses the DP of the pixel runs
DP_DIR=${DP_DIR:-$OUT_ROOT/runs/ogb/$TAG/${DP_NAME:-dp}}
EVAL=${EVAL:-$OUT_ROOT/eval/ogb_$TAG$SUF}
FIG=${FIG:-$OUT_ROOT/figures/ogb_$TAG$SUF}
EPISODES=${EPISODES:-20}

# ---- per-environment settings --------------------------------------------------------------------
# L: OGBench's evaluation episode length (env steps). The world models see 1 latent step = 2 env
# steps, so hindsight goals reach up to L/2 latent steps (half from a uniform draw, as for
# cube-single where L = 200 -> 100), and the SF discount gives a value horizon 1/(1-gamma) of about
# L/4 latent steps (cube-single: 0.98 -> 50).
case $NAME in
  cube-single)              L=200;  GAMMA=0.98 ;;
  cube-double|puzzle-3x3|puzzle-4x4) L=500;  GAMMA=0.99 ;;
  scene)                    L=750;  GAMMA=0.993 ;;
  cube-triple|cube-quadruple|puzzle-4x5|puzzle-4x6) L=1000; GAMMA=0.995 ;;
  *) echo "unknown environment $NAME"; exit 1 ;;
esac
GOAL_MAX=$((L / 2))
# DP hindsight goals: offset ~ Geometric(p) env steps with mean 1/p = L/2 (cube-single: 0.01 -> 100)
DP_GOAL_P=$(python -c "print(round(2 / $L, 5))")
PUSH=1; [[ $NAME == puzzle-* ]] && PUSH=0      # nothing to push in the puzzles (buttons only)

# training windows: every 4th frame for 1M frames (cube-single), scaled with the dataset size so an
# epoch costs the same (OGBench's larger datasets have 3M-5M frames)
stride() {
  python - "$CACHE" <<'EOF'
import json, sys
n = json.load(open(sys.argv[1] + '/meta.json'))['num_frames']
print(max(4, round(4 * n / 1e6)))
EOF
}

MODELS=(--model "MSP+SIG=$RUNS/msp_sigreg" --model "MSP+GCB+SIG=$RUNS/msp_gcb_sigreg"
        --model "MSP+SIG+SF=$RUNS/msp_sigreg_sf")
METHODS="policy,MSP+SIG,MSP+GCB+SIG,MSP+SIG+SF (value)"

train_wm() {   # $1 = config name
  local s; s=$(stride)
  mkdir -p "$RUNS"
  python -m jepa.train --cache "$CACHE" --config configs/ogb/$1$SUF.yaml --save-dir "$RUNS/$1" \
    --set data.goal_max_offset=$GOAL_MAX data.goal_uniform_max=$GOAL_MAX training.sf_gamma=$GAMMA \
          data.start_stride=$s data.val_start_stride=$((4 * s)) model.state_head=$STATE \
          ${EXTRA_SET:-} 2>&1 | tee "$RUNS/$1.log"
}

mkdir -p "$RUNS" "$(dirname "$EVAL")" "$FIG"
case $STAGE in
  data)
    python -m jepa.ogb_data download --dataset "$DATASET" --dir "$DATA_DIR"
    python -m jepa.ogb_data convert --dataset "$DATASET" --dir "$DATA_DIR" --out "$CACHE" ;;
  viz)
    python -m jepa.ogb_viz --cache "$CACHE" --out "$FIG/data_viz" ;;
  wm-sig) train_wm msp_sigreg ;;
  wm-gcb) train_wm msp_gcb_sigreg ;;
  wm-sf)  train_wm msp_sigreg_sf ;;
  dp)
    mkdir -p "$(dirname "$DP_DIR")"
    STEPS=${DP_STEPS:-$([[ $L -ge 750 ]] && echo 100000 || echo 60000)}
    python -m jepa.dp_train --cache "$CACHE" --out "$DP_DIR" --goal-cond --action-mode stored --amp \
      --goal-p "${DP_GOAL_P_OVERRIDE:-$DP_GOAL_P}" --steps "$STEPS" 2>&1 | tee "$DP_DIR.log" ;;
  all-wm)
    for s in wm-sig wm-gcb wm-sf dp; do bash "$0" "$ENVKEY" $s "$GPU"; done ;;
  eval)
    mkdir -p "$EVAL" "$(dirname "$EVAL")"
    python -m jepa.ogb_eval --env "$ENV" --dp "$DP_DIR" --out "$EVAL" --episodes-per-task "$EPISODES" \
      "${MODELS[@]}" 2>&1 | tee "$EVAL.log" ;;
  disturb)
    CONDS="noise:0.2 noise:0.4 burst:0.03"; [[ $PUSH == 1 ]] && CONDS="push:0.08 $CONDS"
    for D in $CONDS; do
      python -m jepa.ogb_eval --env "$ENV" --dp "$DP_DIR" --out "${EVAL}_disturb_${D/:/_}" --disturb "$D" \
        --episodes-per-task "$EPISODES" --methods "$METHODS" "${MODELS[@]}" 2>&1 | tee "${EVAL}_disturb_${D/:/_}.log"
    done ;;
  video)
    python -m jepa.ogb_video --env "$ENV" --dp "$DP_DIR" --eval-dir "$EVAL" --methods "$METHODS" \
      "${MODELS[@]}" --prefer "MSP+SIG+SF (value)" --tasks 1,2,3,4,5
    python -m jepa.ogb_video --env "$ENV" --dp "$DP_DIR" --eval-dir "$EVAL" --methods "MSP+SIG+SF (value)" \
      --strip "MSP+SIG+SF (value)" "${MODELS[@]}" --tasks 1,2,3,4,5 ;;
  analysis)
    python -m jepa.ogb_analysis --cache "$CACHE" --out "$FIG/latent" "${MODELS[@]}" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
