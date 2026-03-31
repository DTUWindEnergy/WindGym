#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# =============================================================================
# WindGym Training Script: Iterative Self-Play (SP Race) - Final
# =============================================================================

# --- Configuration ---
MAX_ITERATIONS=4
SP_TIMESTEPS_PER_ITER=500000
PROCEDURAL_START=false
# IMPORTANT: Update this path to your protagonist trained on procedural noise (.zip file)
PROCEDURAL_PROT_PATH="models/protagonist_training/YOUR_PROCEDURAL_MODEL_ID/final_model.zip" 

MODELS_DIR="models/self_play"
CONFIG_PATH="env_config/two_turbine_yaw.yaml"
SEED=42

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --procedural-start) PROCEDURAL_START=true; shift ;;
        -h|--help) echo "Usage: $0 [--procedural-start]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# --- Dynamic Run Name (Consistent across iterations) ---
if [ "$PROCEDURAL_START" = true ]; then
    RUN_NAME="SP_Race_ProcStart_$(date +%s)"
else
    RUN_NAME="SP_Race_ZeroStart_$(date +%s)"
fi

# --- Helper Functions ---
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

# Finds the latest iteration number in a given run directory by parsing filenames
find_latest_iter() {
    local run_name="$1"
    local models_dir="$2"
    local latest_iter=-1

    # Scan for files matching the run name and parse the iteration number
    for f in "$models_dir"/iter_*_protagonist_${run_name}.pt; do
        # Check if any files were found to avoid errors with glob
        [ -e "$f" ] || continue
        
        # Extract the iteration number using Bash pattern matching
        if [[ $(basename "$f") =~ iter_([0-9]+)_protagonist_${run_name}\.pt ]]; then
            iter_num="${BASH_REMATCH[1]}"
            # BASH treats variables as strings, so we must force integer comparison
            if (( 10#$iter_num > latest_iter )); then
                latest_iter=$((10#$iter_num)) # Use 10# to prevent octal interpretation
            fi
        fi
    done
    
    echo "$latest_iter"
}


# --- Main Logic ---
log_info "Starting Iterative Self-Play Pipeline with Run Name: ${RUN_NAME}"
# Note: The Python script now handles creating the directory

# --- Hot Start: Detect if we are resuming a run ---
LATEST_ITER=$(find_latest_iter "${RUN_NAME}" "${MODELS_DIR}")
CURRENT_ITER=$((LATEST_ITER + 1))

if (( CURRENT_ITER > 0 )); then
    log_info "Hot Start Detected: Resuming training from iteration ${CURRENT_ITER}."
else
    log_info "Cold Start: Beginning new run from iteration 0."
fi

for ((i=CURRENT_ITER; i<MAX_ITERATIONS; i++)); do
    log_info ""
    log_info "╔═══════════════════════════════════════╗"
    log_info "║   Self-Play Iteration: $((i + 1)) / ${MAX_ITERATIONS}"
    log_info "╚═══════════════════════════════════════╝"

    # Assemble the command-line arguments for the python script
    CMD_ARGS=()
    CMD_ARGS+=(--run-name ${RUN_NAME})
    CMD_ARGS+=(--iteration ${i})
    CMD_ARGS+=(--total-timesteps ${SP_TIMESTEPS_PER_ITER})
    CMD_ARGS+=(--yaml-config-path ${CONFIG_PATH})
    CMD_ARGS+=(--seed $((SEED + i)))

    if [[ ${i} -eq 0 ]]; then
        # --- First Iteration Logic ---
        if [ "$PROCEDURAL_START" = true ]; then
            log_info "Procedural Start: Loading pre-trained SB3 protagonist."
            CMD_ARGS+=(--protagonist-path "${PROCEDURAL_PROT_PATH}")
            CMD_ARGS+=(--protagonist-is-sb3)
        else
            log_info "Zero Start: Training both agents from scratch."
        fi
    else
        # --- Subsequent Iterations Logic (Resuming) ---
        PREV_ITER=$((i - 1))
        # Construct paths based on the new saving format
        PREV_PROT_PATH="${MODELS_DIR}/iter_$(printf "%03d" ${PREV_ITER})_protagonist_${RUN_NAME}.pt"
        PREV_ANT_PATH="${MODELS_DIR}/iter_$(printf "%03d" ${PREV_ITER})_antagonist_${RUN_NAME}.pt"
        
        log_info "Resuming from iteration ${PREV_ITER} models."
        CMD_ARGS+=(--protagonist-path "${PREV_PROT_PATH}")
        CMD_ARGS+=(--antagonist-path "${PREV_ANT_PATH}")
    fi

    log_info "Executing command..."
    python train_self_play.py "${CMD_ARGS[@]}"
    
    if [[ $? -ne 0 ]]; then
        log_info "ERROR: Training failed at iteration ${i}. Exiting."
        exit 1
    fi
done

log_info "Iterative Self-Play Training Complete!"
