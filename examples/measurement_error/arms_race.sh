#!/bin/bash
# ===================================================================================
# WindGym Training Script: Iterative Arms Race (v5 - OS Agnostic)
#
# This script automates an iterative, turn-based training process.
# - It is fully compatible with both Linux and macOS.
# - Uses 'ls' for model discovery, per user feedback.
# ===================================================================================
set -e # Exit immediately if a command exits with a non-zero status.

# --- Default Configuration ---
MODE="ssp" # 'ssp' or 'arms_race'. SSP is the default.
CONFIG_PATH="env_config/two_turbine_yaw.yaml"
SEED=42
PROT_DIR="models/protagonist_training"
ADV_DIR="models/adversaries_stateful"
N_ENVS=6
PROTAGONIST_TIMESTEPS=250000
ADVERSARY_TIMESTEPS=250000
MAX_ITERATIONS=8

# ===================================================================================
# Help and Usage Function
# ===================================================================================
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Modes (choose one):"
    echo "  --ssp                     Run the SSP Arms Race (Default). Trains the protagonist against a pool of all prior adversaries."
    echo "  --arms-race               Run the Naive Arms Race. Trains the protagonist against only the latest adversary."
    echo ""
    echo "General Options:"
    echo "  --n-envs <num>            Number of parallel environments. Default: $N_ENVS."
    echo "  --seed <num>              Master random seed. Default: $SEED."
    echo "  --config-path <path>      Path to the environment YAML config. Default: $CONFIG_PATH."
    echo "  --max-iterations <num>    Total number of training iterations. Default: $MAX_ITERATIONS."
    echo "  --prot-steps <num>        Timesteps for each protagonist training cycle. Default: $PROTAGONIST_TIMESTEPS."
    echo "  --adv-steps <num>         Timesteps for each adversary training cycle. Default: $ADVERSARY_TIMESTEPS."
    echo "  -h, --help                Display this help message."
    exit 1
}

# ===================================================================================
# Helper Functions
# ===================================================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# ---
# OS-AGNOSTIC FIX: These functions now use `ls` which is POSIX-compliant and
# works consistently across Linux and macOS for this specific file structure.
# ---
find_latest_model() {
    local dir="$1"
    local filename="$2"
    log_info "DEBUG: Searching for latest model named '$filename' in '$dir' using 'ls -t'..."

    # Use ls -t to sort by modification time (newest first). The wildcard '*' finds
    # the unique run_id subdirectory. head -n 1 gets the top (most recent) result.
    local latest_model
    latest_model=$(ls -t "$dir"/*/"$filename" 2>/dev/null | head -n 1)

    if [[ -z "$latest_model" ]]; then
        log_info "DEBUG: No models found matching the criteria."
        echo ""
    else
        log_info "DEBUG: Found latest model: $latest_model"
        echo "$latest_model"
    fi
}

count_models() {
    local dir="$1"
    local filename="$2"
    # ls to list all matching files and wc -l to count them.
    local count
    count=$(ls "$dir"/*/"$filename" 2>/dev/null | wc -l | tr -d ' ')
    echo "$count"
}

detect_state() {
    local prot_count=$(count_models "$PROT_DIR" "final_model.zip")
    local adv_count=$(count_models "$ADV_DIR" "final_adversary_model.zip")
    log_info "Detected $prot_count protagonist(s) and $adv_count adversary model(s)."

    local next_agent
    if [[ $prot_count -gt $adv_count ]]; then
        next_agent="adversary"
    else
        next_agent="protagonist"
    fi
    echo "$next_agent"
}

# ===================================================================================
# Main Logic
# ===================================================================================
main() {
    # --- Argument Parsing Loop ---
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --ssp) MODE="ssp"; shift ;;
            --arms-race) MODE="arms_race"; shift ;;
            --n-envs) N_ENVS="$2"; shift 2 ;;
            --seed) SEED="$2"; shift 2 ;;
            --config-path) CONFIG_PATH="$2"; shift 2 ;;
            --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
            --prot-steps) PROTAGONIST_TIMESTEPS="$2"; shift 2 ;;
            --adv-steps) ADVERSARY_TIMESTEPS="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown parameter passed: $1"; usage ;;
        esac
    done

    # --- Housekeeping ---
    mkdir -p "$PROT_DIR"
    mkdir -p "$ADV_DIR"

    log_info "Starting Training Pipeline in '$MODE' mode"
    next_agent=$(detect_state)
    
    start_iteration=$(( $(count_models "$PROT_DIR" "final_model.zip") + $(count_models "$ADV_DIR" "final_adversary_model.zip") ))

    for ((i=start_iteration; i<MAX_ITERATIONS; i++)); do
        log_info ""
        log_info "╔════════════════════════════════════════╗"
        log_info "║   Arms Race Iteration: $((i + 1)) / $MAX_ITERATIONS"
        log_info "╚════════════════════════════════════════╝"
        log_info ""
        
        if [[ "$next_agent" == "protagonist" ]]; then
            # --- TRAIN PROTAGONIST ---
            local adv_count=$(count_models "$ADV_DIR" "final_adversary_model.zip")
            local noise_setting=""
            local extra_args=""
            
            if [[ $adv_count -eq 0 ]]; then
                noise_setting="procedural"
                log_info "No adversaries exist. Training protagonist against procedural noise."
            else
                if [[ "$MODE" == "ssp" ]]; then
                    noise_setting="synthetic_self_play"
                    extra_args="--adversary-pool-path $ADV_DIR"
                else # Naive "arms_race" mode
                    noise_setting="adversarial"
                    latest_adversary=$(find_latest_model "$ADV_DIR" "final_adversary_model.zip")
                    extra_args="--antagonist-path $latest_adversary"
                fi
            fi
            
            log_info "=========================================="
            log_info "Training Protagonist (Iteration $i)"
            log_info "Noise type: $noise_setting"
            log_info "=========================================="
            
            local cmd="python train_protagonist.py --project-name WindGymTraining --run-name-prefix Prot_Iter${i} --yaml-config-path $CONFIG_PATH --noise-type $noise_setting --total-timesteps $PROTAGONIST_TIMESTEPS --seed $SEED --n-envs $N_ENVS $extra_args"
            
            log_info "Executing: $cmd"
            if ! eval "$cmd"; then log_error "Protagonist training failed."; exit 1; fi
            
            next_agent="adversary"

        else # [[ "$next_agent" == "adversary" ]]
            # --- TRAIN ADVERSARY ---
            latest_protagonist=$(find_latest_model "$PROT_DIR" "final_model.zip")
            if [[ -z "$latest_protagonist" ]]; then
                log_error "Cannot train adversary: No protagonist found to train against!"; exit 1;
            fi

            log_info "=========================================="
            log_info "Training Adversary (Iteration $i)"
            log_info "Against protagonist: $latest_protagonist"
            log_info "=========================================="

            local cmd="python train_adversary.py --project-name WindGymTraining --run-name-prefix Adv_Iter${i} --protagonist-path $latest_protagonist --yaml-config-path $CONFIG_PATH --total-timesteps $ADVERSARY_TIMESTEPS --seed $SEED --n-envs $N_ENVS"
            
            log_info "Executing: $cmd"
            if ! eval "$cmd"; then log_error "Adversary training failed."; exit 1; fi
            
            next_agent="protagonist"
        fi
        sleep 2
    done

    log_info "Arms Race Training Complete!"
    echo
    echo "=== Final Training Summary ==="
    echo "Protagonists trained in total: $(count_models "$PROT_DIR" "final_model.zip")"
    echo "Adversaries trained in total: $(count_models "$ADV_DIR" "final_adversary_model.zip")"
}

# Pass all script arguments to the main function
main "$@"
