#!/bin/bash
# ===================================================================================
# Arms Race Gauntlet Evaluation Script (v5 - Final Cross-Platform)
#
# This script is fully compatible with both Linux and macOS. It uses the
# correct 'find' syntax for each OS to ensure model discovery works reliably.
# ===================================================================================
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
SEED=1337
SIM_TIME=2000
CONFIG_PATH="env_config/two_turbine_yaw.yaml"

# ===================================================================================
# Help and Usage Function
# ===================================================================================
usage() {
    echo "Usage: $0 [path/to/run_dir_1] [path/to/run_dir_2] ..."
    echo ""
    echo "Arguments:"
    echo "  <run_dir>    One or more paths to top-level training run directories."
    echo "               Each directory will be processed independently."
    echo ""
    echo "Options:"
    echo "  -h, --help   Display this help message."
    exit 1
}

# ===================================================================================
# Main Logic
# ===================================================================================

# --- Argument Parsing ---
if [[ "$#" -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# --- Main Loop: Process Each Directory Provided as an Argument ---
for RUN_DIR in "$@"; do
    if [ ! -d "$RUN_DIR" ]; then
        echo "[WARNING] Directory not found: '$RUN_DIR'. Skipping."
        continue
    fi

    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║      STARTING GAUNTLET FOR: $(basename "$RUN_DIR")"
    echo "╚═════════════════════════════════════════════════════════╝"

    # --- Directory Structure Detection ---
    PROT_DIR="$RUN_DIR/protagonist_training"
    ADV_DIR="$RUN_DIR/adversaries_stateful"
    SELFPLAY_DIR="$RUN_DIR/self_play"
    
    # Check for incompatible .pt file structure
    if [ -d "$SELFPLAY_DIR" ]; then
        echo "[WARNING] Detected 'self_play' directory with .pt models in '$RUN_DIR'."
        echo "[WARNING] This format is incompatible with the current PPO-based evaluation script."
        echo "[WARNING] Skipping this directory."
        echo "---------------------------------------------------------------"
        continue
    fi

    if [ ! -d "$PROT_DIR" ]; then
        echo "[WARNING] No 'protagonist_training' directory found in '$RUN_DIR'. Skipping."
        echo "---------------------------------------------------------------"
        continue
    fi

    # --- Define Output Paths ---
    OUTPUT_DIR="$RUN_DIR/gauntlet_results_csv"
    MATRIX_DIR="$RUN_DIR/gauntlet_matrix_results"

    # --- Housekeeping ---
    echo "[INFO] Cleaning up old results in '$RUN_DIR'..."
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # --- CROSS-PLATFORM FIX: Use a 'while read' loop AND the correct 'find' command for the OS ---
    
    # Discover and sort protagonists
    protagonists=()
    if [[ "$(uname)" == "Darwin" ]]; then # macOS/BSD find command
        while IFS= read -r line; do protagonists+=("$line"); done < <(find "$PROT_DIR" -mindepth 2 -type f -name "final_model.zip" -exec stat -f "%m %N" {} + 2>/dev/null | sort -n | cut -d' ' -f2-)
    else # Linux/GNU find command
        while IFS= read -r line; do protagonists+=("$line"); done < <(find "$PROT_DIR" -mindepth 2 -type f -name "final_model.zip" -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)
    fi
    
    # Discover and sort adversaries
    adversaries=()
    if [[ "$(uname)" == "Darwin" ]]; then # macOS/BSD find command
        while IFS= read -r line; do adversaries+=("$line"); done < <(find "$ADV_DIR" -mindepth 2 -type f -name "final_adversary_model.zip" -exec stat -f "%m %N" {} + 2>/dev/null | sort -n | cut -d' ' -f2-)
    else # Linux/GNU find command
        while IFS= read -r line; do adversaries+=("$line"); done < <(find "$ADV_DIR" -mindepth 2 -type f -name "final_adversary_model.zip" -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)
    fi
    
    echo "[INFO] Discovered ${#protagonists[@]} protagonist(s), ordered by creation time."
    echo "[INFO] Discovered ${#adversaries[@]} adversary model(s), ordered by creation time."

    if [ ${#protagonists[@]} -eq 0 ]; then
        echo "[WARNING] No protagonist models found in '$PROT_DIR'. Skipping run."
        continue
    fi
    
    environments=("clean" "procedural")
    environments+=("${adversaries[@]}")

    # --- Evaluation Loop ---
    for prot_idx in "${!protagonists[@]}"; do
        prot_path="${protagonists[$prot_idx]}"
        prot_name="Prot_${prot_idx}"
        echo
        echo "--- Evaluating ${prot_name} (Path: $prot_path) ---"

        for env_idx in "${!environments[@]}"; do
            env_path_or_name="${environments[$env_idx]}"
            
            scenario_name=""
            extra_args=""
            
            if [[ "$env_path_or_name" == "clean" || "$env_path_or_name" == "procedural" ]]; then
                scenario_name="$env_path_or_name"
            else
                scenario_name="adversarial"
                extra_args="--antagonist-path $env_path_or_name"
            fi
            
            env_name="Env_${env_idx}"
            output_path="$OUTPUT_DIR/${prot_name}_in_${env_name}.csv"
            
            echo "  -> vs ${env_name} (Scenario: $scenario_name)..."
            
            python evaluate_agents.py \
                --agent-type ppo \
                --protagonist-path "$prot_path" \
                --scenario "$scenario_name" \
                --output-path "$output_path" \
                --sim-time "$SIM_TIME" \
                --seed "$SEED" \
                --config-path "$CONFIG_PATH" \
                $extra_args
        done
    done

    # --- Post-Processing ---
    echo
    echo "--- All simulations complete for $(basename "$RUN_DIR"). Generating evaluation matrix... ---"
    python create_matrix_from_csvs.py --input-dir "$OUTPUT_DIR" --output-dir "$MATRIX_DIR"
    echo "✅ Gauntlet complete for $(basename "$RUN_DIR")! Results in '$MATRIX_DIR'."
    echo "---------------------------------------------------------------"
done

echo
echo "All specified directories have been processed."
