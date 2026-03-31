#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# =============================================================================
# Arms Race Gauntlet Evaluation Script
# -----------------------------------------------------------------------------
# This script evaluates every trained protagonist against every trained
# adversary, as well as against the 'clean' and 'procedural' noise scenarios.
# It then generates a summary matrix and heatmap of the results.
# =============================================================================

# --- Configuration ---
SEED=1337
SIM_TIME=2000
NUM_EPISODES=2 # Number of episodes to average over for each matrix cell
CONFIG_PATH="env_config/two_turbine_yaw.yaml"

# --- Directories ---
PROT_DIR="models/protagonist_training"
ADV_DIR="models/adversaries_stateful"
OUTPUT_DIR="gauntlet_results_csv"
MATRIX_DIR="gauntlet_matrix"

echo "╔════════════════════════════════════════╗"
echo "║      STARTING ARMS RACE GAUNTLET       ║"
echo "╚════════════════════════════════════════╝"

# --- Housekeeping ---
echo "[INFO] Cleaning up old results..."
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.csv

# --- Agent Discovery ---
# Use `find` to create arrays of all model paths
protagonist_models=($(find "$PROT_DIR" -type f -name "*.zip" | sort))
adversary_models=($(find "$ADV_DIR" -type f -name "*.zip" | sort))

echo "[INFO] Discovered ${#protagonist_models[@]} protagonist(s)."
echo "[INFO] Discovered ${#adversary_models[@]} adversary model(s)."

if [ ${#protagonist_models[@]} -eq 0 ]; then
    echo "[ERROR] No protagonist models found in '$PROT_DIR'. Exiting."
    exit 1
fi

# --- Main Evaluation Loop ---
# Iterate over every episode for averaging
for i in $(seq 0 $(($NUM_EPISODES - 1))); do
    CURRENT_SEED=$(($SEED + $i))
    echo
    echo "--- Running evaluations for Episode $(($i + 1))/$NUM_EPISODES (Seed: $CURRENT_SEED) ---"

    # Iterate over every protagonist
    for prot_path in "${protagonist_models[@]}"; do
        # Extract a short, unique ID from the model path (e.g., the wandb ID)
        prot_id=$(basename "$(dirname "$prot_path")")
        echo "[PROT: $prot_id] Evaluating..."

        # 1. Evaluate against the CLEAN scenario
        agent_name="Protagonist_($prot_id)"
        scenario_name="Clean"
        output_path="$OUTPUT_DIR/${agent_name}_in_${scenario_name}_ep${i}.csv"
        python evaluate_agents.py --agent-type ppo --protagonist-path "$prot_path" --scenario clean --output-path "$output_path" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH

        # 2. Evaluate against the PROCEDURAL noise scenario
        scenario_name="Procedural"
        output_path="$OUTPUT_DIR/${agent_name}_in_${scenario_name}_ep${i}.csv"
        python evaluate_agents.py --agent-type ppo --protagonist-path "$prot_path" --scenario procedural --output-path "$output_path" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
        
        # 3. Evaluate against EVERY ADVERSARY
        for adv_path in "${adversary_models[@]}"; do
            adv_id=$(basename "$(dirname "$adv_path")")
            scenario_name="Adversary_($adv_id)"
            output_path="$OUTPUT_DIR/${agent_name}_in_${scenario_name}_ep${i}.csv"
            python evaluate_agents.py --agent-type ppo --protagonist-path "$prot_path" --scenario adversarial --antagonist-path "$adv_path" --output-path "$output_path" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
        done
    done
done

# --- Post-Processing ---
echo
echo "--- All simulations complete. Post-processing results... ---"
python create_matrix_from_csvs.py --input-dir "$OUTPUT_DIR" --output-dir "$MATRIX_DIR"

echo
echo "✅ Gauntlet complete! Matrix and heatmap saved to '$MATRIX_DIR'."
