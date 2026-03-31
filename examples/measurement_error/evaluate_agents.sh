#!/bin/bash

# --- Config ---
SEED=69
SIM_TIME=2000 # Simulation time in seconds
NUM_EPISODES=2 # Number of episodes to average over for each matrix cell
OUTPUT_DIR="full_timeseries_results"
MATRIX_DIR="evaluation_matrix"
CONFIG_PATH="env_config/two_turbine_yaw.yaml"
PROT_DIR="models/protagonist_training"
ADV_DIR="models/adversaries_stateful"

# --- Helper Functions to find models ---
find_latest_model() {
    local dir="$1"
    local filename="$2"
    find "$dir" -type f -name "$filename" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# Find the latest protagonist trained against a specific noise type
find_specific_protagonist() {
    local noise_type="$1" # 'procedural' or 'adversarial'
    
    # Heuristic: Find the latest run for the noise type by checking wandb metadata if available,
    # or just find the overall latest if that's too complex.
    # For simplicity, we'll find the first (oldest) for "clean"/procedural and the latest for "adversarial".
    
    if [[ "$noise_type" == "procedural" ]]; then
        # Find oldest model (usually the one trained only on procedural noise)
        find "$PROT_DIR" -type f -name "final_model.zip" -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-
    else
        # Find newest model (usually the one trained against an adversary)
        find_latest_model "$PROT_DIR" "final_model.zip"
    fi
}


# --- Agent Path Discovery ---
# FIX: Automatically find models instead of hardcoding paths
PROT_PROCEDURAL_PATH=$(find_specific_protagonist "procedural")
PROT_ADVERSARIAL_PATH=$(find_specific_protagonist "adversarial")
ANTAGONIST_PATH=$(find_latest_model "$ADV_DIR" "final_adversary_model.zip")

echo "--- Discovered Models for Evaluation ---"
echo "Procedural Protagonist: $PROT_PROCEDURAL_PATH"
echo "Adversarial Protagonist: $PROT_ADVERSARIAL_PATH"
echo "Latest Adversary:       $ANTAGONIST_PATH"
echo "----------------------------------------"

if [[ -z "$PROT_PROCEDURAL_PATH" || -z "$PROT_ADVERSARIAL_PATH" || -z "$ANTAGONIST_PATH" ]]; then
    echo "ERROR: Could not find all necessary models. Please run training first."
    exit 1
fi


# --- Housekeeping ---
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.csv # Clean old results
echo "Starting Evaluation Gauntlet with new configuration..."

# --- Run Matrix Evaluations (Agent vs. Scenario) ---
for i in $(seq 0 $(($NUM_EPISODES - 1))); do
    CURRENT_SEED=$(($SEED + $i))
    echo "--- Running evaluations for Episode $(($i + 1))/$NUM_EPISODES (Seed: $CURRENT_SEED) ---"

    # 1. Procedural Protagonist
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario clean --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario procedural --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_PROCEDURAL_PATH --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PPO_(Procedural)_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH

    # 2. Adversarially-Trained Protagonist
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_ADVERSARIAL_PATH --scenario clean --output-path "$OUTPUT_DIR/PPO_(Adversarial)_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_ADVERSARIAL_PATH --scenario procedural --output-path "$OUTPUT_DIR/PPO_(Adversarial)_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type ppo --protagonist-path $PROT_ADVERSARIAL_PATH --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PPO_(Adversarial)_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH

    # 3. PyWake Agent (Baseline)
    python evaluate_agents.py --agent-type pywake --scenario clean --output-path "$OUTPUT_DIR/PyWakeAgent_in_clean_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type pywake --scenario procedural --output-path "$OUTPUT_DIR/PyWakeAgent_in_procedural_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
    python evaluate_agents.py --agent-type pywake --scenario adversarial --antagonist-path $ANTAGONIST_PATH --output-path "$OUTPUT_DIR/PyWakeAgent_in_adversarial_ep${i}.csv" --sim-time $SIM_TIME --seed $CURRENT_SEED --config-path $CONFIG_PATH
done

# --- Post-Process the Results ---
echo "--- All simulations complete. Post-processing results... ---"
python create_matrix_from_csvs.py --input-dir $OUTPUT_DIR --output-dir $MATRIX_DIR
