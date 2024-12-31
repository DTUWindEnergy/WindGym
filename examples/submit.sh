#!/bin/bash
#SBATCH --partition=rome,workq,windq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --job-name=dt_pow_study
#SBATCH --output=dt_pow_study_%A_%a.log
#SBATCH --array=0-8  # For 3 timesteps × 3 power windows = 9 combinations

. ~/.bashrc
pixi self-update
cd ..
eval "$(pixi shell-hook)"
cd examples

# Define timesteps and power averaging windows to test
declare -a timesteps=(1 10 20)
declare -a power_avgs=(1 5 100)

# Calculate indices for the timestep and power_avg arrays
DT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
POW_IDX=$((SLURM_ARRAY_TASK_ID % 3))

DT_ENV=${timesteps[$DT_IDX]}
POW_AVG=${power_avgs[$POW_IDX]}

# Calculate actual power averaging steps
POWER_AVG_STEPS=$((POW_AVG))
#POWER_AVG_STEPS=$((POW_AVG / DT_ENV))

# Run the Python script with specific parameters
python longer_steps_example.py \
    --dt_env $DT_ENV \
    --power_avg $POWER_AVG_STEPS \
    --seed $((42 + $SLURM_ARRAY_TASK_ID)) \
    --n_env 8 \
    --train_steps 500000 \
    --yaml_path "../DTUWindGym/envs/WindFarmEnv/Examples/EnvConfigs/2turb.yaml" \
    --turbbox_path "Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc"
