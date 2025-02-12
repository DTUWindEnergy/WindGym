#!/bin/bash
#SBATCH --partition=fatq,windfatq
#SBATCH --job-name="RL"
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc
conEnv
conda activate marcus
echo "============================================================================="

#python examples/fulltime.py --dt_env 1 --power_avg 40 --seed 123  --yaml_path DTUWindGym/envs/WindFarmEnv/Examples/EnvConfigs/2turb.yaml --turbbox_path Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc --lookback_window 100 --learning_rate 3e-4 --momentum 0.
python examples/curriculum.py --dt_env 2 --power_avg 40 --seed 123  --yaml_path DTUWindGym/envs/WindFarmEnv/Examples/EnvConfigs/2turb.yaml --turbbox_path Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc --lookback_window 100 --learning_rate 3e-4 --momentum 0. --n_passthrough 3 --penalty_mult 5
