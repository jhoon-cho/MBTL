#!/bin/bash

#SBATCH -o logs/multitransfer_flow/output-%j.log
#SBATCH --job-name=multitransfer_flow
#SBATCH --array=0-149             # number of trials 0-399
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                    # number of cpu per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

export F=$PWD

export WANDB_API_KEY="d35f5974f1d9ff835260b57af640e0d4cd7908ee"
export WANDB_MODE="offline"
export LIBSUMO=true

# source ~/.bash_profile
# export OMP_NUM_THREADS=1

# Loading the required module
# source /etc/profile
# module load anaconda/2022a

VAR2_LIST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950 2000 2050 2100 2150 2200 2250 2300 2350 2400 2450 2500)
TRIAL_LIST=(0 1 2)

VAR_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $VAR_IDX $TRIAL_IDX

VAR2=${VAR2_LIST[VAR_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Source VAR: " ${VAR1_LIST[VAR_IDX]}


echo "Target VAR: $VAR2"
RESULT_PATH="/home/gridsan/xxxx/xxxx/traffic-signal/results/intersection_flowmulti_lane4_length750.0_speed14.0_left0.25_algDQN_trial${TRIAL}/transfer/intersection_flow${VAR2}_lane4_length750.0_speed14.0_left0.25_algDQN_trial${TRIAL}/transfer_results_1.csv"
if [ ! -f "$RESULT_PATH" ]; then
    echo "File $RESULT_PATH does not exist, running simulation."
    python -u transfer_main.py \
        --flow $VAR2 \
        --lane 4 \
        --length 750 \
        --speed 14 \
        --left 0.25 \
        --model_num 2 \
        --source_path_name "intersection_flowmulti_lane4_length750.0_speed14.0_left0.25_algDQN_trial$TRIAL/" \
        --num_episodes 50 \
        --trial $TRIAL \
        --alg DQN
else
    echo "File $RESULT_PATH exists, skipping."
fi