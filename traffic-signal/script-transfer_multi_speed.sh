#!/bin/bash

#SBATCH -o logs/multitransfer_speed/output-%j.log
#SBATCH --job-name=multitransfer_speed
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

VAR2_LIST=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0 14.5 15.0 15.5 16.0 16.5 17.0 17.5 18.0 18.5 19.0 19.5 20.0 20.5 21.0 21.5 22.0 22.5 23.0 23.5 24.0 24.5 25.0)
TRIAL_LIST=(0 1 2)

VAR_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $VAR_IDX $TRIAL_IDX

VAR2=${VAR2_LIST[VAR_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Source VAR: " ${VAR1_LIST[VAR_IDX]}


echo "Target VAR: $VAR2"
RESULT_PATH="/home/gridsan/xxxx/xxxx/traffic-signal/results/intersection_flow1000_lane4_length750.0_speedmulti_left0.25_algDQN_trial${TRIAL}/transfer/intersection_flow1000_lane4_length750.0_speed${VAR2}_left0.25_algDQN_trial${TRIAL}/transfer_results_1.csv"
if [ ! -f "$RESULT_PATH" ]; then
    echo "File $RESULT_PATH does not exist, running simulation."
    python -u transfer_main.py \
        --flow 1000 \
        --lane 4 \
        --length 750 \
        --speed $VAR2 \
        --left 0.25 \
        --model_num 2 \
        --source_path_name "intersection_flow1000_lane4_length750.0_speedmulti_left0.25_algDQN_trial$TRIAL/" \
        --num_episodes 50 \
        --trial $TRIAL \
        --alg DQN
else
    echo "File $RESULT_PATH exists, skipping."
fi