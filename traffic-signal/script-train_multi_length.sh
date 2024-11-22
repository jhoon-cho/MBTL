#!/bin/bash

#SBATCH -o logs/multitrain_length/output-%j.log
#SBATCH --job-name=multitrain_length
#SBATCH --array=0-2             # number of trials 0-389
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 16                    # number of cpu per task
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

VAR_IDX=${SLURM_ARRAY_TASK_ID}

TRIAL_LIST=(0 1 2)

TRIAL_IDX=$SLURM_ARRAY_TASK_ID
echo $TRIAL_IDX

TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "TRIAL: $TRIAL"

python -u training_multi_main.py \
    --flow 1000 \
    --lane 4 \
    --length 750 \
    --speed 14 \
    --left 0.25 \
    --trial $TRIAL \
    --alg DQN \
    --multi length