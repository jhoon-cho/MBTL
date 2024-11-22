#!/bin/bash

#SBATCH -o logs/train_flow/output-%j.log
#SBATCH --job-name=train_flow
#SBATCH --array=0-149             # number of trials 0-389
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

# VAR_LIST=(100 150 200 250 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)
VAR_LIST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950 2000 2050 2100 2150 2200 2250 2300 2350 2400 2450 2500)
TRIAL_LIST=(0 1 2)

VAR_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $VAR_IDX $TRIAL_IDX

VAR=${VAR_LIST[VAR_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "VAR: $VAR TRIAL: $TRIAL"

python -u training_main.py \
    --flow $VAR \
    --lane 4 \
    --length 750 \
    --speed 14 \
    --left 0.25 \
    --trial $TRIAL \
    --alg DQN