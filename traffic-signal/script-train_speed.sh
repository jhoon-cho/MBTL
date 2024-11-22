#!/bin/bash

#SBATCH -o logs/train_speed/output-%j.log
#SBATCH --job-name=train_speed
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

# VAR_LIST=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
VAR_LIST=(0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25)
TRIAL_LIST=(0 1 2)

VAR_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $VAR_IDX $TRIAL_IDX

VAR=${VAR_LIST[VAR_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "VAR: $VAR TRIAL: $TRIAL"

python -u training_main.py \
    --flow 1000 \
    --lane 4 \
    --length 750 \
    --speed $VAR \
    --left 0.25 \
    --trial $TRIAL \
    --alg DQN