#!/bin/bash 
#SBATCH -o logs/no-stop_source/no-stop_source_log-%j.log
#SBATCH --job-name=no-stop_source
#SBATCH --array=0-149             # number of trials 0-49
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                     # number of cpu per task
# SBATCH --gres=gpu:volta:1
#SBATCH --constraint xeon-p8

# source ~/.bashrc
# Loading the required module
# source /etc/profile
# module load anaconda/2022a

# conda activate no-stop
export WANDB_MODE="offline"
export OMP_NUM_THREADS=16

SCALE_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $SCALE_IDX $TRIAL_IDX

SCALE_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5)
TRIAL_LIST=(0 1 2)
SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=10
VAR=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

echo "SCALE: $SCALE VAR: $VAR TRIAL: $TRIAL"

echo "Job: train inflow $VAR trial $TRIAL"

# changing mass of cart

# python -u code/main.py --dir ./results/no-stop/inflow$VAR-penrate0.2-green35-PPO-trial$TRIAL

python -u code/main.py --dir ./results/no-stop/inflow400-penrate0.2-green$VAR-PPO-trial$TRIAL \
	--kwargs "{'run_mode':'train'}" \
	--inflow 400 \
	--penrate 0.2 \
	--green $VAR


# python -u code/main.py --dir wd/$EXP_NAME --kwargs "$KWARGS" --task_context_kwargs "$TASK_KWARGS"
