#!/bin/bash 
#SBATCH -o logs/walker_source/walker_source-%j.log
#SBATCH --job-name=walker_source
#SBATCH --array=0-99             # number of trials 0-299
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH --constraint xeon-p8

source ~/.bashrc
conda activate mtl
export OMP_NUM_THREADS=16

SCALE_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10)
TRIAL_LIST=(0 1 2)

SCALE_IDX=$SLURM_ARRAY_TASK_ID
TRIAL_IDX=0
echo $SCALE_IDX $TRIAL_IDX

SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=2.0
VAR=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

echo "SCALE: $SCALE VAR: $VAR TRIAL: $TRIAL"

echo "Job: train walker gravity $VAR trial $TRIAL"

python train.py --save_path ./results/walker/gravity$VAR-scale30.00-friction2.50-torque80.00-legh1.13-PPO-trial$TRIAL \
	--wk_GRAVITY_Y $VAR \
	--wk_SCALE 30 \
	--wk_FRICTION 2.5 \
	--wk_MOTORS_TORQUE 80 \
	--alg PPO \
	--total_steps 5000000 \
	--env walker

SCALE_IDX=$SLURM_ARRAY_TASK_ID
TRIAL_IDX=1
echo $SCALE_IDX $TRIAL_IDX

SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=2.0
VAR=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

echo "SCALE: $SCALE VAR: $VAR TRIAL: $TRIAL"

echo "Job: train walker gravity $VAR trial $TRIAL"

python train.py --save_path ./results/walker/gravity$VAR-scale30.00-friction2.50-torque80.00-legh1.13-PPO-trial$TRIAL \
	--wk_GRAVITY_Y $VAR \
	--wk_SCALE 30 \
	--wk_FRICTION 2.5 \
	--wk_MOTORS_TORQUE 80 \
	--alg PPO \
	--total_steps 5000000 \
	--env walker

SCALE_IDX=$SLURM_ARRAY_TASK_ID
TRIAL_IDX=2
echo $SCALE_IDX $TRIAL_IDX

SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=2.0
VAR=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

echo "SCALE: $SCALE VAR: $VAR TRIAL: $TRIAL"

echo "Job: train walker gravity $VAR trial $TRIAL"

python train.py --save_path ./results/walker/gravity$VAR-scale30.00-friction2.50-torque80.00-legh1.13-PPO-trial$TRIAL \
	--wk_GRAVITY_Y $VAR \
	--wk_SCALE 30 \
	--wk_FRICTION 2.5 \
	--wk_MOTORS_TORQUE 80 \
	--alg PPO \
	--total_steps 5000000 \
	--env walker
