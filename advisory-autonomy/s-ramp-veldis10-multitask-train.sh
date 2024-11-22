#!/bin/bash

#SBATCH -o output-%j.log
#SBATCH --job-name=ramp_vel_multi
#SBATCH --array=0-4             # number of trials 0-389
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 24                     # number of cpu per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)


export FLOW_RES_DIR="$HOME/pwc/results/"

# Loading the required module
source /etc/profile
module load anaconda/2022a

# HC_IDX=${SLURM_ARRAY_TASK_ID}/5
TRIAL=$((${SLURM_ARRAY_TASK_ID}%5))


python -u pexps/highway_ramp.py ./results/230302-xxxx-ramp/veldis10-multi-TRPO-${TRIAL} 'act_type="vel_discrete"' "worker_kwargs=[{'hc_param':400},{'hc_param':350},{'hc_param':300},{'hc_param':250},{'hc_param':200},{'hc_param':150},{'hc_param':100},{'hc_param':50},{'hc_param':20},{'hc_param':10}]" "n_workers=20" "n_actions=10" "n_rollouts_per_step=200"
