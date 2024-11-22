#!/bin/bash
#SBATCH -o logs/pendulum_transfer/output-%j.log
#SBATCH --job-name=pendulum_transfer
#SBATCH --array=150-299             # number of trials 0-399
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                    # number of cpu per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)


source ~/.bash_profile
export OMP_NUM_THREADS=1

SCALE_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $SCALE_IDX $TRIAL_IDX

SCALE_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10)
TRIAL_LIST=(0 1 2)
SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=1.0
VAR1=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

SCALE2_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10)


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Source VAR: " $VAR1

# Run the transfer for each VAR2 value
for SCALE2 in "${SCALE2_LIST[@]}"; do
    VAR2=$(echo "scale=2; $SCALE2 * $DEFAULT" | bc)
    echo "Target VAR: $VAR2"
    RESULT_PATH="/home/gridsan/xxxx/xxxx/MBTL/results/pendulum/maxspeed8.00-dt0.05-g10.00-m1.00-l$VAR1-PPO-trial$TRIAL/transfer_maxspeed8.00-dt0.05-g10.00-m1.00-l$VAR2-PPO-trial$TRIAL"
    mkdir $RESULT_PATH
    if [ ! -f "$RESULT_PATH/test_reward.csv" ]; then
        echo "File $RESULT_PATH/test_reward.csv does not exist, running simulation."
        python -u transfer.py \
            --save_path $RESULT_PATH \
            --source_path /home/gridsan/xxxx/xxxx/MBTL/results/pendulum/maxspeed8.00-dt0.05-g10.00-m1.00-l$VAR1-PPO-trial$TRIAL \
            --pd_max_speed 8.0 \
            --pd_dt 0.05 \
            --pd_g 10 \
            --pd_m 1 \
            --pd_l $VAR2 \
            --alg PPO \
            --total_steps 1500000 \
            --source_l $VAR1 \
            --source_m 0.6 \
            --total_steps 150000 \
            --env pendulum \
            --test_eps 100
    else
        echo "File $RESULT_PATH exists, skipping."
    fi
done