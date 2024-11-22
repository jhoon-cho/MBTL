#!/bin/bash
#SBATCH -o logs/stop_transfer/output-%j.log
#SBATCH --job-name=no-stop_transfer
#SBATCH --array=0-149             # number of trials 0-399
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                    # number of cpu per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)


source ~/.bash_profile
export OMP_NUM_THREADS=4

SCALE_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $SCALE_IDX $TRIAL_IDX

SCALE_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5)
TRIAL_LIST=(0 1 2)
SCALE=${SCALE_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT=0.2
VAR1=$(echo "scale=2; $SCALE * $DEFAULT" | bc)

SCALE2_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5)


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Source VAR: " $VAR1

# Run the transfer for each VAR2 value
for SCALE2 in "${SCALE2_LIST[@]}"; do
    VAR2=$(echo "scale=2; $SCALE2 * $DEFAULT" | bc)
    echo "Target VAR: $VAR2"
    RESULT_PATH="./results/no-stop/inflow400-penrate$VAR1-green35-PPO-trial$TRIAL/transfer_inflow400-penrate$VAR2-green35-PPO-trial$TRIAL"
    mkdir $RESULT_PATH
    if [ ! -f "$RESULT_PATH/eval_result.csv" ]; then
        echo "File $RESULT_PATH/eval_result.csv does not exist, running simulation."
        python -u code/main.py \
            --dir $RESULT_PATH \
            --source_path ./results/no-stop/inflow400-penrate$VAR1-green35-PPO-trial$TRIAL \
            --kwargs "{'run_mode':'single_eval', 'n_steps':100}" \
            --inflow 400 \
            --penrate $VAR2 \
            --green 35
    else
        echo "File $RESULT_PATH exists, skipping."
    fi
done
