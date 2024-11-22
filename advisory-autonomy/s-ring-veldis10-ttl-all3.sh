#!/bin/bash

#SBATCH -o output-%j.log
#SBATCH --job-name=ring_vel_all3
#SBATCH --array=123-244              # number of trials 0-244
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)


export FLOW_RES_DIR="$HOME/pwc/results/"

# source ~/.bash_profile
# export OMP_NUM_THREADS=1

# Loading the required module
source /etc/profile
module load anaconda/2022a

# SLURM_ARRAY_TASK_ID=0
HC_IDX=${SLURM_ARRAY_TASK_ID}/5
TRIAL=$((${SLURM_ARRAY_TASK_ID}%5))

HC_PARAMS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400) # 49 params

echo "========================================="
echo "hc_param: " ${HC_PARAMS[HC_IDX]}
echo "Trial: " ${TRIAL}


idx=`python analysis/find_best_result_ttlall.py --dir_name "230301-xxxx-ring" --source_name "HC" --accvel "veldis10" --hc_param ${HC_PARAMS[HC_IDX]} --trial ${TRIAL} | tail -n1`
echo "selected idx" $idx


for eval_hc_param in "${HC_PARAMS[@]}"
do
    mkdir -p ./results/230301-xxxx-ring-ttlall3/veldis10-${HC_PARAMS[HC_IDX]}-TRPO-${TRIAL}/HC${eval_hc_param}
    cp -RT ./results/230301-xxxx-ring/veldis10-${HC_PARAMS[HC_IDX]}-TRPO-${TRIAL}/ ./results/230301-xxxx-ring-ttlall3/veldis10-${HC_PARAMS[HC_IDX]}-TRPO-${TRIAL}/HC${eval_hc_param}/
    wait -n	
done
for eval_hc_param in "${HC_PARAMS[@]}"
do
    echo "evaluating on $eval_hc_param"
    python pexps/ring.py ./results/230301-xxxx-ring-ttlall3/veldis10-${HC_PARAMS[HC_IDX]}-TRPO-${TRIAL}/HC${eval_hc_param}/ \
        'act_type="vel_discrete"' \
        n_actions=10 \
        hc_param=$eval_hc_param \
        e=$idx \
        n_steps=50 \
        result_save=results/230301-xxxx-ring-ttlall3/veldis10-${HC_PARAMS[HC_IDX]}-TRPO-${TRIAL}/HC${eval_hc_param}/eval_result_hc${eval_hc_param}_50.csv
done






