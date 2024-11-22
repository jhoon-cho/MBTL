import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import *

home_dir = "../"

data_list = []

data_list = data_list + ["cartpole_masscart_0", "cartpole_masscart_1", "cartpole_masscart_2", "cartpole_lenpole_0", "cartpole_lenpole_1", "cartpole_lenpole_2", "cartpole_masspole_0", "cartpole_masspole_1", "cartpole_masspole_2"]
data_list = data_list + ["pendulum_dt_0", "pendulum_dt_1", "pendulum_dt_2", "pendulum_l_0", "pendulum_l_1", "pendulum_l_2", "pendulum_m_0", "pendulum_m_1", "pendulum_m_2"]
data_list = data_list + ["walker_friction_0", "walker_friction_1", "walker_friction_2", "walker_gravity_0", "walker_gravity_1", "walker_gravity_2", "walker_scale_0", "walker_scale_1", "walker_scale_2"]
data_list = data_list + ['halfcheetah_friction_0', 'halfcheetah_friction_1', 'halfcheetah_friction_2', 'halfcheetah_gravity_0', 'halfcheetah_gravity_1', 'halfcheetah_gravity_2', 'halfcheetah_stiffness_0', 'halfcheetah_stiffness_1', 'halfcheetah_stiffness_2']
data_list = data_list + ["cartpole_lenpole_ppo_0", "cartpole_lenpole_ppo_1", "cartpole_lenpole_ppo_2", "cartpole_masspole_ppo_0", "cartpole_masspole_ppo_1", "cartpole_masspole_ppo_2", "cartpole_masscart_ppo_0", "cartpole_masscart_ppo_1", "cartpole_masscart_ppo_2"]
data_list = data_list + ["cartpole_lenpole_a2c_0", "cartpole_lenpole_a2c_1", "cartpole_lenpole_a2c_2", "cartpole_masspole_a2c_0", "cartpole_masspole_a2c_1", "cartpole_masspole_a2c_2", "cartpole_masscart_a2c_0", "cartpole_masscart_a2c_1", "cartpole_masscart_a2c_2"]
data_list = data_list + ["intersection_flow_0", "intersection_flow_1", "intersection_flow_2", "intersection_speed_0", "intersection_speed_1", "intersection_speed_2", "intersection_length_0", "intersection_length_1", "intersection_length_2"]
data_list = data_list + ["advisoryautonomy_ring_acc_0", "advisoryautonomy_ring_acc_1", "advisoryautonomy_ring_acc_2", 
                         "advisoryautonomy_ring_vel_0", "advisoryautonomy_ring_vel_1", "advisoryautonomy_ring_vel_2",
                         "advisoryautonomy_ramp_acc_0", "advisoryautonomy_ramp_acc_1", "advisoryautonomy_ramp_acc_2",
                         "advisoryautonomy_ramp_vel_0", "advisoryautonomy_ramp_vel_1", "advisoryautonomy_ramp_vel_2",]
data_list = data_list + ['no-stop_green_0', 'no-stop_green_1', 'no-stop_green_2', 'no-stop_penrate_0', 'no-stop_penrate_1', 'no-stop_penrate_2', 'no-stop_inflow_0', 'no-stop_inflow_1', 'no-stop_inflow_2']

for data_name in data_list:
    print("DATA: ", data_name)
    RANDOMNESS = False
    np.random.seed(42)
    
    data_transfer, deltas, delta_min, delta_max, slope, lower_bound, upper_bound, unguided = import_data(home_dir, data_name, RANDOMNESS)
    # plot_heatmap(data_transfer, home_dir, data_name)
    
    # num_transfer_steps = 15
    num_transfer_steps = 100
    
    if data_name in ["advisoryautonomy_ring_acc", "advisoryautonomy_ring_vel", "advisoryautonomy_ramp_acc", "advisoryautonomy_ramp_vel", "advisoryautonomy_ring_acc_0", "advisoryautonomy_ring_acc_1", "advisoryautonomy_ring_acc_2", "advisoryautonomy_ring_acc_3", "advisoryautonomy_ring_acc_4", 
                    "advisoryautonomy_ring_vel_0", "advisoryautonomy_ring_vel_1", "advisoryautonomy_ring_vel_2", "advisoryautonomy_ring_vel_3", "advisoryautonomy_ring_vel_4",
                    "advisoryautonomy_ramp_acc_0", "advisoryautonomy_ramp_acc_1", "advisoryautonomy_ramp_acc_2", "advisoryautonomy_ramp_acc_3", "advisoryautonomy_ramp_acc_4",
                    "advisoryautonomy_ramp_vel_0", "advisoryautonomy_ramp_vel_1", "advisoryautonomy_ramp_vel_2", "advisoryautonomy_ramp_vel_3", "advisoryautonomy_ramp_vel_4"]:
        num_transfer_steps = 40
    elif data_name in ['no-stop_green_0', 'no-stop_green_1', 'no-stop_green_2', 'no-stop_penrate_0', 'no-stop_penrate_1', 'no-stop_penrate_2', 'no-stop_inflow_0', 'no-stop_inflow_1', 'no-stop_inflow_2',
                       "intersection_flow_0", "intersection_flow_1", "intersection_flow_2", "intersection_speed_0", "intersection_speed_1", "intersection_speed_2", "intersection_length_0", "intersection_length_1", "intersection_length_2"]:
        num_transfer_steps = 50

    source_tasks_MBTL, transfer_results_MBTL, _, _ = MBTL(home_dir, data_name, deltas, num_transfer_steps, data_transfer, acquisition_function='new_ucb_beta_log', gap_function='linear', slope=slope)

    # Pseudogreedy Strategy (from T-RO)
    source_tasks_gttl = greedy_temporal_transfer_learning(deltas, num_transfer_steps, delta_min=delta_min, delta_max=delta_max)
    J_gttl = collect_J_matrix(data_transfer, source_tasks_gttl, deltas, num_transfer_steps)
    performance_greedy_temporal_transfer_learning = J_gttl.mean(axis=0)
    
    # Equidistant (C2F) Strategy (from T-RO)
    source_tasks_cttl = coarse_to_fine_temporal_transfer_learning(deltas, num_transfer_steps, delta_min, delta_max)
    J_cttl = collect_J_matrix(data_transfer, source_tasks_cttl, deltas, num_transfer_steps)
    coarse_to_fine_transfer_training = J_cttl.mean(axis=0)

    oracle_transfer = [data_transfer.max(axis=0).mean()] * num_transfer_steps
    
    # diagonal of data_transfer
    data_transfer_diagonal = np.zeros(len(deltas))
    for i in range(len(deltas)):
        data_transfer_diagonal[i] = data_transfer.iloc[i][i]
        
    exhaustive_training = [data_transfer_diagonal.mean()] * num_transfer_steps
    
    # Sequential oracle training
    sequential_oracle_training = []
    sot_deltas = []
    
    # 1st step
    sot_deltas.append(data_transfer.mean(axis=1).argmax())
    sequential_oracle_training.append(data_transfer.iloc[data_transfer.mean(axis=1).argmax(),:].mean())
    for idx in range(num_transfer_steps-1):
        candidate_indices = [x for x in range(len(deltas)) if x not in sot_deltas]
        index_tmp = [data_transfer.T[sot_deltas+[i]].max(axis=1).mean() for i in candidate_indices].index(max([data_transfer.T[sot_deltas+[i]].max(axis=1).mean() for i in candidate_indices]))
        sot_deltas.append(candidate_indices[index_tmp])
        sequential_oracle_training.append(data_transfer.T[sot_deltas].max(axis=1).mean())

    source_tasks_random = [np.random.choice(range(len(deltas)), num_transfer_steps, replace=False) for _ in range(100)]
    performance_random = [evaluate_on_task(data_transfer, source_task_random, deltas, num_transfer_steps) for source_task_random in source_tasks_random]

    performance_random_mean = []
    performance_random_std = []
    for j in range(num_transfer_steps):
        performance_random_mean.append(np.mean([performance_random[i][j] for i in range(100)]))
        performance_random_std.append(np.std([performance_random[i][j] for i in range(100)]))
        
    if "advisoryautonomy" in data_name:
        if "ring" in data_name:
            multitask_data = pd.read_csv(f'{home_dir}/MBTL/advisoryautonomy-ring-multi.csv')
        else:
            multitask_data = pd.read_csv(f'{home_dir}/MBTL/advisoryautonomy-ramp-multi.csv')
        multitask_data = multitask_data[multitask_data["multi"]==40]

        if "acc" in data_name:
            multitask_data = multitask_data[multitask_data["ctrl_type"]=='acccon']
        else:
            multitask_data = multitask_data[multitask_data["ctrl_type"]=='veldis10']
        if "0" in data_name:
            multitask1 = np.array(multitask_data[multitask_data["iter"]==0]["speed_avg"])
        elif "1" in data_name:
            multitask1 = np.array(multitask_data[multitask_data["iter"]==1]["speed_avg"])
        else:
            multitask1 = np.array(multitask_data[multitask_data["iter"]==2]["speed_avg"])
        for i in range(multitask1.shape[0]):
            multitask1[i] = (multitask1[i] - lower_bound) / (upper_bound - lower_bound)

        multitask_final = [multitask_final] * num_transfer_steps
        
    # Save results
    result = pd.DataFrame(columns=range(num_transfer_steps))
    result.loc['MBTL'] = transfer_results_MBTL
    result.loc['Pseudogreedy Strategy'] = performance_greedy_temporal_transfer_learning
    result.loc['Equidistant Strategy (100)'] = coarse_to_fine_transfer_training
    result.loc['Oracle Transfer'] = oracle_transfer
    result.loc['Independent Training'] = exhaustive_training
    result.loc['Sequential Oracle Training'] = sequential_oracle_training
    result.loc['Random_mean'] = performance_random_mean
    result.loc['Random_stdev'] = performance_random_std
    result['Average'] = result.mean(axis=1)

    result.to_csv(f'{home_dir}/MBTL/alg_raw_results_{data_name}_{num_transfer_steps}.csv')
    