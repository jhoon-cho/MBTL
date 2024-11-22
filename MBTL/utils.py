import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import os

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import truncnorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C


def get_dataset_settings(home_dir, data_name):
    # Initialize settings dictionary
    settings = {}
    
    # Define settings for each dataset
    if data_name in ["advisoryautonomy_ring_acc", "advisoryautonomy_ring_vel"]:
        settings = {
            'lower_bound': 0,
            'unguided': 3.6,
            'upper_bound': 4.5,
            'delta_min': 0,
            'delta_max': 39,
            'slope': 0.005
        }
    elif data_name in ["advisoryautonomy_ring_acc_0", "advisoryautonomy_ring_acc_1", "advisoryautonomy_ring_acc_2", "advisoryautonomy_ring_acc_3", "advisoryautonomy_ring_acc_4", "advisoryautonomy_ring_vel_0", "advisoryautonomy_ring_vel_1", "advisoryautonomy_ring_vel_2", "advisoryautonomy_ring_vel_3", "advisoryautonomy_ring_vel_4"]:
        settings = {
            'lower_bound': 0,
            'unguided': 3.6,
            'upper_bound': 4.5,
            'delta_min': 0,
            'delta_max': 39,
            'slope': 0.005
        }
    elif data_name in ["advisoryautonomy_ramp_acc", "advisoryautonomy_ramp_vel"]:
        settings = {
            'lower_bound': 0,
            'unguided': 3.9299,
            'upper_bound': 8.78,
            'delta_min': 0,
            'delta_max': 39,
            'slope': 0.005
        }
    elif data_name in ["advisoryautonomy_ramp_acc_0", "advisoryautonomy_ramp_acc_1", "advisoryautonomy_ramp_acc_2", "advisoryautonomy_ramp_acc_3", "advisoryautonomy_ramp_acc_4", "advisoryautonomy_ramp_vel_0", "advisoryautonomy_ramp_vel_1", "advisoryautonomy_ramp_vel_2", "advisoryautonomy_ramp_vel_3", "advisoryautonomy_ramp_vel_4"]:
        settings = {
            'lower_bound': 0,
            'unguided': 3.9299,
            'upper_bound': 8.78,
            'delta_min': 0,
            'delta_max': 39,
            'slope': 0.005
        }
    elif data_name in ["advisoryautonomy_inter_acc", "advisoryautonomy_inter_vel"]:
        settings = {
            'lower_bound': 0,
            'unguided': 6.8379,
            'upper_bound': 7.7092,
            'delta_min': 1,
            'delta_max': 400,
            'slope': 0.005
        }
    elif data_name in ["intersection_flow", "intersection_speed", "intersection_length"]:
        settings = {
            'lower_bound': -100,
            'unguided': -100,
            'upper_bound': 0,
            'delta_min': 0,
            'delta_max': 19,
            'slope': 0.005
        }
    elif data_name in ["intersection_flow_0", "intersection_flow_1", "intersection_flow_2", "intersection_speed_0", "intersection_speed_1", "intersection_speed_2", "intersection_length_0", "intersection_length_1", "intersection_length_2"]:
        settings = {
            'lower_bound': -100,
            'unguided': -100,
            'upper_bound': 0,
            'delta_min': 0,
            'delta_max': 49,
            'slope': 0.001
        }
    elif data_name == 'cartpole_masscart':
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 500,
            'delta_min': 0,
            'delta_max': 48,
            'slope': 0.005
        }
    elif data_name == 'cartpole_masscart_lenpole':
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 500,
            'delta_min': 0,
            'delta_max': 49,
            'slope': 0.005
        }
    elif data_name in ['cartpole_lenpole', 'cartpole_masspole']:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 500,
            'delta_min': 0,
            'delta_max': 98,
            'slope': 0.005
        }
    elif data_name in ['cartpole_lenpole_0','cartpole_lenpole_1','cartpole_lenpole_2','cartpole_masspole_0','cartpole_masspole_1','cartpole_masspole_2',"cartpole_masscart_0", "cartpole_masscart_1", "cartpole_masscart_2", "cartpole_lenpole_ppo_0", "cartpole_lenpole_ppo_1", "cartpole_lenpole_ppo_2", "cartpole_masspole_ppo_0", "cartpole_masspole_ppo_1", "cartpole_masspole_ppo_2", "cartpole_masscart_ppo_0", "cartpole_masscart_ppo_1", "cartpole_masscart_ppo_2", "cartpole_lenpole_a2c_0", "cartpole_lenpole_a2c_1", "cartpole_lenpole_a2c_2", "cartpole_masspole_a2c_0", "cartpole_masspole_a2c_1", "cartpole_masspole_a2c_2", "cartpole_masscart_a2c_0", "cartpole_masscart_a2c_1", "cartpole_masscart_a2c_2"]:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 500,
            'delta_min': 0,
            'delta_max': 99,
            'slope': 0.005
        }
    elif data_name in ['cartpole_lenpole','cartpole_masspole',"cartpole_masscart","cartpole_lenpole_ppo","cartpole_masspole_ppo", "cartpole_masscart_ppo", "cartpole_lenpole_a2c", "cartpole_masspole_a2c", "cartpole_masscart_a2c"]:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 500,
            'delta_min': 0,
            'delta_max': 99,
            'slope': 0.005
        }
    elif data_name == "lunarlander_mainenginepower":
        settings = {
            'lower_bound': -600,
            'unguided': -600,
            'upper_bound': 300,
            'delta_min': 1,
            'delta_max': 50,
            'slope': 0.005
        }
    elif data_name in ["pendulum_l", "pendulum_m", "pendulum_dt"]:
        settings = {
            'lower_bound': -2200,
            'unguided': -1300,
            'upper_bound': -500,
            'delta_min': 1,
            'delta_max': 50,
            'slope': 0.005
        }
    elif data_name in ["pendulum_dt_0", "pendulum_dt_1", "pendulum_dt_2", "pendulum_l_0", "pendulum_l_1", "pendulum_l_2", "pendulum_m_0", "pendulum_m_1", "pendulum_m_2"]:
        settings = {
            'lower_bound': -2200,
            'unguided': -1300,
            'upper_bound': -500,
            'delta_min': 0,
            'delta_max': 99,
            'slope': 0.005
        }
    elif data_name in ["walker_friction", "walker_gravity", "walker_scale", "walker_friction_0", "walker_friction_1", "walker_friction_2", "walker_gravity_0", "walker_gravity_1", "walker_gravity_2", "walker_scale_0", "walker_scale_1", "walker_scale_2"]:
        settings = {
            'lower_bound': -200,
            'unguided': -200,
            'upper_bound': 10,
            'delta_min': 0,
            'delta_max': 99,
            'slope': 0.005
        }        
    elif data_name in ["halfcheetah_friction", "halfcheetah_gravity", "halfcheetah_stiffness", 'halfcheetah_friction_0', 'halfcheetah_friction_1', 'halfcheetah_friction_2', 'halfcheetah_gravity_0', 'halfcheetah_gravity_1', 'halfcheetah_gravity_2', 'halfcheetah_stiffness_0', 'halfcheetah_stiffness_1', 'halfcheetah_stiffness_2']:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 6000,
            'delta_min': 0,
            'delta_max': 99,
            'slope': 10
        }        
    elif data_name in ["no-stop_green", "no-stop_inflow", "no-stop_penrate", 'no-stop_green_0', 'no-stop_green_1', 'no-stop_green_2', 'no-stop_penrate_0', 'no-stop_penrate_1', 'no-stop_penrate_2', 'no-stop_inflow_0', 'no-stop_inflow_1', 'no-stop_inflow_2']:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 15,
            'delta_min': 0,
            'delta_max': 49,
            'slope': 0.005
        }        
    elif data_name in ["ideal"]:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 1,
            'delta_min': 1,
            'delta_max': 50,
            'slope': 0.005
        }
    elif data_name in ["ideal_n1","ideal_n2","ideal_n3","ideal_n4","ideal_n5","ideal_n6","ideal_n7","ideal_n8","ideal_n9","ideal_n10"]:
        settings = {
            'lower_bound': 0,
            'unguided': 0,
            'upper_bound': 1,
            'delta_min': 1,
            'delta_max': 100,
            'slope': 0.005
        }
    elif data_name in ["walker_friction", "walker_gravity", "walker_gravity_friction", "walker_friction_gravity", "walker_legh_legw", "walker_legw_legh"]:
        base_settings = {
            'lower_bound': -200,
            'unguided': -200,
            'upper_bound': 10,
            'slope': 0.005
        }
        # Customize delta_min and delta_max based on specific datasets
        if data_name in ["walker_friction"]:
            base_settings.update({'delta_min': 1, 'delta_max': 100})
        elif data_name in ["walker_gravity"]:
            base_settings.update({'delta_min': 1, 'delta_max': 20})
        elif data_name in ["walker_gravity_friction", "walker_friction_gravity"]:
            base_settings.update({'delta_min': 0, 'delta_max': 47})
        elif data_name in ["walker_legh_legw", "walker_legw_legh"]:
            base_settings.update({'delta_min': 1, 'delta_max': 16})
        settings = base_settings
    else:
        settings = {'error': "data not recognized"}
    
    return settings


def import_data(home_dir, data_name, random=False):
    settings = get_dataset_settings(home_dir, data_name)
    delta_min = settings['delta_min']
    delta_max = settings['delta_max']
    slope = settings['slope']
    lower_bound = settings['lower_bound']
    upper_bound = settings['upper_bound']
    unguided = settings['unguided']
    
    # make directory if it doesn't exist
    if not os.path.exists(home_dir+'/data/figure'):
        os.makedirs(home_dir+'/data/figure')
    if os.path.exists(home_dir+'/data/'+data_name+'_transfer_result.csv'):
        data_transfer = pd.read_csv(home_dir+'/data/'+data_name+'_transfer_result.csv', header=None)
        if 'intersection' in data_name:
            data_transfer = -data_transfer
    else:
        print("No data found for", data_name)
        
    deltas = data_transfer.columns.values.astype(float)
    
    data_transfer_norm = np.array(data_transfer)
    for i in range(data_transfer_norm.shape[0]):
        for j in range(data_transfer_norm.shape[1]):
            data_transfer_norm[i, j] = (data_transfer_norm[i, j] - lower_bound) / (upper_bound - lower_bound)
    
    data_transfer_norm = pd.DataFrame(data_transfer_norm)

    return data_transfer_norm, deltas, delta_min, delta_max, slope, lower_bound, upper_bound, unguided

# truncated normal distribution with mean=0, std=1, lower=0, upper=inf
def sample_truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)  

def greedy_heuristic_sts(data_transfer, deltas, num_transfer_steps, delta_min, delta_max, slope):
    source_tasks = np.zeros(num_transfer_steps)
    J_transfer = np.zeros((len(deltas), num_transfer_steps))
    for k in range(num_transfer_steps):
        if k == 0:
            tmp = (delta_max + delta_min)/2
        else:
            sorted_idx = np.argsort(J_transfer[:, k-1])
            for idx in range(len(sorted_idx)):
                if deltas[np.abs(deltas - sorted_idx[idx]).argmin()] in source_tasks:
                    continue
                else:
                    tmp = sorted_idx[idx]
                    break
        source_tasks[k] = deltas[np.abs(deltas - tmp).argmin()]
        for j in range(len(deltas)):
            if k==0:
                J_transfer[j, k] = data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][j]
            else:
                J_transfer[j, k] = max(data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][j], J_transfer[j, k-1])
        plt.plot(deltas, J_transfer[:, k], label='step {}'.format(k), c='C{}'.format(k))
        plt.plot(source_tasks[k], J_transfer[np.where(deltas == source_tasks[k])[0][0], k], 'o', c='C{}'.format(k))
    plt.legend(fontsize=10)
        
    return source_tasks, collect_J_matrix(data_transfer, source_tasks, deltas, num_transfer_steps=15).mean(axis=0)

def greedy_temporal_transfer_learning(deltas, num_transfer_steps, delta_min=1, delta_max=50):
    fdelta = np.array([delta_min, delta_max])
    source_tasks = np.zeros(num_transfer_steps)
    for k in range(num_transfer_steps):
        if k==0:
            tmp = (delta_max + delta_min)/2
        else:
            fdelta_diff = np.diff(fdelta)
            if fdelta_diff.argmax() == 0:
                tmp = (2*fdelta[fdelta_diff.argmax()]+fdelta[fdelta_diff.argmax()+1])/3
            elif fdelta_diff.argmax() == len(fdelta)-2:
                tmp = (fdelta[fdelta_diff.argmax()]+2*fdelta[fdelta_diff.argmax()+1])/3
            else:
                tmp = (fdelta[fdelta_diff.argmax()]+fdelta[fdelta_diff.argmax()+1])/2
        source_tasks[k] = deltas[np.abs(deltas - tmp).argmin()]
        fdelta = np.append(fdelta, source_tasks[k])
        fdelta.sort()
    return source_tasks

def coarse_to_fine_temporal_transfer_learning(deltas, budgets, delta_min=1, delta_max=50):
    source_tasks = np.zeros(budgets)
    for k in range(budgets):
        tmp = delta_max - (delta_max - delta_min) / (2*budgets) - (delta_max - delta_min) * (k) / (budgets)
        # find the closest value in deltas
        source_tasks[k] = deltas[np.abs(deltas - tmp).argmin()]
    return source_tasks

def collect_J_matrix(data_transfer, source_tasks, deltas, num_transfer_steps=15):
    J_tmp = np.zeros((len(deltas), num_transfer_steps))
    for k in range(num_transfer_steps):
        for i in range(len(deltas)):
            if k==0:
                J_tmp[i, k] = data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][i]
            else:
                J_tmp[i, k] = max(data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][i], J_tmp[i, k-1])
    return J_tmp

def MBTL(home_dir, data_name, deltas, num_transfer_steps, data_transfer, acquisition_function='new', gap_function='linear', slope=0.005, noise_std=0.1, n_restarts_optimizer=9):
    # collection of source task
    source_tasks = []
    # J_transfer[i,k]: i is the index of the target task, k is the index of the step
    J_transfer = np.zeros((len(deltas), num_transfer_steps))
    # V_estimate[i,j,k]: i is the index of the source task, j is the index of the target task, k is the index of the step
    V_estimate = np.zeros((len(deltas), len(deltas), num_transfer_steps))
    # mean_prediction[i], std_prediction[i]: i is the index of the source task
    mean_prediction = np.zeros(len(deltas))
    std_prediction = np.zeros(len(deltas))
    # V_obs_tmp[i,j]: i is the index of the source task, j is the index of the target task
    V_obs_tmp = np.zeros((len(deltas), len(deltas)))
    # V_estimate_tmp[i,j]: i is the index of the source task, j is the index of the target task
    V_estimate_tmp = np.zeros((len(deltas), len(deltas)))
    
    delta_min = min(deltas)
    delta_max = max(deltas)

    for k in range(num_transfer_steps):
        if k==0:
            tmp = (delta_max + delta_min)/2
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

            gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=n_restarts_optimizer)
            if acquisition_function == 'new_ucb_transfer':
                gaussian_process.fit(np.array(source_tasks[:k]).reshape(-1, 1), np.array([V_estimate.mean(axis=1)[np.where(deltas==source_tasks[l])[0][0], l] for l in range(k)]))
            else:
                gaussian_process.fit(np.array(source_tasks[:k]).reshape(-1, 1), np.array([J_transfer[np.where(deltas==source_tasks[l])[0][0], l] for l in range(k)]))
            mean_prediction, std_prediction = gaussian_process.predict(deltas.reshape(-1, 1), return_std=True)
            
            # diagonals of data_transfer
            data_transfer_diagonal = np.zeros(len(deltas))
            for i in range(len(deltas)):
                data_transfer_diagonal[i] = data_transfer.iloc[i][i]
            
            if acquisition_function == 'EI':
                acquisition = mean_prediction
            elif acquisition_function == 'UCB':
                acquisition = mean_prediction + 1.96*std_prediction
            elif acquisition_function == 'LCB':
                acquisition = mean_prediction - 1.96*std_prediction
            elif acquisition_function == 'VR':
                acquisition = std_prediction
            elif acquisition_function == 'new':
                # calculate new acquisition function
                new_acquisition = np.zeros(len(deltas))
                for i in range(len(deltas)):
                    new_acquisition[i] = np.mean([max(mean_prediction[i] - slope*np.abs(deltas[i]-deltas[j]) - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            elif acquisition_function == 'new_ucb':
                new_acquisition = np.zeros(len(deltas))
                lambdas = [1]*len(deltas)
                for i in range(len(deltas)):
                    new_acquisition[i] = np.mean([max(mean_prediction[i] + lambdas[i]*std_prediction[i] - slope*np.abs(deltas[i]-deltas[j]) - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            elif acquisition_function == 'new_ucb_beta':
                new_acquisition = np.zeros(len(deltas))
                # lambdas to be list starting from 1 to 0 decaying over time
                lambdas = [1/(k+1)]*len(deltas)
                for i in range(len(deltas)):
                    new_acquisition[i] = np.mean([max(mean_prediction[i] + lambdas[i]*std_prediction[i] - slope*np.abs(deltas[i]-deltas[j]) - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            elif acquisition_function == 'new_ucb_beta_log':
                new_acquisition = np.zeros(len(deltas))
                # lambdas to sqrt(log(k+1))
                lambdas = [np.sqrt(np.log(k+1))]*len(deltas)
                for i in range(len(deltas)):
                    new_acquisition[i] = np.mean([max(mean_prediction[i] + lambdas[i]*std_prediction[i] - slope*np.abs(deltas[i]-deltas[j]) - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            elif acquisition_function == 'new_ucb_dist':
                new_acquisition = np.zeros(len(deltas))
                lambda_set = sample_truncated_normal(mean=0, std=1, lower=0, upper=np.inf, size=500)
                for lambda_tmp in lambda_set:
                    for i in range(len(deltas)):
                        new_acquisition[i] += np.mean([max(mean_prediction[i] + lambda_tmp*std_prediction[i] - slope*np.abs(deltas[i]-deltas[j]) - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            elif acquisition_function == 'new_ucb_transfer':
                new_acquisition = np.zeros(len(deltas))
                lambdas = [1/(k+1)]*len(deltas)
                for i in range(len(deltas)):
                    new_acquisition[i] = np.mean([max(mean_prediction[i] + lambdas[i]*std_prediction[i] - V_obs_tmp.max(axis=0)[j], 0) for j in range(len(deltas))])
                acquisition = new_acquisition.copy()
            
            # find the next source task that maximizes acquisition funciton and is not in the source_tasks
            sorted_idx = np.argsort(-acquisition)

            for idx in range(len(sorted_idx)):
                if deltas[np.abs(deltas - sorted_idx[idx]).argmin()] in source_tasks:
                    continue
                else:
                    tmp = sorted_idx[idx]
                    break
            
        source_tasks.append(deltas[np.abs(deltas - tmp).argmin()])

        # Update J_transfer based on the new source task training
        for j in range(len(deltas)):
            if k==0:
                J_transfer[j, k] = data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][j]
            else:
                J_transfer[j, k] = max(data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][j], J_transfer[j, k-1])

        # Calculate V_estimate for diagonal elements
        V_estimate_tmp = np.zeros((len(deltas), len(deltas)))
        for i in range(len(deltas)):
            V_estimate_tmp[i, i] = mean_prediction[i]
        for l in range(k+1):
            idx = np.where(deltas==source_tasks[l])[0][0]
            V_estimate_tmp[idx, idx] = J_transfer[idx, l]

        # Calculate V_estimate for non-diagonal elements with different gap functions
        if gap_function == 'linear':
            for i in range(len(deltas)):
                for j in range(len(deltas)):
                    V_estimate_tmp[i, j] = max(V_estimate_tmp[i, i] - slope*abs(i-j), 0)
                    # V_estimate[i, j, k] = max(V_estimate[i, j, k-1], V_estimate_tmp[i, j])
                    V_estimate[i, j, k] = max(V_obs_tmp[i, j], V_estimate_tmp[i, j])
        elif gap_function == 'linear_estimated':
            # slope estimation based on trained source tasks' generalization slope
            slopes = {}
            for i in range(len(deltas)):
                if deltas[i] in source_tasks:
                    x = []
                    y = []
                    for j in range(len(deltas)):
                        x.append(abs(i-j))
                        y.append(abs(data_transfer.iloc[i,j]-data_transfer.iloc[i,i]))
                    slopes[i] = np.polyfit(x, y, 1)[0]
            for i in range(len(deltas)):
                slope = slopes[np.array(list(slopes.keys()))[np.abs(np.array(list(slopes.keys()))- i).argmin()]]
                for j in range(len(deltas)):
                    V_estimate_tmp[i, j] = max(V_estimate_tmp[i, i] - slope*abs(i-j), 0)
                    V_estimate[i, j, k] = max(V_estimate[i, j, k-1], V_estimate_tmp[i, j])
        elif gap_function == 'true':
            for i in range(len(deltas)):
                for j in range(len(deltas)):
                    V_estimate_tmp[i, j] = data_transfer.iloc[i, j]
                    V_estimate[i, j, k] = max(V_estimate[i, j, k-1], V_estimate_tmp[i, j])
        else:
            for i in range(len(deltas)):
                for j in range(len(deltas)):
                    if i == j:
                        V_estimate_tmp[i, j] = V_estimate_tmp[i, i]
                    else:
                        V_estimate_tmp[i, j] = 0
                    V_estimate[i, j, k] = max(V_estimate[i, j, k-1], V_estimate_tmp[i, j])
                    
        V_obs_tmp = np.zeros((len(deltas), len(deltas)))
        for i in range(len(deltas)):
            if deltas[i] in source_tasks:
                for j in range(len(deltas)):
                    if i == j:
                        V_obs_tmp[i, j] = data_transfer.iloc[i, i]
                    else:
                        V_obs_tmp[i, j] = data_transfer.iloc[i, j]
    transfer_results = J_transfer.mean(axis=0)

    return source_tasks, transfer_results, J_transfer, V_estimate

def evaluate_on_task(data_transfer, source_tasks, deltas, num_transfer_steps):
    assert len(source_tasks) == num_transfer_steps
    return collect_J_matrix(data_transfer, source_tasks, deltas, num_transfer_steps).mean(axis=0)

def mean_of_list(list):
    return sum(list)/len(list)

def plot_heatmap(data_transfer, home, data_name):
    plt.clf()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(8,6))
    
    plt.rcParams.update({'font.size': 12})
    plt.imshow(data_transfer, interpolation='none')
    plt.colorbar(orientation='vertical')
    plt.xlabel("Target task")
    plt.ylabel("Source task")
    plt.savefig(home+f'/MBTL/heatmap_{data_name}.png', bbox_inches="tight", dpi=500)
    
