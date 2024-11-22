from __future__ import absolute_import
from __future__ import print_function

import argparse
import pandas as pd

from transfer_simulation import Simulation
from generator import TrafficGeneratorTransfer
from model import DQN
from utils import import_transfer_configuration, set_sumo, set_transfer_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--flow', type=int, default=1000, help='Flow of cars')
    parser.add_argument('--lane', type=int, default=4, help='Number of lanes')
    parser.add_argument('--length', type=float, default=750, help='Length of lanes')
    parser.add_argument('--speed', type=float, default=14, help='Speed limit')
    parser.add_argument('--left', type=float, default=0.25, help='Left turn ratio')
    parser.add_argument('--model_num', type=int, default=1, help='Model number')
    parser.add_argument('--source_path_name', type=str, default="intersection_flow1000_lane4.0_length750.0_speed13.89_left0.25/", help='pathname')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--alg', type=str, default='DQN', help='RL algorithm')
    args = parser.parse_args()

    config = import_transfer_configuration(config_file='settings/transfer_settings.ini')
    
    # update config with the arguments
    config['n_cars_generated'] = args.flow
    config['num_lanes'] = args.lane
    config['lane_length'] = args.length
    config['speed_limit'] = args.speed
    config['left_turn'] = args.left
    config['model_num'] = args.model_num
    config['source_path_name'] = args.source_path_name
    config['num_episodes'] = args.num_episodes
    config['trial'] = args.trial
    config['alg'] = args.alg
    
    TrafficGen = TrafficGeneratorTransfer(
        config['source_path_name'],
        config['max_steps'], 
        config['n_cars_generated'],
        config['num_lanes'],
        config['lane_length'],
        config['speed_limit'],
        config['left_turn'],
        config['alg'],
        config['trial']
    )
    
    dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}_alg{config['alg']}_trial{config['trial']}"
    sumo_cmd = set_sumo(config['gui'], config['source_path_name']+'transfer/'+dir_name, config['sumocfg_file_name'], config['max_steps'])
    model_path = set_transfer_path(config['source_path_name'], config['model_num'])

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'],
        output_dim=config['num_actions'],
        path=model_path,
        checkpoint=config['model_to_test']
    )
        
    Simulation = Simulation(
        DQN,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['num_lanes'],
        config['lane_length'],
    )

    print('\n----- Test Episode')
    
    episode = 0
    result_save = pd.DataFrame(columns=['episode', 'avg_reward', 'avg_waiting', 'avg_average_speed', 'avg_current_veh', 'avg_passed_veh', 'simulation_time'])

    while episode < config['num_episodes']:    
        print('\n [INFO]----- Episode', str(episode + 1), '/', str(config['num_episodes']), '-----')

        # run the simulation
        simulation_time, avg_reward, avg_waiting, avg_average_speed, avg_current_veh, avg_passed_veh = Simulation.run(config['episode_seed']+episode)
        print("\t [STAT] Average reward:", avg_reward,
            "Average waiting time:", avg_waiting,
            "Simulation time:", simulation_time, 's')
        # save the results
        new_row = pd.DataFrame({'episode': [episode], 
                                'avg_reward': [avg_reward], 
                                'avg_waiting': [avg_waiting], 
                                'avg_average_speed': [avg_average_speed], 
                                'avg_current_veh': [avg_current_veh], 
                                'avg_passed_veh': [avg_passed_veh], 
                                'simulation_time': [simulation_time]})

        result_save = pd.concat([result_save, new_row], ignore_index=True)
        
        episode += 1
        
        print('\n----- End Test Episode')
    result_save.to_csv('results/'+config['source_path_name']+'transfer/'+dir_name + '/transfer_results_'+str(config['model_num'])+'.csv', index=False)
    
        
