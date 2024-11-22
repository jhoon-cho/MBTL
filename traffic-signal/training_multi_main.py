from __future__ import absolute_import
from __future__ import print_function

import datetime
import argparse
from training_multi_simulation import MultiSimulation
from generator_multi import TrafficGenerator
from replay_buffer import ReplayBuffer
from model import DQN
# from stable_baselines3 import DQN
from utils import import_train_configuration, set_sumo, set_train_path
import pandas as pd
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--flow', type=int, default=1000, help='Flow of cars')
    parser.add_argument('--lane', type=int, default=4, help='Number of lanes')
    parser.add_argument('--length', type=float, default=750, help='Length of lanes')
    parser.add_argument('--speed', type=float, default=14, help='Speed limit')
    parser.add_argument('--left', type=float, default=0.25, help='Left turn ratio')
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--alg', type=str, default='DQN', help='RL algorithm')
    parser.add_argument('--multi', type=str)
    args = parser.parse_args()
    
    # import default config and init config
    config = import_train_configuration(config_file='settings/training_settings.ini')
    
    # update config with the arguments
    config['n_cars_generated'] = args.flow
    config['num_lanes'] = args.lane
    config['lane_length'] = args.length
    config['speed_limit'] = args.speed
    config['left_turn'] = args.left
    config['trial'] = args.trial
    config['alg'] = args.alg
    
    if args.multi == "flow":
        dir_name = f"intersection_flowmulti_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}_alg{config['alg']}_trial{config['trial']}"
    elif args.multi == "length":
        dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_lengthmulti_speed{config['speed_limit']}_left{config['left_turn']}_alg{config['alg']}_trial{config['trial']}"
    elif args.multi == "speed":
        dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speedmulti_left{config['left_turn']}_alg{config['alg']}_trial{config['trial']}"
    else:
        dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}_alg{config['alg']}_trial{config['trial']}"
    sumo_cmd = set_sumo(config['gui'], dir_name, config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['path_name']+"/"+dir_name)

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    # TrafficGen = TrafficGenerator(
    #     config['max_steps'], 
    #     config['n_cars_generated'],
    #     config['num_lanes'],
    #     config['lane_length'],
    #     config['speed_limit'],
    #     config['left_turn'],
    #     config['alg'],
    #     config['trial']
    # )
    
    ReplyBuffer = ReplayBuffer(
        config['memory_size_max'], 
        config['memory_size_min']
    )
        
    Simulation = MultiSimulation(
        DQN,
        ReplyBuffer,
        # TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['batch_size'],
        config['learning_rate'],
        config['n_cars_generated'],
        config['num_lanes'],
        config['lane_length'],
        config['speed_limit'],
        config['left_turn'],
        config['trial'],
        args.multi,
        dir_name
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()

    project = "DQN ATL"
    if config['wandb'] == 'True':
        wandb.init(project=project, name=dir_name, config=config)
        
    result_save = pd.DataFrame(columns=['episode', 'avg_reward', 'avg_waiting', 'avg_average_speed', 'avg_current_veh', 'avg_passed_veh', 'training_time', 'simulation_time'])

    while episode < config['total_episodes']*5:
        print('\n [INFO]----- Episode', str(episode + 1), '/', str(config['total_episodes']*5), '-----')
        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / config['total_episodes'])
        # run the simulation
        simulation_time, training_time, avg_reward, avg_waiting, training_loss, avg_average_speed, avg_current_veh, avg_passed_veh = Simulation.run(episode, epsilon)
        print('\t [STAT] Simulation time:', simulation_time, 's - Training time:',
              training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
        if config['wandb'] == 'True':
            # log the training progress in wandb
            wandb.log({
                "all/training_loss": training_loss,
                "all/avg_reward": avg_reward,
                "all/avg_waiting_time": avg_waiting,
                "all/avg_average_speed": avg_average_speed,
                "all/avg_current_vehicles": avg_current_veh,
                "all/avg_passed_vehicles": avg_passed_veh,
                "all/simulation_time": simulation_time,
                "all/training_time": training_time,
                "all/entropy": epsilon}, step=episode)
        # save the results
        new_row = pd.DataFrame({'episode': [episode], 
                                'avg_reward': [avg_reward], 
                                'avg_waiting': [avg_waiting], 
                                'avg_average_speed': [avg_average_speed], 
                                'avg_current_veh': [avg_current_veh], 
                                'avg_passed_veh': [avg_passed_veh], 
                                'training_time': [training_time], 
                                'simulation_time': [simulation_time]})

        result_save = pd.concat([result_save, new_row], ignore_index=True)

        episode += 1
        print('\t [INFO] Saving the model')
        Simulation.save_model(path, episode)
    
    result_save.to_csv(path + '/training_results.csv', index=False)

    print("\n [INFO] End of Training")
    print("\t [STAT] Start time:", timestamp_start)
    print("\t [STAT] End time:", datetime.datetime.now())
    print("\t [STAT] Session info saved at:", path)
