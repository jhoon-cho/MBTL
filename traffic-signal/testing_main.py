from __future__ import absolute_import
from __future__ import print_function

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import DQN
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='settings/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path = set_test_path(config['models_path_name'])

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'],
        output_dim=config['num_actions'],
        path=model_path,
        checkpoint=config['model_to_test']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
        
    Simulation = Simulation(
        DQN,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test Episode')
    # run the simulation
    simulation_time, avg_reward, avg_waiting = Simulation.run(config['episode_seed'])
    print("\t [STAT] Average reward:", avg_reward,
          "Average waiting time:", avg_waiting,
          "Simulation time:", simulation_time, 's')
    print('\n----- End Test Episode')
