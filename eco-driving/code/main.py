from pathlib import Path

import argparse
from containers.config import Config
from containers.constants import *
from experiment import MainExperiment
from containers.task_context import ContinuousSelector, NetGenTaskContext, PathTaskContext

if __name__ == '__main__':
    # set the number of workers here
    parser = argparse.ArgumentParser(description='Model arguments')
    parser.add_argument('--dir', default='wd/new_exp', type=str, help='Result directory')
    parser.add_argument('--source_path', type=str, help='Source directory')
    parser.add_argument('--kwargs', default='{}', help='args to be added to the config')
    parser.add_argument('--task_context_kwargs', default='{}', help='args to be added to the task_context')
    parser.add_argument('--inflow', default=300, type=float)
    parser.add_argument('--penrate', default=0.2, type=float)
    parser.add_argument('--green', default=35, type=float)
    args = parser.parse_args()

    task = NetGenTaskContext(
        base_id=[11],
        penetration_rate=[args.penrate],
        single_approach=True,
        inflow=[args.inflow],
        lane_length=[250],
        speed_limit=[14],
        green_phase=[args.green], 
        red_phase=[35],  
        offset=[0],
    )

    # training config
    config = Config(
        run_mode='train',
        task_context=task,
        working_dir=Path(args.dir),
        source_dir=Path(args.source_path+"/models/model-5000.pth") if args.source_path is not None else None,

        wandb_proj='xxx',
        visualize_sumo=False,
        
        stop_penalty=35,
        emission_penalty=3,
        fleet_reward_ratio=0.0,
        
        moves_emissions_models=['68_46'], #'44_74', '44_64', '30_79'
        moves_emissions_models_conditions=['68_46'], #'44_74', '44_64', '30_79'
    )

    assert len(config.moves_emissions_models) == len(config.moves_emissions_models_conditions), "The evaluations conditions does not have the same dimensions"
    config = config.update({**eval(args.kwargs)})

    main_exp = MainExperiment(config)

    # run experiment
    main_exp.run()
