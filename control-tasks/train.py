import os
import argparse
import json
import importlib
import warnings

from carl.envs import CARLCartPoleEnv_defaults as cartpole_default
from carl.envs import CARLLunarLanderEnv_defaults as lunarlander_default
from carl.envs import CARLMountainCarEnv_defaults as mountaincar_default
from carl.envs import CARLPendulumEnv_defaults as pendulum_default
from carl.envs import CARLBipedalWalkerEnv_defaults as walker_default
from carl.envs import CARLCartPoleEnv, CARLPendulumEnv, CARLBipedalWalkerEnv, CARLLunarLanderEnv, CARLMountainCarContinuousEnv, CARLMountainCarEnv

brax_spec = importlib.util.find_spec("brax")
found = brax_spec is not None
if found:
    from carl.envs import CARLHalfcheetah_defaults as halfcheetah_default
    from carl.envs import CARLAnt_defaults as ant_default
    from carl.envs import CARLHalfcheetah, CARLAnt
    pass
else:
    warnings.warn(
        "Module 'Brax' not found. If you want to use these environments, please follow the installation guide."
    )

from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
from sb3_contrib import TRPO, ARS
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', dest='env_name', type=str, help='environment type')

# cartpole
parser.add_argument('--pole_length', dest='pole_length', default=0.5, type=float)
parser.add_argument('--masscart', dest='masscart', default=1.0, type=float)
parser.add_argument('--masspole', dest='masspole', default=0.1, type=float)
parser.add_argument('--force_magnifier', dest='force_magnifier', default=10.0, type=float)
parser.add_argument('--gravity', dest='gravity', default=9.8, type=float)
parser.add_argument('--update_interval', dest='update_interval', default=0.02, type=float)

# lunar lander
parser.add_argument('--main_engine_power', dest='main_engine_power', default=13.0, type=float)
parser.add_argument('--side_engine_power', dest='side_engine_power', default=0.6, type=float)
parser.add_argument('--initial_random', dest='initial_random', default=1000.0, type=float)
parser.add_argument('--leg_away', dest='leg_away', default=20.0, type=float)
parser.add_argument('--leg_down', dest='leg_down', default=18.0, type=float)

# mountain car
parser.add_argument('--mc_max_speed', dest='mc_max_speed', default=0.07, type=float)
parser.add_argument('--mc_power', dest='mc_power', default=0.0015, type=float)
parser.add_argument('--mc_force', dest='mc_force', default=0.001, type=float)
parser.add_argument('--mc_goal_position', dest='mc_goal_position', default=0.45, type=float)

# pendulum
parser.add_argument('--pd_max_speed', dest='pd_max_speed', default=8.0, type=float)
parser.add_argument('--pd_dt', dest='pd_dt', default=0.05, type=float)
parser.add_argument('--pd_g', dest='pd_g', default=10, type=float)
parser.add_argument('--pd_m', dest='pd_m', default=1, type=float)
parser.add_argument('--pd_l', dest='pd_l', default=1, type=float)

# Walker
parser.add_argument('--wk_GRAVITY_Y', dest='wk_GRAVITY_Y', default=10, type=float)
parser.add_argument('--wk_FRICTION', dest='wk_FRICTION', default=2.5, type=float)
parser.add_argument('--wk_MOTORS_TORQUE', dest='wk_MOTORS_TORQUE', default=80, type=float)
parser.add_argument('--wk_SCALE', dest='wk_SCALE', default=30, type=float)
parser.add_argument('--wk_LEG_H', dest='wk_LEG_H', default=1.13, type=float)
parser.add_argument('--wk_LEG_W', dest='wk_LEG_W', default=0.26, type=float)

# Half Cheetah
parser.add_argument('--hc_joint_stiffness', dest='hc_joint_stiffness', default=15000, type=float)
parser.add_argument('--hc_gravity', dest='hc_gravity', default=9.8, type=float)
parser.add_argument('--hc_friction', dest='hc_friction', default=0.6, type=float)
parser.add_argument('--hc_angular_damping', dest='hc_angular_damping', default=0.05, type=float)
parser.add_argument('--hc_joint_angular_damping', dest='hc_joint_angular_damping', default=20, type=float)
parser.add_argument('--hc_torso_mass', dest='hc_torso_mass', default=9.457333, type=float)

# Ant
parser.add_argument('--ant_joint_stiffness', dest='ant_joint_stiffness', default=5000, type=float)
parser.add_argument('--ant_gravity', dest='ant_gravity', default=9.8, type=float)
parser.add_argument('--ant_friction', dest='ant_friction', default=0.6, type=float)
parser.add_argument('--ant_angular_damping', dest='ant_angular_damping', default=-0.05, type=float)
parser.add_argument('--ant_joint_angular_damping', dest='ant_joint_angular_damping', default=35, type=float)
parser.add_argument('--ant_torso_mass', dest='ant_torso_mass', default=10, type=float)
parser.add_argument('--ant_actuator_strength', dest='ant_actuator_strength', default=300, type=float)

# common config
parser.add_argument('--total_steps', dest='total_steps', default=2000000, type=int, help='number of training steps')
parser.add_argument('--save_freq', dest='save_freq', default=500000, type=int, help='frequency of saving checkpoints')
parser.add_argument('--save_path', dest='save_path', default="run_logs", type=str, help='path for model savings')
parser.add_argument('--alg', dest='alg', default="DQN", type=str, help='RL algorithm for training')
parser.add_argument('--test_eps', dest='test_eps', default=10, type=int, help='number of testing episodes')
args = parser.parse_args()

# create log dir
os.makedirs(args.save_path, exist_ok=True)

# save args to a file
with open(args.save_path + '/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": args.total_steps,
    "env_name": args.env_name,
}

if args.env_name == "cartpole":
    new_context = cartpole_default.copy()
    new_context["pole_length"] = args.pole_length
    new_context["masscart"] = args.masscart
    new_context["masspole"] = args.masspole
    new_context["force_magnifier"] = args.force_magnifier
    new_context["gravity"] = args.gravity
    new_context["update_interval"] = args.update_interval
    contexts = {0: new_context}
    env = CARLCartPoleEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLCartPoleEnv(contexts=contexts)
    
elif args.env_name == "lunarlander":
    new_context = lunarlander_default.copy()
    new_context["main_engine_power"] = args.main_engine_power
    new_context["side_engine_power"] = args.side_engine_power
    new_context["initial_random"] = args.initial_random
    new_context["leg_away"] = args.leg_away
    new_context["leg_down"] = args.leg_down
    contexts = {0: new_context}
    env = CARLLunarLanderEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLLunarLanderEnv(contexts=contexts)
    
elif args.env_name == "mountaincar":
    new_context = mountaincar_default.copy()
    new_context["force"] = args.mc_force
    new_context["goal_position"] = args.mc_goal_position
    new_context["max_speed"] = args.mc_max_speed
    contexts = {0: new_context}
    env = CARLMountainCarEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLMountainCarEnv(contexts=contexts)
    
elif args.env_name == "mountaincarcont":
    new_context = mountaincar_default.copy()
    new_context["power"] = args.mc_power
    new_context["goal_position"] = args.mc_goal_position
    new_context["max_speed"] = args.mc_max_speed
    contexts = {0: new_context}
    env = CARLMountainCarContinuousEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLMountainCarContinuousEnv(contexts=contexts)
    
elif args.env_name == "pendulum":
    new_context = pendulum_default.copy()
    new_context["max_speed"] = args.pd_max_speed
    new_context["dt"] = args.pd_dt
    new_context["g"] = args.pd_g
    new_context["m"] = args.pd_m
    new_context["l"] = args.pd_l
    contexts = {0: new_context}
    env = CARLPendulumEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLPendulumEnv(contexts=contexts)
    
elif args.env_name == "walker":
    new_context = walker_default.copy()
    new_context["GRAVITY_Y"] = -args.wk_GRAVITY_Y # be careful about the sign
    new_context["SCALE"] = args.wk_SCALE
    new_context["FRICTION"] = args.wk_FRICTION
    new_context["MOTORS_TORQUE"] = args.wk_MOTORS_TORQUE
    new_context["LEG_H"] = args.wk_LEG_H
    new_context["LEG_W"] = args.wk_LEG_W
    contexts = {0: new_context}
    env = CARLBipedalWalkerEnv(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLBipedalWalkerEnv(contexts=contexts)
    
elif args.env_name == "ant":
    new_context = ant_default.copy()
    new_context["joint_stiffness"] = args.ant_joint_stiffness
    new_context["gravity"] = -args.ant_gravity
    new_context["friction"] = args.ant_friction
    new_context["angular_damping"] = -args.ant_angular_damping
    new_context["joint_angular_damping"] = args.ant_joint_angular_damping
    new_context["torso_mass"] = args.ant_torso_mass
    new_context["actuator_strength"] = args.ant_actuator_strength
    contexts = {0: new_context}
    env = CARLAnt(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLAnt(contexts=contexts)
    
elif args.env_name == "halfcheetah":
    new_context = halfcheetah_default.copy()
    new_context["joint_stiffness"] = args.hc_joint_stiffness
    new_context["gravity"] = -args.hc_gravity
    new_context["friction"] = args.hc_friction
    new_context["angular_damping"] = -args.hc_angular_damping
    new_context["joint_angular_damping"] = args.hc_joint_angular_damping
    new_context["torso_mass"] = args.hc_torso_mass
    contexts = {0: new_context}
    env = CARLHalfcheetah(contexts=contexts)
    
    # Separate evaluation env
    eval_env = CARLHalfcheetah(contexts=contexts)
    
else:
    print("Environment not recognized!")
    exit()


env = Monitor(env, filename=args.save_path + "/" + args.env_name + "_data",)
eval_env = Monitor(eval_env, filename=args.save_path + "/" + args.env_name + "_eval_data",)

eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path, log_path=args.save_path, eval_freq=500)

checkpoint_callback = CheckpointCallback(
  save_freq=args.save_freq,
  save_path=f"{args.save_path}/runs/source_task_training_checkpoints/",
  name_prefix="model",
)

callbacks = CallbackList([checkpoint_callback, eval_callback])

logger = configure(args.save_path + "/" + args.env_name + "_log", ["stdout", "csv", "log"])

# Algorithm selection
algs = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
    "TRPO": TRPO,
    "ARS": ARS,
}

if args.alg not in algs:
    raise ValueError(f"RL algorithm '{args.alg}' is not valid")

model_class = algs[args.alg]
model = model_class('MlpPolicy', env, verbose=1, tensorboard_log=f"{args.save_path}/runs")
model.set_logger(logger)
model.learn(total_timesteps=args.total_steps, callback=callbacks)

print("Training completed!!")