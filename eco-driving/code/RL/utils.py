from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.signal
import torch

import torch.nn.functional as F
from gym import Space
from gym.spaces import Box, Discrete, Dict as GymDict
from torch import optim
from torch.nn import Module

from containers.config import Config
from sumo.constants import SPEED_NORMALIZATION, GLOBAL_MAX_LANE_LENGTH, GLOBAL_MAX_SPEED, LANE_LENGTH_NORMALIZATION,\
    MAX_TL_CYCLE, TL_CYCLE_NORMALIZATION
from misc_utils import singleton_np_array as npa


def get_observation_space(config: Config) -> GymDict:
    """
    Generates obs space
    """
    speed_space = Box(npa(0), npa(GLOBAL_MAX_SPEED/SPEED_NORMALIZATION))

    other_vehicle_space = GymDict({
        'speed': speed_space,
        'relative_position': Box(npa(-1), npa(GLOBAL_MAX_LANE_LENGTH/LANE_LENGTH_NORMALIZATION)),
        'blinker_left': Discrete(2),
        'blinker_right': Discrete(2)
    })
    
    obs_space = GymDict({
        'speed': speed_space,
        'relative_distance': Box(npa(-GLOBAL_MAX_LANE_LENGTH/LANE_LENGTH_NORMALIZATION), npa(GLOBAL_MAX_LANE_LENGTH/LANE_LENGTH_NORMALIZATION)),
        'tl_phase': Discrete(3),
        'time_remaining': Box(npa(0), npa(MAX_TL_CYCLE/TL_CYCLE_NORMALIZATION)),
        'time_remaining2': Box(npa(0), npa(2 * MAX_TL_CYCLE / TL_CYCLE_NORMALIZATION)),
        'time_remaining3': Box(npa(0), npa(3 * MAX_TL_CYCLE / TL_CYCLE_NORMALIZATION)),
        'edge_id': Discrete(3),  # 4 incoming, 4 outgoing, and 1 for internal lanes
        'follower': other_vehicle_space,
        'leader': other_vehicle_space,
        'lane_index': Box(npa(0), npa(1)),
        'destination': Discrete(3),  # left, straight or right
        'leader_left': other_vehicle_space,
        'leader_right': other_vehicle_space,
        'follower_left': other_vehicle_space,
        'follower_right': other_vehicle_space,

        # context, stay constant
        'penetration_rate': Box(npa(0), npa(1)),
        'lane_length': Box(npa(0), npa(GLOBAL_MAX_LANE_LENGTH/LANE_LENGTH_NORMALIZATION)),
        'speed_limit': speed_space,
        'green_phase': Box(npa(15), npa(120)),
        'red_phase': Box(npa(15), npa(120)),
    })

    return obs_space


def get_action_space(config: Config) -> Space:
    if config.control_lane_change:
        return GymDict({
            # the boundaries of box are not followed, -20 and 10 should include all possible vehicle types
            'accel': Box(low=-20, high=10, shape=(1,), dtype=np.float32),
            'lane_change': Discrete(3)  # left, stay, right
        })
    else:
        return GymDict({
            # the boundaries of box are not followed, -20 and 10 should include all possible vehicle types anyway
            'accel': Box(low=-20, high=10, shape=(1,), dtype=np.float32),
        })


def get_optimizer(config: Config, lr: float, model: Module) -> torch.optim.Optimizer:
    if config.opt == 'Adam':
        return optim.Adam(model.parameters(),
                          lr=lr,
                          betas=config.betas,
                          weight_decay=config.l2)
    elif config.opt == 'RMSprop':
        return optim.RMSprop(model.parameters(),
                             lr=lr,
                             weight_decay=config.l2)
    else:
        raise "No optimizer defined"


def explained_variance(y_pred, y_true):
    if not len(y_pred):
        return np.nan
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def discount(x, gamma):
    if isinstance(x, torch.Tensor):
        n = x.size(0)
        return F.conv1d(F.pad(x, (0, n - 1)).view(1, 1, -1), gamma ** torch.arange(n, dtype=x.dtype).view(1, 1, -1)).view(-1)
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def calc_adv(reward, gamma, value=None, lam=None):
    """
    Calculate advantage with TD-lambda
    """
    if value is None:
        return discount(reward, gamma), None  # TD(1)
    if isinstance(reward, list):
        reward = np.array([np.array(x) for x in reward])
        value = np.array([np.array(x) for x in value])
    assert value.ndim == reward.ndim == 1, f'Value and reward be one dimensional, but got {value.shape} and {reward.shape} respectively'
    assert value.shape[0] - reward.shape[0] in [0, 1], f'Value\'s shape can be at most 1 bigger than reward\'s shape, but got {value.shape} and {reward.shape} respectively'

    if value.shape[0] == reward.shape[0]:
        delta = reward - value
        delta[:-1] += gamma * value[1:]
    else:
        delta = reward + gamma * value[1:] - value[:-1]
    adv = discount(delta, gamma * lam)
    ret = value[:len(adv)] + adv  # discount(reward, gamma)
    return ret, adv


def calc_adv_multi_agent(id_, reward, gamma, value_=None, lam=None):
    """
    Calculate advantage with TD-lambda for multiple agents
    id_ and value_ include the last time step, reward does not include the last time step
    id_ should be something that pandas.Series.groupby works on
    id_, reward, and value_ should be flat arrays with "shape" n_steps * n_agent_per_step
    """
    n_id = len(reward)  # number of ids BEFORE the last time step
    ret = np.empty((n_id,), dtype=np.float32)
    adv = ret.copy()
    for _, group in pd.Series(id_).groupby(id_):
        idxs = group.index
        value_i_ = None if value_ is None else value_[idxs]
        if idxs[-1] >= n_id:
            idxs = idxs[:-1]
        ret[idxs], adv[idxs] = calc_adv(reward=reward[idxs], gamma=gamma, value=value_i_, lam=lam)
    return ret, adv


def preprocess_obs(obs: Dict[str, any], obs_space: GymDict) -> List[float]:
    """
    Transforms the given observation dict into a np array, transforms discrete fields into one hot encoding.
    """
    
    res = []

    for key in sorted(obs):
        obs_field = obs_space[key]
        if isinstance(obs_field, Discrete):
            # boolean
            if obs_field.n == 2:
                res.append(int(obs[key]))
            # One hot encoding
            else:
                res.extend([1 if i == obs[key] else 0 for i in range(obs_field.n)])
        elif isinstance(obs_field, Box) and (obs_field.shape == (1,) or obs_field.shape == (1, 1)):
            res.append(obs[key])
        elif isinstance(obs_field, Box) and obs_field.shape[1] == 1:
            res.extend(obs[key])
        elif isinstance(obs_field, GymDict):
            res.extend(preprocess_obs(obs[key], obs_field))
        else:
            raise NotImplementedError(f'{obs_field}, {type(obs_field)} has not been implemented yet')
    return res


def get_preprocessed_obs_len(obs_space: GymDict) -> int:
    """
    Takes a Gym dict Space and returns the length of the numpy array resulting of preprocess_obs() method.
    """
    return len(preprocess_obs(obs_space.sample(), obs_space))