import numpy as np
import torch
from torch import nn

from RL.utils import get_preprocessed_obs_len
from containers.config import Config
from containers.nammed_arrays import NamedArrays
from gym.spaces import Dict as GymDict


class FFN(nn.Module):
    def __init__(self, config: Config, observation_space, dist_class):
        super().__init__()

        self.config = config
        self.dist_class = dist_class
        layers = list(self.config.layers)
        layers = {'s': [], 'v': layers, 'p': layers}
        s_sizes = [get_preprocessed_obs_len(observation_space) if isinstance(observation_space, GymDict) else observation_space.shape[0], *layers['s']]

        # by default, self.shared is none (no shared network between policy and critic)
        self.shared = _build_fc(*s_sizes)

        self.p_head = _build_fc(s_sizes[-1], *layers['p'], self.dist_class.model_output_size)
        self.sequential_init(self.p_head, 'policy')
        self.v_head = None
        if True:  # self.config.use_critic:
            self.v_head = _build_fc(s_sizes[-1], *layers['v'], 1)
            self.sequential_init(self.v_head, 'value')

    def sequential_init(self, seq, key):
        linears = [m for m in seq if isinstance(m, nn.Linear)]

        for i, m in enumerate(linears):
            if isinstance(self.config.weight_scale, (int, float)):
                scale = self.config.weight_scale
            elif isinstance(self.config.weight_scale, (list, tuple)):
                scale = self.config.weight_scale[i]
            elif isinstance(self.config.weight_scale, dict):
                scale = self.config.weight_scale[key][i]
            else:
                scale = 0.01 if m == linears[-1] else 1

            if self.config.weight_init == 'normc':  # normalize along input dimension
                weight = torch.randn_like(m.weight)
                m.weight.data = weight * scale / weight.norm(dim=1, keepdim=True)
            elif self.config.weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=scale)
            elif self.config.weight_init == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=scale)

            nn.init.zeros_(m.bias)

    def forward(self, inp, value=False, policy=False, argmax=None):
        s = self.shared(inp)
        pred = {}
        if value and self.v_head:
            pred['value'] = self.v_head(s).view(-1)
        if policy or argmax is not None:
            pred['policy'] = self.p_head(s)
            if argmax is not None:
                if len(pred['policy']) == 0:
                    # This is a weird corner case, happens rarely but enough that it will make training fails.
                    # The simplest way to fix this would be to remove the (arguably bad) NamedArray class
                    pred['action'] = NamedArrays(
                        {'accel': torch.Tensor(torch.Size((0, 1))),
                         'lane_change': torch.Tensor(torch.Size((0, 3)))}
                        if self.config.control_lane_change
                        else {'accel': torch.Tensor(torch.Size((0, 1)))}
                    )
                else:
                    dist = self.dist_class(pred['policy'])
                    pred['action'] = dist.argmax() if argmax else dist.sample()
        return pred


def _build_fc(input_size, *sizes_and_modules):
    """
    Build a fully connected network
    """
    layers = []
    str_map = dict(relu=nn.ReLU(inplace=True),
                   tanh=nn.Tanh(),
                   sigmoid=nn.Sigmoid(),
                   flatten=nn.Flatten(),
                   softmax=nn.Softmax())
    for x in sizes_and_modules:
        if isinstance(x, (int, np.integer)):
            input_size, x = x, nn.Linear(input_size, x)
        if isinstance(x, str):
            x = str_map[x]
        layers.append(x)
    return nn.Sequential(*layers)