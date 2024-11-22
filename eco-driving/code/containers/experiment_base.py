import logging
import re
from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer

from containers.config import Config
from misc_utils import to_torch


class ExperimentBase:
    def __init__(self, config: Config):
        self.config = config
        self.models_dir.mkdir(exist_ok=True, parents=True)

    @property
    def models_dir(self) -> Path:
        return self.config.working_dir / 'models'

    @property
    def model_file_best(self) -> Path:
        return self.models_dir / 'best_model.pth'

    def model_file_with_index(self, step: int) -> Path:
        return self.models_dir / ('model-%s.pth' % step)

    def set_state(self, net, opt=None, step='max', path: Path = None) -> any:
        if path is None:
            if step == 'best':
                path = self.model_file_best
            else:
                if step == 'max':
                    steps = sorted([x for x in map(_model_step, self.models_dir.iterdir()) if x is not None])
                    if len(steps) == 0:
                        return 0
                    step = max(steps)
                path = self.model_file_with_index(step)
        if path.exists():
            state = to_torch(torch.load(path), device=self.config.device)
        else:
            return 0

        net.load_state_dict(state['net'])
        if opt:
            if 'opt' in state:
                opt.load_state_dict(state['opt'])
            else:
                logging.warning('No state for optimizer to load')

        return state.get('step', 0)

    def save_state(self, step: int, state: dict, link_best: bool = False) -> Path:
        save_path = self.model_file_with_index(step)
        if save_path.exists():
            return save_path
        torch.save(state, save_path)
        logging.info('Saved model %s at step %s' % (save_path, step))
        if link_best:
            self.model_file_best.link_to(save_path)
            logging.info('Linked %s to new saved model %s' % (self.model_file_best, save_path))
        return save_path


def _model_step(path: Path) -> Optional[int]:
    m = re.match('.+/model-(\d+)\.pth', str(path))
    if m:
        return int(m.groups()[0])
    return None


def get_state(net: Module, opt: Optimizer, step: int) -> dict:
    try:
        net_dict = net.module.state_dict()
    except AttributeError:
        net_dict = net.state_dict()
    state = dict(step=step, net=net_dict, opt=opt.state_dict())
    return to_torch(state, device='cpu')