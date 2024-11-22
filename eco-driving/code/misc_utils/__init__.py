from numbers import Number

import numpy as np
import torch


def singleton_np_array(x: Number) -> np.ndarray:
    return np.array([x], np.float32)


def flatten(x):
    return [z
            for y in x
            for z in y]


def split(x, sizes):
    return np.split(x, np.cumsum(sizes[:-1]))


def recurse(x, fn):
    if isinstance(x, dict):
        return type(x)((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(recurse(v, fn) for v in x)
    return fn(x)


def to_torch(x, device, **kwargs):
    def helper(xs):
        if xs is None:
            return None
        elif isinstance(xs, torch.Tensor):
            return xs.to(device=device, **kwargs)
        elif np.isscalar(xs):
            return xs
        return torch.from_numpy(xs).to(device=device, **kwargs)

    return recurse(x, helper)


def from_torch(t, force_scalar=False):
    def helper(ts):
        if not isinstance(ts, torch.Tensor):
            return ts
        x = ts.detach().cpu().numpy()
        if force_scalar and (x.size == 1 or np.isscalar(x)):
            return np.asscalar(x)
        return x

    return recurse(t, helper)