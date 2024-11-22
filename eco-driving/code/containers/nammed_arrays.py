import itertools

import numpy as np
import torch

from misc_utils import flatten


class NamedArrays(dict):
    """
    Data structure for keeping track of a dictionary of arrays (used for rollout information)
    e.g. {'reward': [...], 'action': [...]}
    """

    def __init__(self, dict_of_arrays={}, **kwargs):
        kwargs.update(dict_of_arrays)
        super().__init__(kwargs)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __getitem__(self, k):
        if isinstance(k, (slice, int, np.ndarray, list)):
            return type(self)((k_, arr[k]) for k_, arr in self.items())
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, (slice, int, np.ndarray, list)):
            if isinstance(v, dict):
                for vk, vv in v.items():
                    self[vk][k] = vv
            else:
                for k_, arr in self.items():
                    arr[k] = v
        else:
            super().__setitem__(k, v)

    def append(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).append(**v)
            else:
                self.setdefault(k, []).append(v)

    def extend(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).extend(**v)
            else:
                self.setdefault(k, []).extend(v)

    def to_array(self, inplace=True, dtype=None, concat=False):
        return self.apply(np.concatenate if concat else lambda x: np.asarray(x, dtype=dtype), inplace)

    def to_torch(self, dtype=None, device=None):
        for k, v in self.items():
            if isinstance(v, list):
                v = np.asarray(v, dtype=dtype)
            if isinstance(v, NamedArrays):
                v.to_torch(dtype, device)
            else:
                if v.dtype == object:
                    self[k] = [torch.tensor(np.ascontiguousarray(x), device=device) for x in v]
                else:
                    self[k] = torch.tensor(np.ascontiguousarray(v), device=device)
        return self

    def trim(self):
        min_len = len(self)
        for k, v in self.items():
            self[k] = v[:min_len]

    def __len__(self):
        return len(self.keys()) and min(len(v) for v in self.values())

    def filter(self, *args):
        return type(self)((k, v) for k, v in self.items() if k in args)

    def iter_minibatch(self, n_minibatches=None, concat=False, device='cpu'):
        if n_minibatches in [1, None]:
            yield slice(None), self.to_array(inplace=False, concat=concat).to_torch(device=device)
        else:
            for idxs in np.array_split(np.random.permutation(len(self)), n_minibatches):
                mini_batch = {}
                # TODO use recursive formulation
                for k, v in self.items():
                    if isinstance(v, list):
                        # TODO fix that, probably not good to have to precise dtype=object
                        mini_batch[k] = (np.array(v, dtype=object)[idxs.tolist()]).tolist()
                    elif isinstance(v, NamedArrays):
                        mini_batch[k] = NamedArrays({
                            k1: (np.array(v1, dtype=object)[idxs.tolist()]).tolist()
                            for k1, v1 in v.items()
                        })
                na = NamedArrays(mini_batch)
                yield idxs, na.to_array(inplace=False, concat=concat).to_torch(device=device)

    def apply(self, fn, inplace=True):
        if inplace:
            for k, v in self.items():
                if isinstance(v, NamedArrays):
                    v.apply(fn)
                else:
                    self[k] = fn(v)
            return self
        else:
            return type(self)((k, v.apply(fn, inplace=False) if isinstance(v, NamedArrays) else fn(v)) for k, v in self.items())

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, d):
        self.update(d)

    @classmethod
    def concat(cls, named_arrays, fn=None):
        # TODO raises numpy deprecation warning, fix it
        named_arrays = list(named_arrays)
        if not len(named_arrays):
            return cls()

        def concat(xs):
            """
            Common error with np.concatenate: conside arrays a and b, both of which are lists of arrays.
            If a contains irregularly shaped arrays and b contains arrays with the same shape, the numpy
            will treat b as a 2D array, and the concatenation will fail.
            Solution: use flatten instead of np.concatenate for lists of arrays
            """
            try:
                return np.concatenate(xs)
            except:
                return flatten(xs)

        get_concat = lambda v: v.concat if isinstance(v, NamedArrays) else fn or (torch.cat if isinstance(v, torch.Tensor) else concat)
        return cls((k, get_concat(v)([x[k] for x in named_arrays])) for k, v in named_arrays[0].items())