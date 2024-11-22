import numpy as np
import torch
from gym.spaces import Box, Discrete, Dict as GymDict

from containers.nammed_arrays import NamedArrays


class Dist:
    """ Distribution interface """

    def __init__(self, inputs):
        self.inputs = inputs

    def sample(self, shape=torch.Size([])):
        return self.dist.sample(shape)

    def argmax(self):
        raise NotImplementedError

    def logp(self, actions):
        return self.dist.log_prob(actions)

    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    def entropy(self):
        return self.dist.entropy()

    def __getitem__(self, idx):
        return type(self)(self.inputs[idx])


class CatDist(Dist):
    """ Categorical distribution (for discrete action spaces) """

    def __init__(self, inputs):
        super().__init__(inputs)
        self.dist = torch.distributions.categorical.Categorical(logits=inputs)

    def argmax(self):
        return self.dist.probs.argmax(dim=-1)


class DiagGaussianDist(Dist):
    """ Diagonal Gaussian distribution (for continuous action spaces) """

    def __init__(self, inputs):
        super().__init__(inputs)
        self.mean, self.log_std = torch.chunk(inputs, 2, dim=-1)
        self.std = self.log_std.exp()
        self.dist = torch.distributions.normal.Normal(self.mean, self.std)

    def argmax(self):
        return self.dist.mean

    def logp(self, actions):
        return super().logp(actions).sum(dim=-1)

    def kl(self, other):
        return super().kl(other).squeeze(dim=-1)

    def entropy(self):
        return super().entropy().squeeze(dim=-1)


def build_dist(space):
    """
    Build a nested distribution
    """
    if isinstance(space, Box):
        class DiagGaussianDist_(DiagGaussianDist):
            model_output_size = np.prod(space.shape) * 2

        return DiagGaussianDist_
    elif isinstance(space, Discrete):
        class CatDist_(CatDist):
            model_output_size = space.n
        return CatDist_

    elif isinstance(space, GymDict):
        names, subspaces = zip(*space.items())
        to_list = lambda x: [x[name] for name in names]
        from_list = lambda x: NamedArrays(zip(names, x))
        subdist_classes = [build_dist(subspace) for subspace in subspaces]
        subsizes = [s.model_output_size for s in subdist_classes]

        class DictDist(Dist):
            model_output_size = sum(subsizes)

            def __init__(self, inputs):
                super().__init__(inputs)
                self.dists = from_list(cl(x) for cl, x in zip(subdist_classes, inputs.split(subsizes, dim=-1)))

            def sample(self, shape=torch.Size([])):
                return from_list([dist.sample(shape) for dist in to_list(self.dists)])

            def argmax(self):
                return from_list([dist.argmax() for dist in to_list(self.dists)])

            def logp(self, actions):
                return sum(dist.logp(a) for a, dist in zip(to_list(actions), to_list(self.dists)))

            def kl(self, other):
                return sum(s.kl(o) for s, o in zip(to_list(self.dists), to_list(other.dists)))

            def entropy(self):
                return sum(dist.entropy() for dist in to_list(self.dists))

        return DictDist
    else:
        # Does not support lists at the moment
        # since there's no list equivalent of NamedArrays that allows advanced indexing
        raise NotImplementedError('No distribution for the given action space.')