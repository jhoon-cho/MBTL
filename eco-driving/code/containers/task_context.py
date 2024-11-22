"""
TaskContext is an abstraction for changes to the environment (= route/net files) that can happen during a training.

Can either be a set of files =PathTaskContext,
or just a set of characteristics =NetGenTaskContext (the files are then generated on the fly).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Union

from sumo.utils import get_directions

"""
Specifies whether approaches have traffic at the same time or not:
Can be
- False, i.e. all approaches are enabled at the same time
- a str with a single approach name which will be enabled alone
- True, i.e. a single approach which will be enabled, without specifying one
"""
MultiApproachSelector = Union[str, bool]


class ContinuousSelector(NamedTuple):
    """
    Specifies a continuous range of values a parameter can take its values from. Also specifies the number of samples to use
    when evaluating.

    The values are sampled following uniform distribution.
    """
    start: float
    end: float
    num_samples: int = 5

    def sample(self) -> float:
        return random.random() * (self.end - self.start) + self.start

    def list(self) -> List[float]:
        return [i * (self.end - self.start) / (self.num_samples - 1) + self.start for i in range(self.num_samples)]


"""

"""
Selector = Union[float, ContinuousSelector, list]


@dataclass
class TaskContextABC(ABC):
    """
    Example of what the TaskContext Superclass would be. Unfortunately NamedTuple doesn't support multiple inheritance.
    """
    penetration_rate: float

    @abstractmethod
    def list_tasks(self, add_0_penrate: bool = False) -> List["TaskContext"]:
        pass

    @abstractmethod
    def sample_task(self) -> "TaskContext":
        pass

    @abstractmethod
    def compact_str(self) -> str:
        pass

    @abstractmethod
    def base_net_file(self) -> Path:
        pass


class PathTaskContext(NamedTuple):
    """
    A PathTaskContext is a path to a dir containing a net.net.xml and a routes.rou.xml files.
    """
    dir: Path
    """ Where to find the dataset or the net.net.xml file"""
    single_approach: MultiApproachSelector
    """ Which approach to use:
    a str like "A", "B"...,
    True to use them separately, or
    False to use all of them at the same time """
    penetration_rate: Selector
    """ The penetration rate, between 0 and 1 (both included) """
    aadt_conversion_factor: Selector = 0.084
    """ Converts AADT into hourly traffic, default is 8.4% (average for peak hour traffic) """

    def list_tasks(self, add_0_penrate: bool = False) -> List["TaskContext"]:
        return [PathTaskContext(penetration_rate=penrate,
                                dir=path, single_approach=approach,
                                aadt_conversion_factor=aadt_factor)
                for path in ([self.dir] if (self.dir / 'net.net.xml').exists() else self.dir.glob('*'))
                if path.is_dir()
                for approach in (get_directions(path / 'net.net.xml') if self.single_approach is True else [self.single_approach])
                for penrate in _singleton(self.penetration_rate, False) + ([0] if add_0_penrate else [])
                for aadt_factor in _singleton(self.aadt_conversion_factor, False)]

    def sample_task(self) -> "TaskContext":
        task = random.sample(self.list_tasks(), 1)[0]._asdict()
        task['penetration_rate'] = _sample(self.penetration_rate)
        if isinstance(self.aadt_conversion_factor, ContinuousSelector):
            task['aadt_conversion_factor'] = _sample(self.aadt_conversion_factor)
        return PathTaskContext(**task)

    @abstractmethod
    def compact_str(self) -> str:
        return f'{self.dir.name}_{self.single_approach}_{self.penetration_rate}_{self.aadt_conversion_factor}'

    @abstractmethod
    def base_net_file(self) -> Path:
        file = self.dir / 'net.net.xml'
        if file.exists():
            return file
        else:
            raise FileNotFoundError('The PathTaskContext represents many files, choose one before calling this')


class NetGenTaskContext(NamedTuple):
    """
    A TaskContext that is extracted from the config. If the config fields ('base_id', 'penetration_rate', 'inflow',
    'green_phase', 'red_pase', 'lane_length', 'speed_limit') are lists, each element of the list will give one
    TaskContext (thus if there are too many elements the number of combinations might be very high).
    """

    base_id: Union[int, List[int]]
    """ number of lanes and phases stuck together first digit is lane number, second is hase number.
        Only 11, 21, 31, 41, 22, 32, 42 supported """
    inflow: Selector
    """ inflow, in vehicle per hour. Used as is (no more factor for single lane scenario """
    green_phase: Selector
    """ Duration of the main green phase for approaches A and C """
    red_phase: Selector
    """ Duration of the main red phase (not including amber) for approaches A and C """
    lane_length: Selector
    """ Lane length, meter """
    speed_limit: Selector
    """ Speed limit, m/s """
    offset: Selector
    """ Offset between ghost cells TL programs and the main intersection TL program.
        Proportion of the total program duration, so between 0 and 1."""
    single_approach: MultiApproachSelector
    """ Which approach to use:
        a str like "A", "B"...,
        True to use the default, or
        False to use all of them at the same time """
    penetration_rate: Selector
    """ The penetration rate, between 0 and 1 (both included) """

    def list_tasks(self, add_0_penrate: bool = False) -> List["TaskContext"]:
        return [NetGenTaskContext(
            base_id=base_id,
            penetration_rate=penetration_rate,
            single_approach=self.single_approach,
            inflow=inflow,
            green_phase=green_phase,
            red_phase=red_phase,
            lane_length=lane_length,
            speed_limit=speed_limit,
            offset=offset
        )
            for base_id in _singleton(self.base_id, False)
            for penetration_rate in _singleton(
                [*_singleton(self.penetration_rate, False), 0]
                if add_0_penrate else self.penetration_rate, False)
            for inflow in _singleton(self.inflow, False)
            for green_phase in _singleton(self.green_phase, False)
            for red_phase in _singleton(self.red_phase, False)
            for lane_length in _singleton(self.lane_length, False)
            for speed_limit in _singleton(self.speed_limit, False)
            for offset in _singleton(self.offset, False)
        ]

    def sample_task(self) -> "TaskContext":
        return NetGenTaskContext(**{k: _sample(v) for k, v in self._asdict().items()})

    @abstractmethod
    def compact_str(self) -> str:
        return '_'.join(str(x) for x in self)

    @abstractmethod
    def base_net_file(self) -> Path:
        return Path('resources/sumo_static') / f'{self.base_id}/net.net.xml'

    @property
    def num_lanes(self) -> int:
        return self.base_id // 10

    @property
    def num_phases(self) -> int:
        return self.base_id % 10


TaskContext = Union[PathTaskContext, NetGenTaskContext]


def _sample(x: Selector):
    if isinstance(x, list):
        return random.sample(x, 1)[0]
    elif isinstance(x, ContinuousSelector):
        return x.sample()
    else:
        return x


def _singleton(x: any, keep_selectors: bool) -> list:
    if isinstance(x, list):
        return x
    elif isinstance(x, ContinuousSelector):
        if keep_selectors:
            return [x]
        else:
            return [i * (x.end - x.start) / (x.num_samples - 1) + x.start for i in range(x.num_samples)]
    else:
        return [x]