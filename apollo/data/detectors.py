from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from apollo.data.utils import JSONSerializable


@dataclass
class Module(JSONSerializable):
    """
    Detection module.

    Attributes:
        position: np.ndarray
            Module position (x, y, z)
        noise_rate: float
            Noise rate in 1/ns
        efficiency: float
            Module efficiency (0, 1]
        self.key: collection
            Module identifier
    """
    position: np.ndarray
    key: Tuple
    noise_rate: float = 1
    efficiency: float = 0.2

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {str(self.position)} [m], {self.noise_rate} [Hz], {self.efficiency}"
        )

    def as_json(self) -> dict:
        return {
            'position': list(self.position),
            'key': self.key,
            'noise_rate': self.noise_rate,
            'efficiency': self.efficiency
        }

    @classmethod
    def from_json(cls, dictionary: dict):
        return cls(
            position=np.array(dictionary['position']),
            key=dictionary['key'],
            noise_rate=dictionary['noise_rate'],
            efficiency=dictionary['efficiency']
        )

@dataclass
class Detector(JSONSerializable):
    modules: List[Module]

    @property
    def module_coordinates(self) -> np.ndarray:
        return np.vstack([m.position for m in self.modules])

    @property
    def module_efficiencies(self) -> np.array:
        return np.asarray([m.efficiency for m in self.modules])

    @property
    def module_noise_rates(self) -> np.ndarray:
        return np.asarray([m.noise_rate for m in self.modules])

    @property
    def number_of_modules(self) -> int:
        return len(self.modules)

    def as_json(self) -> dict:
        return {
            'modules': [module.as_json() for module in self.modules]
        }

    @classmethod
    def from_json(cls, dictionary: dict):
        return cls(modules=[Module.from_json(module) for module in dictionary['modules']])