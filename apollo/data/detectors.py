from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from apollo.data.geometric import Vector
from apollo.data.utils import JSONSerializable


@dataclass
class Module(JSONSerializable):
    """Detection module."""

    position: Vector
    key: Tuple[int, int]
    noise_rate: float = 1
    efficiency: float = 0.2

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            string representation

        """
        return repr(
            f"Module {self.key}, {str(self.position)} [m], {self.noise_rate} [Hz],"
            f" {self.efficiency}"
        )

    def as_json(self) -> Dict[str, Union[List[float], Tuple[int, int], float]]:
        """Generates JSON dictionary of Module.

        Returns:
            JSON representation

        """
        position = np.array(self.position)
        return {
            "position": list(position),
            "key": self.key,
            "noise_rate": self.noise_rate,
            "efficiency": self.efficiency,
        }

    @classmethod
    def from_json(cls, dictionary: Dict[str, Any]) -> Module:
        """Reads Module from dictionary.

        Args:
            dictionary: dictionary representation of module

        Returns:
            Module based on input dictionary

        """
        return cls(
            position=Vector.from_ndarray(dictionary["position"]),
            key=dictionary["key"],
            noise_rate=dictionary["noise_rate"],
            efficiency=dictionary["efficiency"],
        )


@dataclass
class Detector(JSONSerializable):
    """Data model for a P-ONE detector."""

    modules: List[Module]

    @property
    def module_coordinates(self) -> np.typing.NDArray[np.float64]:
        """Creates a stack of module positions in numpy.

        Returns:
            numpy array of module coordinates

        """
        return np.vstack([m.position for m in self.modules])

    @property
    def module_efficiencies(self) -> np.typing.NDArray[np.float64]:
        """Creates an array of module efficiencies.

        Returns:
            numpy array containing module efficiencies.

        """
        return np.asarray([m.efficiency for m in self.modules])

    @property
    def module_noise_rates(self) -> np.typing.NDArray[np.float64]:
        """Creates an array of noise efficiencies.

        Returns:
            numpy array containing noise efficiencies.

        """
        return np.asarray([m.noise_rate for m in self.modules])

    @property
    def number_of_modules(self) -> int:
        """Number of detector modules.

        Returns:
            length of modules

        """
        return len(self.modules)

    def as_json(self) -> Dict[str, List[Dict[str, Any]]]:
        """Transforms detector to valid json dictionary.

        Returns:
            JSON representation of detector

        """
        return {"modules": [module.as_json() for module in self.modules]}

    @classmethod
    def from_json(cls, dictionary: Dict[str, List[Dict[str, Any]]]) -> Detector:
        """
        Reads Detector from jsonable dictionary.

        Args:
            dictionary: json dictionary to read in

        Returns:
            Detector read from input dictionary

        """
        return cls(
            modules=[Module.from_json(module) for module in dictionary["modules"]]
        )
