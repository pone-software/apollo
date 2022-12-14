from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from apollo.data.utils import JSONSerializable


@dataclass
class Vector(JSONSerializable):
    """Three-dimensional vector."""

    x: float
    y: float
    z: float

    @classmethod
    def from_json(cls, dictionary: Dict[str, float]) -> Vector:
        """Reads Histogram Config from jsonable dictionary.

        Args:
            dictionary: json dictionary to read in

        Returns:
            Config read from input dictionary

        """
        return cls(x=dictionary["x"], y=dictionary["y"], z=dictionary["z"])

    def as_json(self) -> Dict[str, float]:
        """Transforms vector to valid json dictionary.

        Returns:
            JSON representation of vector

        """
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_ndarray(cls, ndarray: np.typing.NDArray[np.float64]) -> Vector:
        """Reads Vector from numpy array.

        Args:
            ndarray: numpy array to read in

        Returns:
            Vector read in from numpy array

        """
        return cls(x=ndarray[0], y=ndarray[1], z=ndarray[2])

    def __repr__(self) -> str:
        """String representation of the vector.

        Returns:
            String representation of the vector

        """
        return f"Point (x: {self.x}, y: {self.y}, z: {self.z})"

    def __array__(
        self, dtype: Optional[Union[np.int64, np.float64]] = None
    ) -> np.typing.NDArray[Union[np.int64, np.float64]]:
        """Allow numpy to import vector directly.

        Args:
            dtype: Numpy dtype of the vector

        Returns:
            Numpy array representation of the vector

        """
        return np.array(astuple(self), dtype=dtype)

    def __len__(self) -> int:
        """Determines the length of the point. In this case 3.

        Returns:
            Array lenght of the point

        """
        return astuple(self).__len__()

    def __getitem__(self, item: Any) -> float:
        """Get a specific set of dataclass tuple.

        Args:
            item: Item number or slice

        Returns:
            item of point

        """
        return astuple(self).__getitem__(item)  # type: ignore
