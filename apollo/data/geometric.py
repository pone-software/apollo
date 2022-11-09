from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apollo.data.utils import JSONSerializable


@dataclass
class Vector(JSONSerializable):
    """
    Three-dimensional vector.
    """
    x: float
    y: float
    z: float

    @classmethod
    def from_json(cls, dictionary: dict) -> Vector:
        """
        Reads Histogram Config from jsonable dictionary.
        
        Args:
            dictionary: json dictionary to read in

        Returns:
            Config read from input dictionary

        """
        return cls(
            x=dictionary['x'],
            y=dictionary['y'],
            z=dictionary['z']
        )

    def as_json(self) -> dict:
        """
        Transforms vector to valid json dictionary.

        Returns:
            JSON representation of vector

        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

    @classmethod
    def from_ndarray(cls, ndarray: np.ndarray) -> Vector:
        """
        Reads Vector from numpy array

        Args:
            ndarray: numpy array to read in

        Returns:
            Vector read in from numpy array

        """
        return cls(
            x=ndarray[0],
            y=ndarray[1],
            z=ndarray[2]
        )

    def __repr__(self):
        """
        String representation of the vector

        Returns:
            String representation of the vector

        """
        return f"Point (x: {self.x}, y: {self.y}, z: {self.z})"

    def __array__(self, dtype=None):
        """
        Allow numpy to import vector directly

        Args:
            dtype: Numpy dtype of the vector

        Returns:
            Numpy array representation of the vector

        """
        return np.array([self.x, self.y, self.z], dtype=dtype)
