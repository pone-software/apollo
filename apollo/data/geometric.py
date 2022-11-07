from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apollo.data.utils import JSONSerializable


@dataclass
class Vector(JSONSerializable):
    x: float
    y: float
    z: float

    @classmethod
    def from_json(cls, json: dict) -> Vector:
        return cls(
            x=json['x'],
            y=json['y'],
            z=json['z']
        )

    def as_json(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

    @classmethod
    def from_ndarray(cls, ndarray: np.ndarray) -> Vector:
        return cls(
            x=ndarray[0],
            y=ndarray[1],
            z=ndarray[2]
        )

    def __repr__(self):
        return f"Point (x: {self.x}, y: {self.y}, z: {self.z})"

    def __array__(self, dtype=None):
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def __aray_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """

        Args:
            ufunc:
            method:
            *inputs:
            **kwargs:

        Returns:

        Todo:
            * Decide whether to implement or not

        """
        pass
