from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from apollo.data.detectors import Detector
from apollo.data.utils import JSONSerializable


@dataclass
class Interval(JSONSerializable):
    start: Optional[float] = 0
    end: Optional[float] = 1000

    @property
    def range(self) -> Tuple[float, float]:
        return self.start, self.end

    @property
    def length(self):
        """
        Represents length of the interval

        Returns:
            length of the interval

        """
        return self.end - self.start

    def is_between(self, value: float) -> bool:
        """
        Tells you whether your value is between or outside
        Args:
            value:

        Returns:

        """
        left = self.start is not None and value >= self.start
        right = self.end is not None and value < self.end
        return left and right

    @classmethod
    def from_json(cls, json: dict) -> Interval:
        return cls(
            start=json['start'],
            end=json['end']
        )

    def as_json(self) -> dict:
        """
        Creates a json compatible version of interval config

        Returns:
            json compatible dict of interval config

        """
        return {
            'start': self.start,
            'end': self.end
        }

    def __repr__(self):
        return f"Interval: [{self.start}, {self.end})"

    def __array__(self, dtype=None):
        return np.array([self.start, self.end], dtype=dtype)


@dataclass
class HistogramConfig(Interval):
    bin_size: int = 10

    @classmethod
    def from_json(cls, json: dict) -> HistogramConfig:
        """
        creates histogram config from json like dict

        Args:
            json: json like version of histogram config

        Returns:
            histogram config object based on json

        """
        return HistogramConfig(start=json['start'],
                               end=json['end'],
                               bin_size=json['bin_size'])

    @property
    def number_of_bins(self) -> int:
        return int(np.ceil(self.length / self.bin_size))

    def as_json(self) -> dict:
        """
        Generates a json compatible histogram config

        Returns:
            json compatible histogram config

        """
        return_json = super().as_json()
        return_json['bin_size'] = self.bin_size
        return return_json

    def __repr__(self):
        return str(super().__init__()) + f"; Bin Size: {self.bin_size}"

    def __array__(self, dtype=None):
        return np.array([self.start, self.end, self.bin_size], dtype=dtype)


@dataclass
class HistogramDatasetConfig(JSONSerializable):
    path: str
    detector: Detector
    histogram_config: HistogramConfig

    @classmethod
    def from_json(cls, dictionary: dict):
        return HistogramDatasetConfig(path=dictionary['path'],
                                      detector=Detector.from_json(dictionary['detector']),
                                      histogram_config=HistogramConfig.from_json(dictionary['histogram_config']))

    def as_json(self) -> dict:
        return {
            'path': self.path,
            'detector': self.detector.as_json(),
            'histogram_config': self.histogram_config.as_json()
        }
