from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from apollo.data.detectors import Detector
from apollo.data.utils import JSONSerializable


@dataclass
class Interval(JSONSerializable):
    """
    Class defining a basic interval
    """

    start: Optional[float] = 0
    end: Optional[float] = 1000

    @property
    def range(self) -> Tuple[float, float]:
        """
        Tuple containing the interval range

        Returns:
            Tuple containing interval range

        """
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
        Tells you whether your value is between or outside.

        Args:
            value: Value to check

        Returns: Boolean containing whether value is between start and end.

        """
        left = self.start is not None and value >= self.start
        right = self.end is not None and value < self.end
        return left and right

    @classmethod
    def from_json(cls, dictionary: dict) -> Interval:
        """
        reads from JSON dict

        Args:
            dictionary: json dict to read in

        Returns:
            Interval read in from dict

        """
        return cls(start=dictionary["start"], end=dictionary["end"])

    def as_json(self) -> dict:
        """
        Creates a json compatible version of interval config

        Returns:
            json compatible dict of interval config

        """
        return {"start": self.start, "end": self.end}

    def __repr__(self):
        """
        String representation of the interval

        Returns:
            String representation of the interval

        """
        return f"Interval: [{self.start}, {self.end})"

    def __array__(self, dtype=None):
        """
        Allow numpy to import interval directly

        Args:
            dtype: Numpy dtype of the array

        Returns:
            Numpy array representation of the interval

        """
        return np.array([self.start, self.end], dtype=dtype)


@dataclass
class HistogramConfig(Interval):
    """
    Subclass of Interval adding tht bin size to configure a histogram.
    """

    bin_size: int = 10

    @classmethod
    def from_json(cls, dictionary: dict) -> HistogramConfig:
        """
        creates histogram config from json like dict

        Args:
            dictionary: json like version of histogram config

        Returns:
            histogram config object based on json

        """
        return HistogramConfig(
            start=dictionary["start"],
            end=dictionary["end"],
            bin_size=dictionary["bin_size"],
        )

    @property
    def number_of_bins(self) -> int:
        """
        Calculate how many bins are between start and end

        Returns:
            number of bins

        """
        return int(np.ceil(self.length / self.bin_size))

    def as_json(self) -> dict:
        """
        Generates a json compatible histogram config

        Returns:
            json compatible histogram config

        """
        return_json = super().as_json()
        return_json["bin_size"] = self.bin_size
        return return_json

    def __repr__(self):
        """
        String representation of the histogram config

        Returns:
            string representation

        """
        return str(super().__init__()) + f"; Bin Size: {self.bin_size}"

    def __array__(self, dtype=None):
        """
        Enable numpy type coercion.

        Args:
            dtype: Numpy dtype of final interval

        Returns:
            numpy array of Histogram Config

        """
        return np.array([self.start, self.end, self.bin_size], dtype=dtype)


@dataclass
class HistogramDatasetConfig(JSONSerializable):
    """
    Configuration for creating and reading histogram dataset.
    """

    path: str
    detector: Detector
    histogram_config: HistogramConfig

    @classmethod
    def from_json(cls, dictionary: dict):
        """
        Reads Histogram Config from jsonable dictionary.

        Args:
            dictionary: json dictionary to read in

        Returns:
            Config read from input dictionary

        """
        return HistogramDatasetConfig(
            path=dictionary["path"],
            detector=Detector.from_json(dictionary["detector"]),
            histogram_config=HistogramConfig.from_json(dictionary["histogram_config"]),
        )

    def as_json(self) -> dict:
        """
        Transforms config to valid json dictionary.

        Returns:
            JSON representation of config

        """
        return {
            "path": self.path,
            "detector": self.detector.as_json(),
            "histogram_config": self.histogram_config.as_json(),
        }
