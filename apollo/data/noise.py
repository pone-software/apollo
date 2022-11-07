import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Optional, Tuple, List, TypeVar, Type

import numpy as np
import awkward as ak

from apollo.data.configs import HistogramConfig, Interval
from apollo.data.detectors import Detector
from apollo.utils.random import get_rng

NoiseGeneratorType = TypeVar('NoiseGeneratorType', bound='NoiseGenerator')


class NoiseGenerator(ABC):
    def __init__(self, detector: Detector, distribution: str,
                 rng: Optional[np.random.Generator] = None,
                 **kwargs):
        self.detector = detector
        if rng is None:
            rng = get_rng()
        self.rng = rng
        if not hasattr(rng, distribution):
            raise ValueError(f'Generator does not have distribution {distribution} implemented')
        self.distribution = getattr(self.rng, distribution)
        self.kwargs = kwargs

    def get_distribution_parameters(self, config: HistogramConfig) -> dict:
        return {}

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        raise NotImplementedError('number_of_parameters not implemented')

    def __get_size(self, config: HistogramConfig) -> Tuple[int, int, int]:
        number_of_modules = self.detector.number_of_modules
        number_of_bins = config.number_of_bins
        return number_of_modules, number_of_bins, self.number_of_parameters

    def generate(self, config: Union[Interval, HistogramConfig]) -> np.ndarray:
        """

        Args:
            config:

        Returns:

        Todo:
            * Implement possibility to create per module rates

        """
        if type(config) is Interval:
            config = HistogramConfig(start=config.start, end=config.end, bin_size=config.length)
        size = self.__get_size(config=config)

        distribution_parameters = self.get_distribution_parameters(config=config)

        raw_histogram = self.distribution(size=size, **distribution_parameters, **self.kwargs)

        return np.sum(raw_histogram, axis=2)

    def generate_hits(self, config: Union[Interval, HistogramConfig]) -> ak.Array:
        histogram = self.generate(config=config)
        hits_per_module = np.sum(histogram, axis=1)
        hits = []

        for hits_for_module in hits_per_module:
            hits.append(ak.Array(self.rng.uniform(low=config.start, high=config.end, size=hits_for_module)))

        return hits

    @staticmethod
    def scale_per_bin_size(config: HistogramConfig, parameter: Union[float, Tuple[float, ...]]):
        new_rate = np.multiply(parameter, config.bin_size)
        if isinstance(parameter, tuple):
            return tuple(new_rate)

        return new_rate


class PoissonNoiseGenerator(NoiseGenerator):
    def __init__(self, detector: Detector, lam: Union[float, Tuple[float, ...]], **kwargs):
        super().__init__(detector, distribution='poisson', **kwargs)
        self.lam = lam

    @property
    def number_of_parameters(self) -> int:
        if isinstance(self.lam, tuple):
            return len(self.lam)
        return 1

    def get_distribution_parameters(self, config: HistogramConfig) -> dict:
        lam = self.scale_per_bin_size(config, self.lam)

        return {
            'lam': lam
        }


class NormalNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 detector: Detector,
                 location: Union[float, Tuple[float, ...]],
                 scale: Union[float, Tuple[float, ...]],
                 **kwargs):
        super().__init__(detector, distribution='normal', **kwargs)
        self.location = location
        self.scale = scale

    @property
    def number_of_parameters(self) -> int:
        location_number = 1
        if isinstance(self.location, tuple):
            location_number = len(self.location)
        scale_number = 1
        if isinstance(self.scale, tuple):
            scale_number = len(self.scale)
        return np.max([location_number, scale_number])

    def get_distribution_parameters(self, config: HistogramConfig) -> dict:
        location = self.scale_per_bin_size(config, self.location)
        scale = self.scale_per_bin_size(config, self.scale)
        return {
            'loc': location,
            'scale': scale
        }


class NoiseGeneratorEnum(Enum):
    POISSON = PoissonNoiseGenerator
    NORMAL = NormalNoiseGenerator


class NoiseGeneratorFactory:
    def __init__(self, detector: Detector, rng: np.random.Generator = None):
        self.detector = detector
        self.rng = rng

    def create(self, noise_generator_type: NoiseGeneratorEnum, **kwargs) -> Type[NoiseGeneratorType]:
        return noise_generator_type.value(detector=self.detector, rng=self.rng, **kwargs)


class NoiseGeneratorCollection:
    def __init__(self, noise_generators: Union[Type[NoiseGeneratorType], List[Type[NoiseGeneratorType]]]):
        self.noise_generators = noise_generators

    def generate(self, config: Union[Interval, HistogramConfig]) -> np.ndarray:
        if isinstance(self.noise_generators, NoiseGenerator):
            return self.noise_generators.generate(config=config)

        if len(self.noise_generators) == 0:
            raise ValueError('Cannot generate histograms with no generator in place')
        final_histogram = None

        for noise_generator in self.noise_generators:
            generated_histogram = noise_generator.generate(config)
            if final_histogram is None:
                final_histogram = generated_histogram
            else:
                final_histogram += generated_histogram

        return final_histogram
