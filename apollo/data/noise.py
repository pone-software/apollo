from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import awkward as ak
import numpy as np

from apollo.data.configs import HistogramConfig, Interval
from apollo.data.detectors import Detector
from apollo.utils.random import get_rng


class NoiseGenerator(ABC):
    """Abstract class as parent for all noise generators."""

    def __init__(
        self,
        detector: Detector,
        distribution: str,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ):
        """Constructor for the NoiseGenerator.

        Args:
            detector: Detector to generate noise for
            distribution: String of the distribution to use
            rng: Generator for random numbers
            **kwargs: arguments to be passed
        """
        self.detector = detector
        if rng is None:
            rng = get_rng()
        self.rng = rng
        if not hasattr(rng, distribution):
            raise ValueError(
                f"Generator does not have distribution {distribution} implemented"
            )
        self.distribution = getattr(self.rng, distribution)
        self.kwargs = kwargs

    def get_distribution_parameters(
        self, histogram_config: HistogramConfig
    ) -> Dict[str, Any]:
        """method to construct the parameters to be passed to the probability
        distribution.

        Args:
            histogram_config: to scale parameters according to the bin size

        Returns:
            dictionary with all the kwargs for the distribution

        """
        return {}

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        """Calculates the number of parameters.

        Example:
            The poisson distribution is called with
            ``self.rng.poisson(size=5, lambda=(3,4))``. Then the number of
            parameters should be three.


        """
        raise NotImplementedError("number_of_parameters not implemented")

    def __get_size(self, histogram_config: HistogramConfig) -> Tuple[int, int, int]:
        """Returns the total size of the generated array.

        Args:
            histogram_config: histogram config to determine shape.

        Returns:
            tuple containing the final size of the distribution output

        """
        number_of_modules = self.detector.number_of_modules
        number_of_bins = histogram_config.number_of_bins
        return number_of_modules, number_of_bins, self.number_of_parameters

    def generate(
        self, config: Union[Interval, HistogramConfig]
    ) -> Union[Any, np.typing.NDArray[np.float64]]:
        """Generates Noise given for a certain interval or histogram config.

        Args:
            config: interval and histogram to generate noise for

        Returns:
            Array containing the noise for a given histogram

        Todo:
            * Implement possibility to create per module rates

        """
        if type(config) is not HistogramConfig:
            config = HistogramConfig(
                start=config.start, end=config.end, bin_size=int(np.ceil(config.length))
            )
        size = self.__get_size(histogram_config=config)

        distribution_parameters = self.get_distribution_parameters(
            histogram_config=config
        )

        raw_histogram = self.distribution(
            size=size, **distribution_parameters, **self.kwargs
        )

        return np.sum(raw_histogram, axis=2)

    def generate_hits(self, config: Union[Interval, HistogramConfig]) -> List[ak.Array]:
        """Generates an awkward array containing hits per module for a given
        interval or histogram.

        Args:
            config: interval or histogram to generate noise for

        Returns:
            Awkward array containing hits per module

        """
        histogram = self.generate(config=config)
        hits_per_module = np.sum(histogram, axis=1)
        hits = []

        for hits_for_module in hits_per_module:
            hits.append(
                ak.Array(
                    self.rng.uniform(
                        low=config.start, high=config.end, size=hits_for_module
                    )
                )
            )

        return hits

    @staticmethod
    def scale_per_bin_size(
        histogram_config: HistogramConfig, parameter: Union[float, Tuple[float, ...]]
    ) -> Union[float, Tuple[float, ...]]:
        """As each histogram bin could have a length larger than one the distribution
        parameters like ``lambda`` have to be scaled to achieve a valid output.

        Args:
            histogram_config: Histogram config by which to scale the parameter.
            parameter: Parameter value(s) to scale

        Returns:
            Scaled set of parameters

        """
        new_rate = np.multiply(parameter, histogram_config.bin_size)
        if isinstance(parameter, tuple):
            return tuple(new_rate)

        return float(new_rate)


class PoissonNoiseGenerator(NoiseGenerator):
    """Generates noise based on the poisson distribution."""

    def __init__(
        self, detector: Detector, lam: Union[float, Tuple[float, ...]], **kwargs: Any
    ):
        """Constructor for the poisson noise generator.

        Args:
            detector: Detector to generate noise for
            lam: lambda value(s) to generate the distribution for.
            **kwargs: arguments to be passed on
        """
        super().__init__(detector, distribution="poisson", **kwargs)
        self.lam = lam

    @property
    def number_of_parameters(self) -> int:
        """Calculates the number of poisson distribution parameters.

        Returns:
            length of passed lambda values

        """
        if isinstance(self.lam, tuple):
            return len(self.lam)
        return 1

    def get_distribution_parameters(
        self, histogram_config: HistogramConfig
    ) -> Dict[str, Union[float, Tuple[float, ...]]]:
        """Creates kwargs for the poisson distribution.

        Args:
            histogram_config: to scale parameters according to the bin size

        Returns:
            kwargs for poisson distribution

        """
        lam = self.scale_per_bin_size(histogram_config, self.lam)

        return {"lam": lam}


class NormalNoiseGenerator(NoiseGenerator):
    """Generates noise based on the normal distribution."""

    def __init__(
        self,
        detector: Detector,
        location: Union[float, Tuple[float, ...]],
        scale: Union[float, Tuple[float, ...]],
        **kwargs: Any,
    ):
        """Constructor for the poisson noise generator.

        Args:
            detector: Detector to generate noise for
            location: value(s) for the expectation value.
            scale: value(s) for the scale.
            **kwargs: arguments to be passed on
        """
        super().__init__(detector, distribution="normal", **kwargs)
        self.location = location
        self.scale = scale

    @property
    def number_of_parameters(self) -> int:
        """Calculates the number of normal distribution parameters.

        Returns:
            length of passed scale and location values

        """
        location_number = 1
        if isinstance(self.location, tuple):
            location_number = len(self.location)
        scale_number = 1
        if isinstance(self.scale, tuple):
            scale_number = len(self.scale)
        return int(np.max([location_number, scale_number]))

    def get_distribution_parameters(
        self, histogram_config: HistogramConfig
    ) -> Dict[str, Union[float, Tuple[float, ...]]]:
        """Creates kwargs for the normal distribution.

        Args:
            histogram_config: to scale parameters according to the bin size

        Returns:
            kwargs for normal distribution

        """
        location = self.scale_per_bin_size(histogram_config, self.location)
        scale = self.scale_per_bin_size(histogram_config, self.scale)
        return {"loc": location, "scale": scale}


class NoiseGeneratorEnum(Enum):
    """Enum of implemented noise generators."""

    POISSON = PoissonNoiseGenerator
    NORMAL = NormalNoiseGenerator


class NoiseGeneratorFactory:
    """Provides a standardized way of creating a noise generator."""

    def __init__(self, detector: Detector, rng: Optional[np.random.Generator] = None):
        """Constructor of Factory.

        Args:
            detector: Detector to generate noise for
            rng: Generator of random numbers
        """
        self.detector = detector
        self.rng = rng

    def create(
        self, noise_generator_type: NoiseGeneratorEnum, **kwargs: Any
    ) -> NoiseGenerator:
        """Construct a noise generator based on arguments.

        Args:
            noise_generator_type: Noise generator to construct.
            **kwargs: Arguments to be passed down

        Returns:

        """
        return noise_generator_type.value(  # type: ignore
            detector=self.detector, rng=self.rng, **kwargs
        )


class NoiseGeneratorCollection:
    """Build a collection of noise generators to collectively generate noise."""

    def __init__(
        self,
        noise_generators: Union[NoiseGenerator, List[NoiseGenerator]],
    ):
        """Constructor of the NoiseGeneratorCollection.

        Args:
            noise_generators: NoiseGenerator(s) to start with.
        """
        self.noise_generators = noise_generators

    def generate(
        self, histogram_config: Union[Interval, HistogramConfig]
    ) -> Union[Any, np.typing.NDArray[np.float64]]:
        """Generates noise for all the noise generators in the collection and combines
        it.

        Args:
            histogram_config: Histogram to generate events by

        Returns:
            Numpy array containing the combined noise.

        """
        if isinstance(self.noise_generators, NoiseGenerator):
            return self.noise_generators.generate(config=histogram_config)

        if len(self.noise_generators) == 0:
            raise ValueError("Cannot generate histograms with no generator in place")
        final_histogram = None

        for noise_generator in self.noise_generators:
            generated_histogram = noise_generator.generate(histogram_config)
            if final_histogram is None:
                final_histogram = generated_histogram
            else:
                final_histogram += generated_histogram

        return final_histogram
