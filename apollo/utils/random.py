"""This module contains all the utility functions for randomness."""

from typing import Sequence, Union

import numpy as np

from numpy.random import BitGenerator, Generator, SeedSequence


DEFAULT_SEED = 1337


def get_rng(
    seed: Union[
        None, int, Sequence[int], SeedSequence, BitGenerator, Generator
    ] = DEFAULT_SEED
) -> np.random.Generator:
    """Function to be able to retrieve Random number generators with a given Seed.

    Args:
        seed: Seed to set the state of the generated Generator

    Returns:
        Newly generated Numpy random Generator

    Todo:
        * Check import and export of Random Number Generators
    """
    return np.random.default_rng(seed=seed)
