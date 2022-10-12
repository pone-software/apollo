import numpy as np

DEFAULT_SEED = 1337

def get_rng(seed = DEFAULT_SEED) -> np.random.PCG64:
    return np.random.default_rng(seed)