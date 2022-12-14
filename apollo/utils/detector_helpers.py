import numpy as np

from olympus.event_generation.detector import (
    Detector as ODetector,
    make_line,
    make_triang,
)

from apollo.data.detectors import Detector
from apollo.data.importers import DetectorImporter


def get_line_detector() -> Detector:
    """Helper to generate line detector with standard params from olympus.

    Returns:
        Detector having all modules in line

    """
    rng = np.random.RandomState(1337)
    dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

    pmts_per_module = 16
    pmt_cath_area_r = 75e-3 / 2  # m
    module_radius = 0.21  # m

    # Calculate the relative area covered by PMTs
    efficiency = (
        pmts_per_module
        * pmt_cath_area_r**2
        * np.pi
        / (4 * np.pi * module_radius**2)
    )
    det = ODetector(
        make_line(0, 0, 20, 50, rng, dark_noise_rate, 0, efficiency=efficiency)
    )

    return DetectorImporter.from_olympus(det)


def get_triangle_detector() -> Detector:
    """Helper to generate triangular detector with standard params from olympus.

    Returns:
        Detector having three lines of modules

    """
    oms_per_line = 20
    dist_z = 50  # m
    dark_noise_rate = 16 * 1e-5  # 1/ns
    side_len = 100  # m
    pmts_per_module = 16
    pmt_cath_area_r = 75e-3 / 2  # m
    module_radius = 0.21  # m
    rng = np.random.RandomState(31338)

    # Calculate the relative area covered by PMTs
    efficiency = (
        pmts_per_module
        * (pmt_cath_area_r) ** 2
        * np.pi
        / (4 * np.pi * module_radius**2)
    )
    det = ODetector(
        make_triang(
            side_len, oms_per_line, dist_z, dark_noise_rate, rng, efficiency=efficiency
        )
    )

    return DetectorImporter.from_olympus(det)
