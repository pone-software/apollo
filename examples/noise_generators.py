from apollo.data.configs import HistogramConfig
from apollo.data.noise import NoiseGeneratorEnum, NoiseGeneratorFactory
from apollo.utils.detector_helpers import get_line_detector


detector = get_line_detector()

lam = 4e5 / 1e9 * 20, 1.6e5 / 1e9 * 20, 100e5 / 1e9 * 20

factory = NoiseGeneratorFactory(detector=detector)
noise_gen1 = factory.create(NoiseGeneratorEnum.POISSON, lam=lam)

histogram_config = HistogramConfig()

histogram = noise_gen1.generate(histogram_config)

print("cool")
