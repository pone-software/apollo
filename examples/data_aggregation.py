import os
import numpy as np
import logging
from apollo.data.events import EventCollection
from olympus.event_generation.detector import Detector, make_line
from olympus.event_generation.generators import GeneratorFactory, GeneratorCollection

logging.getLogger().setLevel(logging.INFO)

data_base_path = os.path.join('../../data/')

def load_events_by_type(type: str):
    return EventCollection.from_folder(os.path.join(data_base_path, type))

# cascades = EventCollection.from_pickle(os.path.join(data_base_path, 'cascades/events_cascade_0.pickle'))
# tracks = EventCollection.from_pickle(os.path.join(data_base_path, 'tracks/events_track_0.pickle'))
# starting_tracks = EventCollection.from_pickle(os.path.join(data_base_path, 'starting_tracks/events_starting_track_0.pickle'))


cascades = load_events_by_type('cascades')
tracks = load_events_by_type('tracks')
starting_tracks = load_events_by_type('starting_tracks')

rng = np.random.RandomState(1337)
oms_per_line = 20
dist_z = 50  # m
dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

start_time = 0
end_time = 10000000
step_size = 50

# Calculate the relative area covered by PMTs
efficiency = (
        pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (4 * np.pi * module_radius ** 2)
)
det = Detector(make_line(0, 0, 20, 50, rng, dark_noise_rate, 0, efficiency=efficiency))
# det = make_triang(100, 20, dist_z, dark_noise_rate, rng, efficiency)

generator_factory = GeneratorFactory(det)

cascades.detector = det

cascades_generator = generator_factory.create('event_collection', event_collection=cascades, rate=0.001)
starting_tracks_generator = generator_factory.create('event_collection', event_collection=starting_tracks, rate=0.001)
tracks_generator = generator_factory.create('event_collection', event_collection=tracks, rate=0.001)
noise_generator = generator_factory.create("noise")

generator_collection = GeneratorCollection()
generator_collection.add_generator(cascades_generator)
generator_collection.add_generator(starting_tracks_generator)
generator_collection.add_generator(tracks_generator)
generator_collection.add_generator(noise_generator)

final_event_collection = generator_collection.generate_per_timeframe(start_time=start_time, end_time=end_time)
final_event_collection.detector = det

final_event_collection.to_folder(os.path.join(data_base_path, 'training/single_line_all_events'))

# histogram = final_event_collection.generate_histogram(bin_size=bin_size, end=end, end=end)

# print(histogram)
