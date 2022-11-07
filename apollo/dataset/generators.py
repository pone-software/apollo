import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional
import uuid

import pandas as pd
import numpy as np

from numpy.random import BitGenerator

from ..utils.random import get_rng
from ..data.events import EventCollection
from ..data.configs import HistogramConfig, HistogramDatasetConfig
from olympus.event_generation.generators import GeneratorCollection


class AbstractGenerator(ABC):
    def __init__(self, event_collection: EventCollection):
        self.event_collection = event_collection

    @abstractmethod
    def generate(self, save_path):
        raise NotImplementedError('Method \'generate\' not implemented')


class AbstractHistogramGenerator(AbstractGenerator, ABC):
    def __init__(self,
                 event_collection: EventCollection, histogram_config: HistogramConfig):
        super().__init__(event_collection)
        self.histogram_config = histogram_config

    def _generate_histogram(self, event_collection: EventCollection) -> np.ndarray:
        return event_collection.get_histogram(histogram_config=self.histogram_config)

    def save(self, save_path, data: pd.DataFrame):
        is_exist = os.path.exists(save_path)

        if is_exist:
            logging.warning('Folder %s already exists', save_path)
        else:
            os.makedirs(save_path)

        relative_data_path = os.path.join('data')
        absolute_data_path = os.path.join(save_path, relative_data_path)

        is_data_exist = os.path.exists(absolute_data_path)

        if is_data_exist:
            logging.warning('Data folder %s already exists', absolute_data_path)
        else:
            os.makedirs(absolute_data_path)

        index_path = os.path.join(save_path, 'index.h5')
        config_path = os.path.join(save_path, 'config.json')
        data = data.assign(file='')

        for index, row in data.iterrows():
            filename = os.path.join(relative_data_path, 'histogram_' + str(uuid.uuid4()) + '.npy')
            np.save(os.path.join(save_path, filename), row['histogram'])
            row['file'] = filename

        data[['events', 'file']].to_feather(index_path)
        histogram_dataset_config = HistogramDatasetConfig(path=save_path, detector=self.event_collection.detector,
                                                          histogram_config=self.histogram_config)
        with open(config_path, 'wb') as config_file:
            json.dump(histogram_dataset_config.as_json(), config_file)


class AbstractNoisedHistogramGenerator(AbstractHistogramGenerator, ABC):
    def __init__(self, event_collection: EventCollection,
                 histogram_config: HistogramConfig,
                 noise_generators: Optional[GeneratorCollection] = None):
        super().__init__(event_collection=event_collection,
                         histogram_config=histogram_config)
        if noise_generators is None:
            noise_generators = GeneratorCollection(event_collection.detector)
        self.noise_generators = noise_generators

    def _generate_histogram(self, event_collection: Optional[EventCollection] = None) -> np.ndarray:
        # TODO: Fix here and remove generator collection
        generated_noise = self.noise_generators.generate_per_timeframe(self.histogram_config.start,
                                                                       self.histogram_config.end)
        histogram = generated_noise.get_histogram(histogram_config=self.histogram_config)

        if event_collection is not None:
            histogram += super()._generate_histogram(event_collection)

        return histogram


class SingleNoisedHistogramGenerator(AbstractNoisedHistogramGenerator):
    def __init__(self, event_collection: EventCollection,
                 histogram_config: HistogramConfig,
                 noise_generators: Optional[GeneratorCollection] = None):
        super().__init__(event_collection,
                         histogram_config=histogram_config,
                         noise_generators=noise_generators)

    def generate(self, save_path):
        histogram = self._generate_histogram(self.event_collection)
        events = self.event_collection.as_json(valid_only=True)

        events.assign(histogram_index=0)
        events['histogram_index'] = np.floor((events['time'] - times[0]) / self.histogram_config.bin_size)

        overlap = self.histogram_config.length / self.histogram_config.bin_size

        histogram_length = histogram.shape[1]
        number_of_histograms = histogram_length - overlap
        index = np.arange(0, number_of_histograms, 1)

        row_list = []

        for x in index:
            relevant_records = events[(events['histogram_index'] >= x) & (events['histogram_index'] < x + overlap)]
            row_list.append({
                'histogram': histogram[:, x:x + overlap],
                'records': relevant_records
            })

        data = pd.DataFrame(row_list)

        self.save(save_path, data)


class NoisedHistogramGenerator(AbstractNoisedHistogramGenerator):
    def __init__(self, event_collection: EventCollection,
                 histogram_config: HistogramConfig,
                 noise_generators: Optional[GeneratorCollection] = None,
                 signal_to_background_ratio: Optional[float] = 0.3,
                 multi_event_poisson_lambda: Optional[float] = 0.5,
                 rng: BitGenerator = None):
        super().__init__(event_collection,
                         histogram_config=histogram_config,
                         noise_generators=noise_generators)
        if rng is None:
            rng = get_rng()
        self.rng = rng
        self.signal_to_background_ratio = signal_to_background_ratio
        self.multi_event_poisson_lambda = multi_event_poisson_lambda

    def __generate_random_multi_event(self):
        return self.rng.poisson(self.multi_event_poisson_lambda)

    def generate(self, save_path):
        number_of_events = len(self.event_collection)

        i = 0
        next_number_of_events = self.__generate_random_multi_event()

        event_collections = []

        while i + next_number_of_events < number_of_events:
            events = self.event_collection.events[i:i + next_number_of_events]
            event_collections.append(
                EventCollection(events, detector=self.event_collection.detector, rng=self.event_collection.rng))
            i += next_number_of_events
            next_number_of_events = self.__generate_random_multi_event()

        final_number_of_events = len(event_collections)

        rows = []

        for event_collection in event_collections:
            self._generate_histogram(event_collection)
            rows.append({
                'histogram': self._generate_histogram(event_collection),
                'events': event_collection.get_event_dicts()
            })

        number_of_histograms = np.ceil(number_of_events / self.signal_to_background_ratio - number_of_events)

        noise_histograms_range = np.arange(0, number_of_histograms - final_number_of_events)

        for x in noise_histograms_range:
            noise_histogram = self._generate_histogram(None)
            rows.append({
                'histogram': noise_histogram,
                'events': []
            })

        self.save(save_path, pd.DataFrame(rows))
