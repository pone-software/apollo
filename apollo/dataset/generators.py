import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import uuid


import pandas as pd
import numpy as np

from numpy.random.bit_generator import BitGenerator

from apollo.dataset.datasets import EventLabels
from apollo.utils.random import get_rng
from olympus.event_generation.data import EventCollection
from olympus.event_generation.generators import GeneratorCollection


class AbstractGenerator(ABC):
    def __init__(self, event_collection: EventCollection):
        self.event_collection = event_collection

    @abstractmethod
    def generate(self, save_path):
        raise NotImplementedError('Method \'generate\' not implemented')


class AbstractHistogramGenerator(AbstractGenerator, ABC):
    def __init__(self,
                 event_collection: EventCollection,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 bin_size: int = 10):
        super().__init__(event_collection)
        self.start_time = start_time
        self.end_time = end_time
        self.bin_size = bin_size
        
    def _generate_histogram(self, event_collection: EventCollection)-> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        return event_collection.generate_histogram(self.bin_size, self.start_time, self.end_time)

    @staticmethod
    def save(save_path, data: pd.DataFrame):
        is_exist = os.path.exists(save_path)

        if is_exist:
            logging.warning('Folder %s already exists', save_path)
        else:
            os.makedirs(save_path)

        data_path = os.path.join(save_path, 'data')

        is_data_exist = os.path.exists(data_path)

        if is_data_exist:
            logging.warning('Data folder %s already exists', data_path)
        else:
            os.makedirs(data_path)


        index_path = os.path.join(save_path, 'index.h5')
        data.assign(file='')

        for index, row in data:
            filename = os.path.join(data_path, 'histogram_' + str(uuid.uuid4()) + '.npy')
            np.save(filename, row['histogram'])
            row['file'] = filename

        data.to_feather(index_path)


class AbstractNoisedHistogramGenerator(AbstractHistogramGenerator, ABC):
    def __init__(self, event_collection: EventCollection,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 bin_size: Optional[int] = 10,
                 noise_generators: Optional[GeneratorCollection] = None):
        super().__init__(event_collection,
                         start_time=start_time,
                         end_time=end_time,
                         bin_size=bin_size)
        self.noise_generators = noise_generators

    def _generate_histogram(self, event_collection: EventCollection)-> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        histogram, records, times = super()._generate_histogram(event_collection)
        if self.noise_generators is not None:
            generated_noise = self.noise_generators.generate_per_timeframe(self.start_time, self.end_time)
            generated_noise_histogram = generated_noise.generate_histogram(self.bin_size, self.start_time, self.end_time)
            histogram += generated_noise_histogram
        
        return histogram, records, times

    
class SingleNoisedHistogramGenerator(AbstractNoisedHistogramGenerator):
    def __init__(self, event_collection: EventCollection,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 bin_size: Optional[int] = 10,
                 histogram_length: Optional[int] = 1000,
                 noise_generators: Optional[GeneratorCollection] = None):
        super().__init__(event_collection,
                         start_time=start_time,
                         end_time=end_time,
                         bin_size=bin_size,
                         noise_generators=noise_generators)
        self.strip_length = histogram_length

    def generate(self, save_path):
        histogram, records, times = self._generate_histogram(self.event_collection)

        records.assign(label=0)
        for label in EventLabels:
            records.loc[records['type'] == label.name, 'label'] = label.value

        records.assign(histogram_index=0)
        records['histogram_index'] = np.floor((records['time'] - times[0]) / self.bin_size)

        overlap = self.strip_length / self.bin_size

        histogram_length = histogram.shape[1]
        number_of_histograms = histogram_length - overlap
        index = np.arange(0, number_of_histograms, 1)

        row_list = []

        for x in index:
            relevant_records = records[(records['histogram_index'] >= x) & (records['histogram_index'] < x + overlap)]
            row_list.append({
                'histogram':histogram[:, x:x+overlap],
                'records': relevant_records
            })

        data = pd.DataFrame(row_list)

        self.save(save_path, data)



class NoisedHistogramGenerator(AbstractNoisedHistogramGenerator):
    def __init__(self, event_collection: EventCollection,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 bin_size: Optional[int] = 10,
                 noise_generators: Optional[GeneratorCollection] = None,
                 signal_to_background_ratio: Optional[float] = 0.3,
                 multi_event_poisson_lambda: Optional[float] = 0.5,
                 rng: BitGenerator = None):
        super().__init__(event_collection,
                         start_time=start_time,
                         end_time=end_time,
                         bin_size=bin_size,
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
            event_collections.append(self.event_collection[i:i+next_number_of_events])
            i += next_number_of_events
            next_number_of_events = self.__generate_random_multi_event()


        final_number_of_events = len(event_collections)

        number_of_histograms = np.ceil(number_of_events / self.signal_to_background_ratio - number_of_events)
