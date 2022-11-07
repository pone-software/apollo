from abc import abstractmethod
from typing import Callable, Optional, Union
import numpy as np
from enum import Enum
import logging
import os

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import math

from ..data.configs import HistogramConfig
from ..utils.random import get_rng
from ..data.events import EventCollection


class AbstractDataset(Dataset):
    @classmethod
    @abstractmethod
    def from_event_collection(cls, event_collection: EventCollection):
        raise NotImplementedError('Method \'from_event_collection\' not implemented in subclass')

    @abstractmethod
    def save(self, folder):
        raise NotImplementedError('Method \'save\' not implemented in subclass')

    @classmethod
    @abstractmethod
    def load(cls, folder: str):
        raise NotImplementedError('Method \'load\' not implemented in subclass')


class MultiHistogramDataset(AbstractDataset):
    def __init__(self,
                 path,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = get_rng()
        self.rng = rng
        self.path = path

    @classmethod
    def from_event_collection(cls, event_collection: EventCollection):
        pass

    def save(self, folder):
        pass

    @classmethod
    def load(cls, folder: str):
        pass

    def __getitem__(self, index) -> T_co:
        pass


class SingleHistogramDataset(AbstractDataset):
    def __init__(self,
                 histogram: np.ndarray,
                 labels: np.ndarray,
                 sequence_length: int = 1,
                 transform_fn: Optional[Callable] = None,
                 transform_opts: Optional[dict] = None):
        if transform_opts is None:
            transform_opts = {}
        if transform_fn is not None:
            histogram = transform_fn(histogram, **transform_opts)
        self.histogram = histogram
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return self.histogram.shape[1] - self.sequence_length

    def __getitem__(self, index) -> T_co:
        seq = self.histogram[:, index:index + self.sequence_length]
        label = self.labels[index:index + self.sequence_length]
        return seq, label

    @classmethod
    def from_event_collection(cls,
                              event_collection: EventCollection,
                              histogram_config: Optional[HistogramConfig] = None,
                              sequence_length: Optional[int] = 1,
                              transform_fn: Optional[Callable] = None,
                              transform_opts: Optional[dict] = None):
        if histogram_config is None:
            histogram_config = HistogramConfig()
        histogram = event_collection.get_histogram(histogram_config=histogram_config)
        records = event_collection.get_events_as_data_frame(include_sources=False, valid_only=True)

        records.assign(label=0)
        for label in EventLabels:
            records.loc[records['type'] == label.name, 'label'] = label.value

        records.assign(histogram_index=0)
        records['histogram_index'] = np.floor((records['time'] - times[0]) / step_size)

        labels = np.zeros((histogram.shape[1], len(EventLabels)))

        for index, row in records.iterrows():
            label = row['label']
            if not math.isnan(label):
                labels[int(row['histogram_index']), int(row['label'])] = 1

        return cls(histogram, labels, sequence_length, transform_fn, transform_opts)

    def save(self, path):
        is_exist = os.path.exists(path)

        if is_exist:
            logging.warning('Folder %s already exists', path)
        else:
            os.makedirs(path)
        np.save(os.path.join(path, 'histogram.npy'), self.histogram)
        np.save(os.path.join(path, 'labels.npy'), self.labels)
        with open(os.path.join(path, 'sequence_length.txt'), 'w') as f:
            f.write(str(self.sequence_length))

    @classmethod
    def load(cls,
             folder,
             transform_fn: Optional[Callable] = None,
             transform_opts: Optional[dict] = None):
        histogram = np.load(os.path.join(folder, 'histogram.npy'))
        labels = np.load(os.path.join(folder, 'labels.npy'))
        with open(os.path.join(folder, 'sequence_length.txt'), 'r') as f:
            line = f.readline()
            sequence_length = int(line)

        return cls(histogram, labels, sequence_length, transform_fn, transform_opts)
