import json
import logging
import os
import uuid

from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from numpy.random import BitGenerator

from ..data.configs import HistogramConfig, HistogramDatasetConfig
from ..data.events import EventCollection
from ..utils.random import get_rng


class AbstractGenerator(ABC):
    def __init__(self, event_collection: EventCollection):
        self.event_collection = event_collection

    @abstractmethod
    def generate(self, save_path):
        raise NotImplementedError("Method 'generate' not implemented")


class AbstractHistogramGenerator(AbstractGenerator, metaclass=ABCMeta):
    def __init__(
        self, event_collection: EventCollection, histogram_config: HistogramConfig
    ):
        super().__init__(event_collection)
        self.histogram_config = histogram_config

    def _generate_histogram(self, event_collection: EventCollection) -> np.ndarray:
        return event_collection.get_histogram(histogram_config=self.histogram_config)

    def save(self, save_path, data: pd.DataFrame):
        is_exist = os.path.exists(save_path)

        if is_exist:
            logging.warning("Folder %s already exists", save_path)
        else:
            os.makedirs(save_path)

        relative_data_path = os.path.join("data")
        absolute_data_path = os.path.join(save_path, relative_data_path)

        is_data_exist = os.path.exists(absolute_data_path)

        if is_data_exist:
            logging.warning("Data folder %s already exists", absolute_data_path)
        else:
            os.makedirs(absolute_data_path)

        index_path = os.path.join(save_path, "index.h5")
        config_path = os.path.join(save_path, "config.json")
        data = data.assign(file="")

        for index, row in data.iterrows():
            filename = os.path.join(
                relative_data_path, "histogram_" + str(uuid.uuid4()) + ".npy"
            )
            np.save(os.path.join(save_path, filename), row["histogram"])
            row["file"] = filename

        data[["events", "file"]].to_feather(index_path)
        histogram_dataset_config = HistogramDatasetConfig(
            path=save_path,
            detector=self.event_collection.detector,
            histogram_config=self.histogram_config,
        )
        with open(config_path, "w") as config_file:
            json.dump(histogram_dataset_config.as_json(), config_file)


class MultiHistogramGenerator(AbstractHistogramGenerator):
    def __init__(
        self,
        event_collection: EventCollection,
        histogram_config: HistogramConfig,
        multi_event_poisson_lambda: Optional[float] = 0.5,
        rng: BitGenerator = None,
    ):
        super().__init__(event_collection, histogram_config=histogram_config)
        if rng is None:
            rng = get_rng()
        self.rng = rng
        self.multi_event_poisson_lambda = multi_event_poisson_lambda

    def __generate_random_multi_event(self):
        return self.rng.poisson(self.multi_event_poisson_lambda)

    def generate(self, save_path):
        number_of_events = len(self.event_collection)

        i = 0
        next_number_of_events = self.__generate_random_multi_event()

        event_collections = []  # type: List[EventCollection]

        while i + next_number_of_events < number_of_events:
            event_collections.append(
                self.event_collection[i : i + next_number_of_events]
            )
            i += next_number_of_events
            next_number_of_events = self.__generate_random_multi_event()

        rows = []

        for event_collection in event_collections:
            self._generate_histogram(event_collection)
            rows.append(
                {
                    "histogram": self._generate_histogram(event_collection),
                    "events": event_collection.get_event_features(valid_only=True),
                }
            )

        self.save(save_path, pd.DataFrame(rows))
