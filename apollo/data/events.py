from __future__ import annotations

import copy
import logging

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import awkward as ak
import numpy as np

from apollo.data.configs import HistogramConfig, Interval
from apollo.data.detectors import Detector
from apollo.data.geometric import Vector
from apollo.data.utils import FolderLoadable, FolderSavable, JSONSerializable
from apollo.utils.random import get_rng


class EventType(Enum):
    STARTING_TRACK = auto()
    CASCADE = auto()
    REALISTIC_TRACK = auto()


class SourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = auto()
    ISOTROPIC = auto()


class EventTimeframeMode(Enum):
    # I want the start time be that it is in the interval
    # ..............start....first_hit....last_hit........
    # ...........[..............................]..............
    START_TIME = auto()
    # I want the new start time be that the at least one hit overlaps with the time
    # ....start....first_hit....last_hit........
    # ...........[.................].................
    CONTAINS_HIT = auto()
    # I want the new start time be that all hits overlap with the time
    # ....start....first_hit....last_hit........
    # ...........[..............................]....
    CONTAINS_EVENT = auto()
    # I want the new start time to be that x percent of the event are included
    # I want the new start time be that all hits overlap with the time
    # ....start....first_hit....last_hit........
    # ....................[..............]...........
    CONTAINS_PERCENTAGE = auto()


DEFAULT_EVENT_TIMEFRAME_MODE = EventTimeframeMode.START_TIME


@dataclass
class Record(JSONSerializable):
    direction: Vector
    position: Vector
    time: float

    def as_json(self) -> dict:
        return {
            "direction": self.direction.as_json(),
            "position": self.position.as_json(),
            "time": self.time,
        }

    @classmethod
    def from_json(cls, dictionary: dict) -> Record:
        return cls(
            direction=Vector.from_json(dictionary["direction"]),
            position=Vector.from_json(dictionary["position"]),
            time=dictionary["time"],
        )


@dataclass
class SourceRecord(Record):
    number_of_photons: int
    source_type: SourceType = SourceType.STANDARD_CHERENKOV

    def as_json(self) -> dict:
        parent_dict = super().as_json()
        child_dict = {
            "source_type": self.source_type.name,
            "number_of_photons": self.number_of_photons,
        }

        return {**parent_dict, **child_dict}

    @classmethod
    def from_json(cls, dictionary: dict) -> SourceRecord:
        source_type = dictionary["source_type"]
        if source_type not in SourceType:
            source_type = SourceType[source_type.to_upper()]

        return cls(
            source_type=source_type,
            time=dictionary["time"],
            position=Vector.from_json(dictionary["position"]),
            direction=Vector.from_json(dictionary["direction"]),
            number_of_photons=dictionary["number_of_photons"],
        )


@dataclass
class Event(Record, JSONSerializable):
    event_type: EventType
    energy: float
    hits: List[ak.Array] = None
    sources: Optional[List[SourceRecord]] = None
    detector: Optional[Detector] = None
    default_value: Optional[float] = 0.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = EventType[self.event_type]
        if self.rng is None:
            self.rng = get_rng()

    @property
    def first_hit(self) -> float:
        if self.number_of_hits:
            return ak.min(self.hits)
        return self.default_value

    @property
    def last_hit(self) -> float:
        if self.number_of_hits:
            return ak.max(self.hits)
        return self.default_value

    @property
    def percentile_levels(self) -> np.array:
        start = 0
        end = 110
        step = 10
        return np.arange(start, end, step, dtype=np.int16)

    @property
    def percentiles(self) -> np.ndarray:
        percentile_levels = self.percentile_levels
        percentiles = np.full_like(percentile_levels, self.default_value)
        if self.number_of_hits:
            hits = ak.flatten(self.hits, axis=None)
            percentiles[1:-1] = np.percentile(hits, percentile_levels[1:-1])

        percentiles[0] = self.first_hit
        percentiles[-1] = self.last_hit

        return percentiles

    @property
    def number_of_hits(self) -> int:
        return ak.count(self.hits)

    def as_features(self) -> dict:
        event_dict = {
            **self.as_json(
                include_sources=False, include_hits=False, include_detector=False
            ),
            "number_of_hits": self.number_of_hits,
            "percentiles": list(self.percentiles),
        }
        return event_dict

    def is_in_timeframe(
        self,
        interval: Interval,
        is_in_timeframe_mode: EventTimeframeMode = DEFAULT_EVENT_TIMEFRAME_MODE,
    ) -> bool:
        start = interval.start
        end = interval.end
        if is_in_timeframe_mode == EventTimeframeMode.START_TIME:
            if start is not None and self.time < start:
                return False
            if end is not None and self.time >= end:
                return False
        elif is_in_timeframe_mode == EventTimeframeMode.CONTAINS_HIT:
            if start > self.last_hit:
                return False
            if end < self.first_hit:
                return False
        elif is_in_timeframe_mode == EventTimeframeMode.CONTAINS_EVENT:
            if start >= self.first_hit:
                return False
            if end < self.last_hit:
                return False
        else:
            error_msg = "EventInTimeframeMode {0} not implemented".format(
                is_in_timeframe_mode.name
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        return True

    def redistribute(
        self,
        interval: Interval,
        is_in_timeframe_mode: EventTimeframeMode = DEFAULT_EVENT_TIMEFRAME_MODE,
        rng: np.random.Generator = None,
        percentile: int = 50,
        create_copy: bool = False,
    ) -> Event:
        if create_copy:
            new_event = copy.deepcopy(self)
        else:
            new_event = self

        if rng is None:
            rng = new_event.rng

        modified_start = None
        modified_end = None

        if is_in_timeframe_mode.value == EventTimeframeMode.START_TIME.value:
            modified_start = interval.start
            modified_end = interval.end

        if is_in_timeframe_mode.value == EventTimeframeMode.CONTAINS_HIT.value:
            modified_start = interval.start - new_event.last_hit + new_event.time
            modified_end = interval.end - new_event.first_hit + new_event.time

        if (
            is_in_timeframe_mode.value == EventTimeframeMode.CONTAINS_EVENT.value
            or is_in_timeframe_mode.value
            == EventTimeframeMode.CONTAINS_PERCENTAGE.value
        ):
            if is_in_timeframe_mode.value == EventTimeframeMode.CONTAINS_EVENT.value:
                last_hit = new_event.last_hit
                first_hit = new_event.first_hit
            else:
                percentile_levels = new_event.percentile_levels
                indices = list(percentile_levels)
                if percentile not in indices:
                    err_message = "percentile {0} not available in ({1})".format(
                        str(percentile), percentile_levels.join(", ")
                    )
                    logging.error(err_message)
                    raise ValueError(err_message)
                index = len(indices) - 1 - indices.index(percentile)
                percentiles = new_event.percentiles
                rounded_half = np.ceil(index / 2)
                # fill up from front if uneven
                percentile_start_index = int(index - rounded_half)
                # starting to take of in the end
                percentile_end_index = int(-1 * (rounded_half + 1))
                first_hit = percentiles[percentile_start_index]
                last_hit = percentiles[percentile_end_index]
            event_length = last_hit - first_hit
            event_offset = first_hit - new_event.time
            modified_start = interval.start - event_offset
            modified_end = interval.end - event_offset - event_length

        if modified_start is None or modified_end is None:
            error_msg = "EventInTimeframeMode {0} not implemented".format(
                is_in_timeframe_mode.name
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        if modified_end < modified_start:
            modified_end = modified_start + 1
            error_msg = (
                "Event to long for interval {0}. Switched to contains hit".format(
                    str(interval)
                )
            )
            logging.warning(error_msg)

        new_start_time = rng.integers(modified_start, modified_end)
        difference = new_event.time - new_start_time

        for source in new_event.sources:
            source.time = source.time - difference

        new_event.hits = np.subtract(new_event.hits, difference)

        new_event.time = new_start_time

        return new_event

    def get_histogram(self, histogram_config: HistogramConfig) -> np.ndarray:
        number_of_modules = len(self.detector.modules)
        number_of_bins = histogram_config.number_of_bins
        histogram = np.zeros([number_of_modules, number_of_bins])

        for module_index, module in enumerate(self.hits):
            histogram[module_index] += np.histogram(
                ak.to_numpy(module), bins=number_of_bins, range=histogram_config.range
            )[0]

        return histogram

    def as_json(
        self,
        include_sources: bool = True,
        include_hits: bool = True,
        include_detector: bool = True,
    ) -> dict:
        parent_dict = super().as_json()
        child_dict = {
            "event_type": self.event_type.name,
            "energy": self.energy,
            "default_value": self.default_value,
        }

        if include_sources:
            child_dict["sources"] = [source.as_json() for source in self.sources]

        if include_hits:
            child_dict["hits"] = [list(module_hits) for module_hits in self.hits]

        if include_detector:
            child_dict["detector"] = self.detector.as_json()

        return {**parent_dict, **child_dict}

    @classmethod
    def from_json(cls, dictionary: dict) -> Event:
        sources = None
        detector = None
        hits = None
        event_type = dictionary["event_type"]
        if event_type not in EventType:
            event_type = EventType[event_type.to_upper()]
        if hasattr(dictionary, "sources"):
            sources = [
                SourceRecord.from_json(source) for source in dictionary["sources"]
            ]
        if hasattr(dictionary, "detector"):
            detector = Detector.from_json(dictionary["detector"])
        if hasattr(dictionary, "hits"):
            hits = [ak.Array(module_hits) for module_hits in dictionary["hits"]]

        return cls(
            event_type=event_type,
            sources=sources,
            energy=dictionary["energy"],
            hits=hits,
            detector=detector,
            default_value=dictionary["default_value"],
            time=dictionary["time"],
            position=Vector.from_json(dictionary["position"]),
            direction=Vector.from_json(dictionary["direction"]),
        )


@dataclass
class EventCollection(FolderSavable, FolderLoadable, JSONSerializable):
    """
    Global class for maintaining a common form of Events
    """

    detector: Detector
    events: Optional[List[Event]] = None
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = get_rng()

        if self.events is None:
            self.events = []

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, item):
        events = self.events[item]
        return EventCollection(events=events, detector=self.detector, rng=self.rng)

    def __add__(self, other: EventCollection) -> EventCollection:
        events = self.events + other.events
        return EventCollection(detector=self.detector, events=events, rng=self.rng)

    def __get_reference(self, create_copy: bool) -> EventCollection:
        if create_copy:
            return copy.deepcopy(self)

        return self

    @property
    def valid_events(self) -> List[Event]:
        """
        Property containing only the events with hits for every detector module

        Returns:
            List of events with matching hits and module count

        """

        valid_events = []
        if self.detector is None:
            raise ValueError("detector is not provided.")
        number_of_modules = self.detector.number_of_modules

        for event in self.events:
            if len(event.hits) != number_of_modules:
                continue

            valid_events.append(event)

        return valid_events

    def get_sources_per_bin(self, histogram_config: Optional[HistogramConfig] = None):
        if histogram_config is None:
            histogram_config = HistogramConfig()

        histogram_bins = [
            (x, x + histogram_config.bin_size)
            for x in np.arange(
                histogram_config.start, histogram_config.end, histogram_config.bin_size
            )
        ]
        source_bins = [[] for _ in histogram_bins]

        for event in self.events:
            for source in event.sources:
                if (
                    source.time < histogram_config.start
                    or source.time >= histogram_config.end
                ):
                    continue

                index = int(
                    np.floor(
                        (source.time - histogram_config.start)
                        / histogram_config.bin_size
                    )
                )
                source_bins[index].append(source)

        return source_bins

    def get_histogram(
        self, histogram_config: Optional[HistogramConfig] = None
    ) -> np.ndarray:
        if histogram_config is None:
            histogram_config = HistogramConfig()
        events_to_account = self.valid_events

        if not len(events_to_account):
            logging.warning("No events to generate Histogram")

        total_histogram = np.zeros(
            (self.detector.number_of_modules, histogram_config.number_of_bins)
        )

        for event in events_to_account:
            histogram = event.get_histogram(histogram_config=histogram_config)
            total_histogram = np.add(total_histogram, histogram)

        return total_histogram

    def redistribute(
        self,
        interval: Interval,
        nr_events: int = None,
        is_in_timeframe_mode: EventTimeframeMode = EventTimeframeMode.START_TIME,
        rng: np.random.Generator = None,
        create_copy: bool = False,
        **kwargs,
    ) -> EventCollection:
        logging.info(
            "Start redistribute events (%s, mode %s)",
            str(interval),
            is_in_timeframe_mode.name,
        )

        collection = self.__get_reference(create_copy)
        events_to_account = collection.events

        if nr_events is not None:
            indices = rng.random_integers(0, len(collection), nr_events)
            events_to_account = [collection.events[index] for index in indices]

        if rng is None:
            rng = collection.rng

        for event in events_to_account:
            event.redistribute(
                interval=interval,
                is_in_timeframe_mode=is_in_timeframe_mode,
                rng=rng,
                create_copy=False,
                **kwargs,
            )

        logging.info(
            "Finish redistribute events (%s, mode %s)",
            str(interval),
            is_in_timeframe_mode.name,
        )

        return collection

    def get_within_timeframe(
        self,
        interval: Interval,
        create_copy: bool = False,
        is_in_timeframe_mode: EventTimeframeMode = DEFAULT_EVENT_TIMEFRAME_MODE,
    ) -> EventCollection:
        collection = self.__get_reference(create_copy)

        events_in_timeframe = []

        logging.info(
            "Start collecting events within %s (%d Events available)",
            str(interval),
            len(collection),
        )
        for event in collection.events:
            if event.is_in_timeframe(
                interval=interval, is_in_timeframe_mode=is_in_timeframe_mode
            ):
                events_in_timeframe.append(event)

        logging.info(
            "Finish collecting events within %s (%d Events collected)",
            str(interval),
            len(events_in_timeframe),
        )

        return EventCollection(
            events=events_in_timeframe, detector=collection.detector, rng=collection.rng
        )

    def get_event_features(self, valid_only: bool = True) -> List[dict]:
        if valid_only:
            events = self.valid_events
        else:
            events = self.events

        return [event.as_features() for event in events]

    def get_events_as_json(
        self, valid_only: bool = True, include_sources: bool = False
    ) -> List[dict]:
        if valid_only:
            events = self.valid_events
        else:
            events = self.events
        return [event.as_json(include_sources=include_sources) for event in events]

    def as_json(self, valid_only: bool = True, include_sources: bool = False) -> dict:
        return {
            "events": self.get_events_as_json(
                valid_only=valid_only, include_sources=include_sources
            ),
            "detector": self.detector.as_json(),
        }

    @classmethod
    def from_json(cls, dictionary: dict) -> EventCollection:
        return cls(
            events=[Event.from_json(event) for event in dictionary["events"]],
            detector=Detector.from_json(dictionary["events"]),
        )
