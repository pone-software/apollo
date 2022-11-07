from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import awkward as ak

from apollo.data.detectors import Detector, Module
from apollo.data.events import EventCollection, Event, EventRecord, EventType
from apollo.data.geometric import Vector


class ImporterMeta(ABC):
    """
    Metaclass for importers
    """

    @classmethod
    @abstractmethod
    def from_olympus(cls, input_to_import: Any, **kwargs) -> Any:
        raise NotImplementedError('from_olympus not implemented in importer')


class EventRecordImporter(EventRecord, ImporterMeta):
    @classmethod
    def from_olympus(cls, mc_record: Any, **kwargs) -> EventRecord:
        """
        Loads event from MC Record
        Args:
            mc_record (MCRecord): MCRecord from Olympus to Load

        Returns:
            Event from the apollo package

        """
        info = mc_record.mc_info[0]
        return EventRecord(
            type=EventType[mc_record.event_type.upper()],
            sources=mc_record.sources,
            direction=info["dir"],
            energy=info["energy"],
            time=info["time"],
            position=info["pos"]
        )


class EventCollectionImporter(EventCollection, ImporterMeta):
    """
    Importer for the event collections
    """

    @classmethod
    def from_olympus(
            cls,
            event_collection_tuple: Tuple,
            **kwargs
    ) -> EventCollection:
        """
        loads an event collection from the olympus package

        Args:
            event_collection_tuple: tuple containing hits and events
            **kwargs: parameters to be passed down

        Returns:
            Event collection containing all the events from the input files

        """
        events = []
        hits = event_collection_tuple[0]
        records = event_collection_tuple[1]
        for index in range(0, len(hits)):
            event_hits = hits[index]
            record = records[index]
            if not isinstance(record, EventRecord):
                record = EventRecordImporter.from_olympus(record)
            event = Event(
                hits=event_hits,
                record=record,
                **kwargs
            )
            events.append(event)
        return EventCollection(events=events, **kwargs)

    @classmethod
    def _load_result(cls, input_to_load: Tuple, **kwargs) -> EventCollection:
        """
        Defines the possibilities to be loaded when loading foreign event collection from folder

        Args:
            input_to_load: object to be transformed to event collection
            **kwargs: parameters to be passed down

        Returns:
            event collection read from input

        """
        new_object = None
        if isinstance(input_to_load, Tuple):
            new_object = cls.from_olympus(
                input_to_load,
                **kwargs
            )

        if new_object is None:
            raise ValueError('EventCollectionImporter does not recognize result to import')
        return new_object


class ModuleImporter(Module, ImporterMeta):
    @classmethod
    def from_olympus(cls, module_to_import: Any, **kwargs) -> Module:
        position = Vector(
            x=module_to_import.pos[0],
            y=module_to_import.pos[1],
            z=module_to_import.pos[2]
        )

        return Module(
            position=position,
            noise_rate=module_to_import.noise_rate,
            efficiency=module_to_import.efficiency,
            key=module_to_import.key
        )


class DetectorImporter(Detector, ImporterMeta):
    @classmethod
    def from_olympus(cls, detector_to_import: Any, **kwargs) -> Detector:
        return Detector(
            modules=[ModuleImporter.from_olympus(module) for module in detector_to_import.modules]
        )
