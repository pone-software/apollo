from abc import ABC, abstractmethod
import dgl
import torch as th
import numpy as np

from ..data.events import EventCollection


class AbstractGraphAnnotator(ABC):
    def __init__(self, event_collection: EventCollection) -> None:
        super().__init__()
        self._event_collection = event_collection

    @property
    def _detector(self):
        return self._event_collection.detector

    @abstractmethod
    def annotate_graph(self, graph: dgl.DGLGraph, **kwargs) -> dgl.DGLGraph:
        pass


class DimensionTimelineAnnotator(AbstractGraphAnnotator):
    def annotate_graph(
        self, graph: dgl.DGLGraph, histogram: np.ndarray, **kwargs
    ) -> dgl.DGLGraph:
        graph.ndata["dimensions"] = th.tensor(self._detector.module_coordinates)
        graph.ndata["timeline"] = th.tensor(histogram)
        return graph
