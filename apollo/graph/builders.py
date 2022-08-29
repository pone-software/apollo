from abc import ABC, abstractmethod
import dgl
import networkx as nx

from olympus.event_generation.data import EventCollection


class AbstractGraphBuilder(ABC):
    def __init__(self, event_collection: EventCollection) -> None:
        super().__init__()
        self._event_collection = event_collection
        self._detector = event_collection.detector

    @abstractmethod
    def build_graph(self, **kwargs) -> dgl.DGLGraph:
        pass


class CompleteGraphBuilder(AbstractGraphBuilder):
    def build_graph(self, **kwargs) -> dgl.DGLGraph:
        nx_graph = nx.complete_graph(self._detector.n_modules)
        dgl_graph = dgl.from_networkx(nx_graph)
        return dgl_graph
