from abc import ABC, abstractmethod
import dgl  
import networkx as nx
from typing import Optional, List
from multiprocessing import Pool, cpu_count
import numpy as np
import torch as th
import time
from tqdm import tqdm
from apollo.graph.annotators import DimensionTimelineAnnotator

from olympus.event_generation.data import EventCollection


class AbstractGraphGenerator(ABC):
    def __init__(self, graph: dgl.DGLGraph, event_collection: EventCollection) -> None:
        super().__init__()
        self._event_collection = event_collection
        self._graph = graph

    @abstractmethod
    def generate_graphs(self, **kwargs) -> dgl.DGLGraph:
        pass


class TimeSeriesGraphGenerator(AbstractGraphGenerator):
    def __init__(self, graph: dgl.DGLGraph, event_collection: EventCollection, bin_size: Optional[int] = 50) -> None:
        super().__init__(graph, event_collection)
        self._bin_size = bin_size
        self._graphs = []

    @property
    def graphs(self) -> List[dgl.DGLGraph]:
        return self._graphs

    @property
    def _histogram(self) -> np.ndarray:
        return self._event_collection.generate_histogram(self._bin_size)[0]

    def generate_graphs(
        self,
        step_length: Optional[int] = 10,
        step_size: Optional[int] = 1,
        **kwargs
    ) -> List[dgl.DGLGraph]:
        global get_graph
        global graphs
        histogram = self._histogram
        graphs = []
        time_series_length = histogram.shape[1]
        annotator = DimensionTimelineAnnotator(self._event_collection)
        
        def get_graph(i, base_graph, histogram):
            base_graph = annotator.annotate_graph(base_graph, histogram)

            return (i, base_graph)

        def get_results(result):
            global graphs
            graphs.append(result)

        def get_error(error):
            print(error)

        pool = Pool(cpu_count())
        i = 0

        while i < time_series_length - step_length:
            current_histogram = histogram[:, i:i+step_length]
            edges = self._graph.edges()
            base_graph = dgl.graph((edges[0].clone(), edges[1].clone()))
            # graph = get_graph(i, self, current_histogram)
            # graphs.append(graph)
            pool.apply_async(get_graph, args=(i, base_graph, current_histogram), callback=get_results, error_callback=get_error)
            i += step_size

        pool.close()
        pool.join()

        self._graphs = graphs

        return graphs




class PerEventGraphGenerator(AbstractGraphGenerator):
    def __init__(self, graph: dgl.DGLGraph, event_collection: EventCollection, bin_size: Optional[int] = 50) -> None:
        super().__init__(graph, event_collection)
        self._bin_size = bin_size
        self._graphs = []
        self._per_event_histograms = []

    @property
    def graphs(self) -> List[dgl.DGLGraph]:
        return self._graphs

    def _get_per_event_histograms(self):
        global histograms
        global get_histogram
        histograms = []
        
        def get_histogram(i, event_collection, bin_size):
            histogram = event_collection.generate_histogram(bin_size, event_index=i)

            return (i, histogram)

        
        def get_histogram_results(result):
            global histograms
            histograms.append(result)

        def get_error(error):
            print(error)

        pool = Pool(cpu_count())
        
        for key, event in enumerate(self._event_collection.events):
            pool.apply_async(get_histogram, args=(key, self._event_collection, self._bin_size), callback=get_histogram_results, error_callback=get_error)

        pool.close()
        pool.join()

        self._per_event_histograms = histograms
        return histograms

    def generate_graphs(
        self,
        step_length: Optional[int] = 10,
        step_size: Optional[int] = 1,
        **kwargs
    ) -> List[dgl.DGLGraph]:
        global get_graph
        global graphs
        graphs = []
        annotator = DimensionTimelineAnnotator(self._event_collection)
        
        def get_graph(i, base_graph, histogram):
            base_graph = annotator.annotate_graph(base_graph, histogram)

            return (i, base_graph)

        def get_results(result):
            global graphs
            graphs.append(result)

        def get_error(error):
            print(error)

        def get_results(result):
            global graphs
            graphs.append(result)

        pool = Pool(cpu_count())

        histograms = self._get_per_event_histograms()

        for histogram in tqdm(histograms):
            edges = self._graph.edges()
            base_graph = dgl.graph((edges[0].clone(), edges[1].clone()))
            # graph = get_graph(i, self, current_histogram)
            # graphs.append(graph)
            pool.apply_async(get_graph, args=(i, base_graph, histogram[1]), callback=get_results, error_callback=get_error)

        pool.close()
        pool.join()

        self._graphs = graphs

        return dgl_graph
